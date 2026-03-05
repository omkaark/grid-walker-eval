import asyncio
import base64
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import async_playwright

from ..common.browser import capture_screenshot, execute_command, get_state, setup_game
from ..common.prompts import SYSTEM_PROMPT
from ..common.vlm import VLMClient, parse_response
from .vllm_utils import ADAPTER_NAME

MODEL_NAME = os.getenv("MODEL", "Qwen/Qwen3-VL-2B-Instruct")
VLLM_BASE_URL = "http://localhost:8000/v1"
GRID_SIZE = 8
N_BLOCKS = 2
MAX_PARALLEL_ROLLOUTS = int(os.getenv("MAX_PARALLEL_ROLLOUTS", 2))

async def _preflight_check_chromium() -> None:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            await browser.close()
    except Exception as exc:
        msg = str(exc)
        if ("Executable doesn't exist" in msg) or ("browserType.launch" in msg):
            raise RuntimeError(
                "Playwright Chromium is not installed. "
                "Run: python -m playwright install chromium"
            ) from exc
        raise RuntimeError(f"Playwright Chromium preflight failed: {msg}") from exc


@dataclass
class RolloutSample:
    request: dict[str, Any]
    turns: list[list[float]]
    seed: int | None = None
    group_id: int | None = None
    won: bool = False
    steps: int = 0
    turns_taken: int = 0
    invalid_count: int = 0
    failed_count: int = 0
    history: list[str] = field(default_factory=list)


class RolloutVLMClient(VLMClient):
    """Rollout client extending shared VLMClient with training-message and logprob support."""

    def __init__(self, use_lora: bool) -> None:
        self.model = ADAPTER_NAME if use_lora else MODEL_NAME
        super().__init__(model=self.model, base_url=VLLM_BASE_URL, api_key="EMPTY")
        self.training_messages: list[dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self.training_messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ]

    @staticmethod
    def _extract_logprobs(choice: Any) -> list[float]:
        logprobs_obj = getattr(choice, "logprobs", None)
        if logprobs_obj is None:
            return []

        # vLLM OpenAI chat format: choice.logprobs.content[].logprob
        token_items = getattr(logprobs_obj, "content", None)
        if not token_items:
            return []

        values: list[float] = []
        for item in token_items:
            lp = getattr(item, "logprob", None)
            if lp is None:
                continue
            values.append(float(lp))
        return values

    async def query(self, screenshot: bytes, turn: int) -> tuple[str, list[float]]:
        b64_image = base64.b64encode(screenshot).decode("ascii")
        # vLLM OpenAI chat renderer expects image_url blocks.
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Turn {turn}. What is your next move?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        )
        # Trainer expects canonical image blocks.
        self.training_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Turn {turn}. What is your next move?"},
                    {"type": "image", "image": f"data:image/png;base64,{b64_image}"},
                ],
            }
        )

        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=self.messages,
            max_tokens=32,
            temperature=0.7,
            top_p=0.95,
            logprobs=True,
        )

        choice = response.choices[0]
        assistant_text = choice.message.content
        token_logprobs = self._extract_logprobs(choice)
        self.messages.append({"role": "assistant", "content": assistant_text})
        self.training_messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        )
        return assistant_text, token_logprobs


def _compute_reward(
    won: bool
) -> float:
    reward = 1.0 if won else 0

    return reward


async def _run_single_rollout(
    rollout_idx: int,
    max_turn_number: int,
    use_lora: bool,
    sem: asyncio.Semaphore,
    seed: int,
    group_id: int | None = None,
) -> tuple[int, RolloutSample, float]:
    async with sem:
        vlm = RolloutVLMClient(use_lora=use_lora)
        vlm.reset()

        history: list[str] = []
        turns_logprobs: list[list[float]] = []
        invalid_count = 0
        failed_count = 0
        turns_taken = 0
        state: dict[str, Any] = {"won": False, "steps": 0}

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_viewport_size({"width": 256, "height": 256})

            try:
                await setup_game(page, grid_size=GRID_SIZE, seed=str(seed), blocks=N_BLOCKS)

                for turn in range(1, max_turn_number + 1):
                    screenshot = await capture_screenshot(page)
                    response, token_logprobs = await vlm.query(screenshot, turn)

                    turns_logprobs.append(token_logprobs)
                    turns_taken += 1

                    command = parse_response(response)
                    if command is None:
                        invalid_count += 1
                        truncated = response[:80] if response else "None"
                        history.append(f"[invalid: {truncated}]")
                    else:
                        success = await execute_command(page, command)
                        if success:
                            history.append(command)
                        else:
                            failed_count += 1
                            history.append(f"[failed: {command}]")

                    state = await get_state(page)
                    if state.get("won"):
                        break
            finally:
                await browser.close()

        won = bool(state.get("won"))
        steps = int(state.get("steps", 0))
        reward = _compute_reward(
            won=won
        )

        sample = RolloutSample(
            request={"messages": vlm.training_messages},
            turns=turns_logprobs,
            seed=seed,
            group_id=group_id,
            won=won,
            steps=steps,
            turns_taken=turns_taken,
            invalid_count=invalid_count,
            failed_count=failed_count,
            history=history,
        )
        return rollout_idx, sample, reward


async def run_rollouts(
    n_rollouts: int,
    max_turn_number: int = 2,
    use_lora: bool = False,
) -> tuple[list[RolloutSample], list[float]]:
    if n_rollouts <= 0:
        return [], []

    group_size = int(os.getenv("GRPO_GROUP_SIZE", "4"))
    if group_size <= 0:
        group_size = 1
    group_size = min(group_size, n_rollouts)

    n_groups = math.ceil(n_rollouts / group_size)
    group_seeds = [random.randint(0, 2_000_000_000) for _ in range(n_groups)]

    await _preflight_check_chromium()
    sem = asyncio.Semaphore(min(MAX_PARALLEL_ROLLOUTS, n_rollouts))
    tasks = [
        _run_single_rollout(
            rollout_idx=i,
            max_turn_number=max_turn_number,
            use_lora=use_lora,
            sem=sem,
            seed=group_seeds[i // group_size],
            group_id=i // group_size,
        )
        for i in range(n_rollouts)
    ]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda item: item[0])

    rollouts = [sample for _, sample, _ in results]
    rewards = [reward for _, _, reward in results]

    finite_rewards = [r for r in rewards if math.isfinite(r)]
    reward_mean = sum(finite_rewards) / len(finite_rewards) if finite_rewards else float("nan")
    wins = sum(1 for r in rollouts if r.won)
    total_turns = sum(r.turns_taken for r in rollouts)
    total_invalid = sum(r.invalid_count for r in rollouts)
    total_failed = sum(r.failed_count for r in rollouts)
    valid_exec = max(0, total_turns - total_invalid - total_failed)
    exec_rate = (valid_exec / total_turns) if total_turns > 0 else 0.0
    print(
        f"Rollout summary: n={len(rollouts)}, wins={wins}, "
        f"win_rate={wins / len(rollouts):.2f}, reward_mean={reward_mean:.3f}, "
        f"turns={total_turns}, invalid={total_invalid}, failed={total_failed}, "
        f"exec_rate={exec_rate:.2f}"
    )
    if total_invalid > 0:
        invalid_examples = [
            h for r in rollouts for h in r.history if h.startswith("[invalid:")
        ][:3]
        if invalid_examples:
            print(f"Invalid examples: {invalid_examples}")
    if total_failed > 0:
        failed_examples = [h for r in rollouts for h in r.history if h.startswith("[failed:")][:3]
        if failed_examples:
            print(f"Failed examples: {failed_examples}")

    return rollouts, rewards
