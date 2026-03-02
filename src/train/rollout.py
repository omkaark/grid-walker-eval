import asyncio
import base64
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from playwright.async_api import async_playwright

from ..common.browser import capture_screenshot, execute_command, get_state, setup_game
from ..common.prompts import SYSTEM_PROMPT
from ..common.vlm import parse_response
from .vllm_utils import ADAPTER_NAME

MODEL_NAME = os.getenv("GRID_WALKER_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
VLLM_BASE_URL = "http://localhost:8000/v1"
GRID_SIZE = 8
N_BLOCKS = 2
MAX_PARALLEL_ROLLOUTS = 2
DEBUG_LOGPROBS = os.getenv("GRID_WALKER_DEBUG_LOGPROBS", "1") == "1"


@dataclass
class RolloutSample:
    request: dict[str, Any]
    turns: list[list[float]]
    seed: int | None = None
    won: bool = False
    steps: int = 0
    turns_taken: int = 0
    invalid_count: int = 0
    failed_count: int = 0
    history: list[str] = field(default_factory=list)


class RolloutVLMClient:
    """Chat client for vLLM OpenAI-compatible endpoint with token logprobs."""

    def __init__(self, use_lora: bool) -> None:
        # vLLM generally accepts any non-empty key for local OpenAI-compatible mode.
        self.client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
        self.model = ADAPTER_NAME if use_lora else MODEL_NAME
        self.messages: list[dict[str, Any]] = []
        self.training_messages: list[dict[str, Any]] = []
        self._printed_logprobs_debug = False

    def reset(self) -> None:
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        self.messages = [system_msg]
        self.training_messages = [system_msg]

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "\n".join(parts)
        return ""

    @staticmethod
    def _extract_logprobs(choice: Any) -> list[float]:
        try:
            logprobs_obj = getattr(choice, "logprobs", None)
            if logprobs_obj is None:
                return []

            # OpenAI chat format: choice.logprobs.content[].logprob
            token_items = getattr(logprobs_obj, "content", None)
            if token_items:
                values: list[float] = []
                for item in token_items:
                    lp = getattr(item, "logprob", None)
                    if lp is None and isinstance(item, dict):
                        lp = item.get("logprob")
                    if lp is None:
                        continue
                    try:
                        values.append(float(lp))
                    except (TypeError, ValueError):
                        continue
                if values:
                    return values

            # vLLM compatibility format: choice.logprobs.token_logprobs
            token_logprobs = getattr(logprobs_obj, "token_logprobs", None)
            if token_logprobs is None and isinstance(logprobs_obj, dict):
                token_logprobs = logprobs_obj.get("token_logprobs")
            if token_logprobs:
                values = []
                for lp in token_logprobs:
                    try:
                        values.append(float(lp))
                    except (TypeError, ValueError):
                        continue
                if values:
                    return values

            # Some SDK versions store raw payload in model_extra.
            model_extra = getattr(logprobs_obj, "model_extra", None)
            if isinstance(model_extra, dict):
                token_logprobs = model_extra.get("token_logprobs")
                if token_logprobs:
                    values = []
                    for lp in token_logprobs:
                        try:
                            values.append(float(lp))
                        except (TypeError, ValueError):
                            continue
                    if values:
                        return values

            return []
        except Exception:
            return []

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

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=32,
            temperature=0.0,
            logprobs=True,
        )

        choice = response.choices[0]
        assistant_text = self._extract_text(choice.message.content)
        token_logprobs = self._extract_logprobs(choice)
        if DEBUG_LOGPROBS and not self._printed_logprobs_debug:
            raw_logprobs = getattr(choice, "logprobs", None)
            keys: list[str] = []
            if isinstance(raw_logprobs, dict):
                keys = sorted(raw_logprobs.keys())
            elif raw_logprobs is not None:
                for k in ("content", "token_logprobs", "top_logprobs"):
                    if getattr(raw_logprobs, k, None) is not None:
                        keys.append(k)
            print(
                f"[logprobs_debug] present={raw_logprobs is not None}, "
                f"keys={keys}, extracted_n={len(token_logprobs)}"
            )
            self._printed_logprobs_debug = True
        self.messages.append({"role": "assistant", "content": assistant_text})
        self.training_messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text, token_logprobs


def _compute_reward(
    won: bool,
    turns_taken: int,
    max_turn_number: int,
    invalid_count: int,
    failed_count: int,
) -> float:
    reward = 1.0 if won else -0.5

    # Fewer turns are better when the episode is successful.
    if won and max_turn_number > 0:
        reward += 0.3 * (1.0 - (turns_taken / max_turn_number))

    reward -= 0.10 * invalid_count
    reward -= 0.05 * failed_count

    return max(-1.5, min(1.5, reward))


def _minimal_failed_rollout(seed: int) -> RolloutSample:
    return RolloutSample(
        request={"messages": [{"role": "system", "content": SYSTEM_PROMPT}]},
        turns=[],
        seed=seed,
        won=False,
        steps=0,
        turns_taken=0,
        invalid_count=0,
        failed_count=0,
        history=["[rollout_crash]"],
    )


async def _run_single_rollout(
    rollout_idx: int,
    max_turn_number: int,
    use_lora: bool,
    sem: asyncio.Semaphore,
) -> tuple[int, RolloutSample, float]:
    seed = random.randint(0, 2_000_000_000)

    async with sem:
        try:
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
                        try:
                            response, token_logprobs = await vlm.query(screenshot, turn)
                        except Exception as exc:
                            response = f"[model_error: {type(exc).__name__}]"
                            token_logprobs = []
                            vlm.messages.append({"role": "assistant", "content": response})
                            vlm.training_messages.append(
                                {"role": "assistant", "content": response}
                            )

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
                won=won,
                turns_taken=turns_taken,
                max_turn_number=max_turn_number,
                invalid_count=invalid_count,
                failed_count=failed_count,
            )

            sample = RolloutSample(
                request={"messages": vlm.training_messages},
                turns=turns_logprobs,
                seed=seed,
                won=won,
                steps=steps,
                turns_taken=turns_taken,
                invalid_count=invalid_count,
                failed_count=failed_count,
                history=history,
            )
            return rollout_idx, sample, reward
        except Exception as exc:
            # Keep logs compact; Playwright exceptions can include very long browser dumps.
            summary = str(exc).splitlines()[0][:180]
            print(f"[rollout {rollout_idx}] crashed: {type(exc).__name__}: {summary}")
            sample = _minimal_failed_rollout(seed=seed)
            return rollout_idx, sample, -1.0


async def run_rollouts(
    n_rollouts: int,
    max_turn_number: int = 2,
    use_lora: bool = False,
) -> tuple[list[RolloutSample], list[float]]:
    if n_rollouts <= 0:
        return [], []

    sem = asyncio.Semaphore(min(MAX_PARALLEL_ROLLOUTS, n_rollouts))
    tasks = [
        _run_single_rollout(
            rollout_idx=i,
            max_turn_number=max_turn_number,
            use_lora=use_lora,
            sem=sem,
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
