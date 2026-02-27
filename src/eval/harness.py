import os
from dataclasses import dataclass, field
from playwright.async_api import async_playwright, Page

from ..common.browser import setup_game, capture_screenshot, get_state, execute_command
from ..common.vlm import VLMClient, parse_response


@dataclass
class EpisodeResult:
    seed: str
    success: bool
    turns: int
    steps: int
    history: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass
class EvalResults:
    model: str
    grid_size: int
    blocks: int
    max_turns: int
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    @property
    def avg_turns_on_success(self) -> float | None:
        successful = [e for e in self.episodes if e.success]
        if not successful:
            return None
        return sum(e.turns for e in successful) / len(successful)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "config": {
                "grid_size": self.grid_size,
                "blocks": self.blocks,
                "max_turns": self.max_turns
            },
            "episodes": [
                {
                    "seed": e.seed,
                    "success": e.success,
                    "turns": e.turns,
                    "steps": e.steps,
                    "history": e.history,
                    "reason": e.reason
                }
                for e in self.episodes
            ],
            "summary": {
                "success_rate": self.success_rate,
                "total_episodes": len(self.episodes),
                "avg_turns_on_success": self.avg_turns_on_success
            }
        }


async def run_episode(
    page: Page,
    vlm: VLMClient,
    seed: str = "0",
    grid_size: int = 8,
    blocks: int = 3,
    max_turns: int = 50,
    verbose: bool = False,
    log_images_dir: str | None = None
) -> EpisodeResult:
    await setup_game(page, grid_size, seed, blocks)
    vlm.reset()

    history: list[str] = []

    for turn in range(1, max_turns + 1):
        screenshot = await capture_screenshot(page)

        if log_images_dir:
            img_path = os.path.join(log_images_dir, f"step_{turn}.png")
            with open(img_path, "wb") as f:
                f.write(screenshot)

        response = vlm.query(screenshot, turn)

        if verbose:
            print(f"  Turn {turn}: VLM response: {response[:80]}...")

        command = parse_response(response)

        if command is None:
            truncated = response[:50] if response else "None"
            history.append(f"[invalid: {truncated}]")
            if verbose:
                print(f"  Turn {turn}: Invalid response")
            continue

        success = await execute_command(page, command)

        if success:
            history.append(command)
            if verbose:
                print(f"  Turn {turn}: Executed '{command}'")
        else:
            history.append(f"[failed: {command}]")
            if verbose:
                print(f"  Turn {turn}: Command failed '{command}'")

        state = await get_state(page)
        if state["won"]:
            return EpisodeResult(
                seed=seed,
                success=True,
                turns=turn,
                steps=state["steps"],
                history=history,
                reason="game_won"
            )

    state = await get_state(page)
    return EpisodeResult(
        seed=seed,
        success=False,
        turns=max_turns,
        steps=state["steps"],
        history=history,
        reason="max_turns"
    )


async def run_eval(
    model: str,
    api_key: str,
    seeds: list[str],
    grid_size: int = 8,
    blocks: int = 3,
    max_turns: int = 50,
    verbose: bool = False,
    log_images: str | None = None
) -> EvalResults:
    vlm = VLMClient(model=model, api_key=api_key)
    results = EvalResults(
        model=model,
        grid_size=grid_size,
        blocks=blocks,
        max_turns=max_turns
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_viewport_size({ "width": 256, "height": 256 })

        for seed in seeds:
            if verbose:
                print(f"Running seed: {seed}")

            log_images_dir = None
            if log_images:
                log_images_dir = os.path.join("outputs", log_images, f"seed_{seed}")
                os.makedirs(log_images_dir, exist_ok=True)

            episode = await run_episode(
                page=page,
                vlm=vlm,
                seed=seed,
                grid_size=grid_size,
                blocks=blocks,
                max_turns=max_turns,
                verbose=verbose,
                log_images_dir=log_images_dir
            )
            results.episodes.append(episode)

            if verbose:
                status = "SUCCESS" if episode.success else "FAILED"
                print(f"  Result: {status} in {episode.turns} turns, {episode.steps} steps")

        await browser.close()

    return results
