import asyncio
from dataclasses import dataclass
from enum import Enum

from playwright.async_api import async_playwright, Page

from ..common.browser import setup_game, capture_screenshot, get_state, execute_command
from ..common.vlm import parse_response, VLMClient

@dataclass
class EnvResult:
    grid_size: int
    seed: int
    n_block: int
    max_turns: int

    response_history: list[str]
    turns_taken: int

class EnvEndReason(Enum):
    GAME_WON = 1
    MAX_TURNS = 2

class GridWalkerEnv:
    def __init__(self, vlm_client: VLMClient, grid_size: int, seed: int, n_block: int, max_turns: int = 20):
        self.vlm_client = vlm_client
        self.grid_size = grid_size
        self.seed = seed
        self.n_block = n_block
        self.max_turns = max_turns
        self.response_history: list[str] = []

    async def run(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_viewport_size({ "width": 256, "height": 256 })
            await setup_game(page, self.grid_size, self.seed, self.n_block)

            self.vlm_client.reset()
            state = await get_state(page)
            turn = 0

            while turn < self.max_turns and not state['won']:
                screenshot = await capture_screenshot(page)
                response = self.vlm_client.query(screenshot, turn)
                command = parse_response(response)

                if command is None:
                    truncated = response[:50] if response else "None"
                    self.response_history.append(f"[invalid: {truncated}]")
                    continue

                success = await execute_command(page, command)
                if success:
                    self.response_history.append(command)
                else:
                    self.response_history.append(f"[failed: {command}]")
                
                state = await get_state(page)
                turn += 1

            await browser.close()

        return EnvResult(
            grid_size=self.grid_size,
            seed=self.seed,
            n_block=self.n_block,
            max_turns=self.max_turns,
            response_history=self.response_history,
            turns_taken=turn
        )
