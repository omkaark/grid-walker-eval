from pathlib import Path
from playwright.async_api import Page

GAME_PATH = Path(__file__).parent.parent / "game" / "index.html"

async def setup_game(page: Page, grid_size: int = 8, seed: str = "0", blocks: int = 3) -> None:
    await page.goto(f"file://{GAME_PATH.resolve()}") # visit game
    await page.wait_for_load_state("networkidle")
    await page.evaluate(f"BOARD = {grid_size}") # set board size
    await page.evaluate(f"ANIM_DUR = 0.001;") # skip animations since human is not playing
    await page.fill("#seed-input", seed) # set seed
    await page.fill("#blocks-input", str(blocks)) # set n blocks
    await page.evaluate("initGame()") # start game
    await page.wait_for_timeout(100)


async def capture_screenshot(page: Page) -> bytes:
    await page.wait_for_function( # wait till animation plays out
        "typeof animating !== 'undefined' && animating === false && animQueue.length === 0",
        timeout=5000
    )
    await page.wait_for_timeout(250)

    container = page.locator("#canvas-container")
    return await container.screenshot(type="png")


async def get_state(page: Page) -> dict:
    return await page.evaluate("""() => ({
        steps: state.totalSteps,
        won: state.won
    })""")


async def execute_command(page: Page, command: str) -> bool:
    success = await page.evaluate(f"parseAndExecute('{command}')")
    if success:
        await page.wait_for_function(
            "animating === false && animQueue.length === 0",
            timeout=5000
        )
        await page.wait_for_timeout(100)
    return success
