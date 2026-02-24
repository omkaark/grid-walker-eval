from pathlib import Path
from playwright.async_api import Page

GAME_PATH = Path(__file__).parent.parent / "game" / "index.html"

async def setup_game(page: Page, grid_size: int = 8, seed: str = "0", blocks: int = 3) -> None:
    # Load the game
    await page.goto(f"file://{GAME_PATH.resolve()}")
    await page.wait_for_load_state("networkidle")

    # Set grid size
    await page.evaluate(f"BOARD = {grid_size}")

    # Set input fields
    await page.fill("#seed-input", seed)
    await page.fill("#blocks-input", str(blocks))

    # Initialize the game
    await page.evaluate("initGame()")

    # Wait for render
    await page.wait_for_timeout(300)


async def capture_screenshot(page: Page) -> bytes:
    await page.wait_for_function(
        "typeof animating !== 'undefined' && animating === false && animQueue.length === 0",
        timeout=5000
    )
    await page.wait_for_timeout(100)

    container = page.locator("#canvas-container")
    return await container.screenshot(type="png")


async def get_state(page: Page) -> dict:
    return await page.evaluate("""() => ({
        position: { x: state.gridX, z: state.gridZ },
        goal: { x: state.goalX, z: state.goalZ },
        facing: ['north', 'east', 'south', 'west'][state.facing],
        steps: state.totalSteps,
        turns: state.totalTries,
        won: state.won,
        obstacles: Array.from(state.obstacles),
        gridSize: BOARD
    })""")


async def execute_command(page: Page, command: str) -> bool:
    """Execute a command in the game. Returns True if the command was valid."""
    # Escape single quotes in command
    escaped = command.replace("'", "\\'")

    # Execute and check if valid
    success = await page.evaluate(f"parseAndExecute('{escaped}')")

    if success:
        # Wait for animation queue to drain
        await page.wait_for_function(
            "animating === false && animQueue.length === 0",
            timeout=5000
        )
        await page.wait_for_timeout(100)

    return success
