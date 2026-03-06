from pathlib import Path
from playwright.async_api import Page

GAME_PATH = Path(__file__).parent.parent / "game" / "index.html"

async def setup_game(page: Page, grid_size: int = 8, seed: str = "0", blocks: int = 3) -> None:
    try:
        await page.goto(f"file://{GAME_PATH.resolve()}", wait_until="domcontentloaded") # visit game
        await page.wait_for_function(
            """() =>
              !!document.getElementById('seed-input') &&
              !!document.getElementById('blocks-input') &&
              !!document.getElementById('canvas-container') &&
              typeof parseAndExecute === 'function' &&
              typeof state === 'object' &&
              Array.isArray(animQueue) &&
              typeof animating === 'boolean'
            """,
            timeout=5000,
        )
        await page.evaluate(f"BOARD = {grid_size}") # set board size
        await page.evaluate(f"ANIM_DUR = 0.001;") # skip animations since human is not playing
        await page.fill("#seed-input", seed) # set seed
        await page.fill("#blocks-input", str(blocks)) # set n blocks
        await page.evaluate("initGame()") # start game
        # First screenshot can race before RAF render; force one immediate draw.
        await page.evaluate("renderer.render(scene, camera)")
        await page.wait_for_function(
            """() =>
              typeof renderer !== 'undefined' &&
              !!renderer.info &&
              !!renderer.info.render &&
              renderer.info.render.frame > 0 &&
              typeof cameraInitialized !== 'undefined' &&
              cameraInitialized === true
            """,
            timeout=5000,
        )
        await page.wait_for_function(
            """() => {
              const canvas = document.querySelector('#canvas-container canvas');
              if (!canvas || canvas.width === 0 || canvas.height === 0) return false;
              const probe = document.createElement('canvas');
              probe.width = 1;
              probe.height = 1;
              const ctx = probe.getContext('2d', { willReadFrequently: true });
              if (!ctx) return false;
              ctx.drawImage(canvas, 0, 0, 1, 1);
              const d = ctx.getImageData(0, 0, 1, 1).data;
              const opaque = d[3] > 0;
              const nearWhite = d[0] > 245 && d[1] > 245 && d[2] > 245;
              return opaque && !nearWhite;
            }""",
            timeout=5000,
        )
        await page.wait_for_timeout(100)
    except Exception as exc:
        raise RuntimeError(f"setup_game failed during page readiness check: {exc}") from exc


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
