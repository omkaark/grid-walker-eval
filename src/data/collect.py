import argparse
import asyncio
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright, Page

from ..common.browser import capture_screenshot, get_state, setup_game


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset"


@dataclass
class SampleRecord:
    sample_id: int
    frame_file: str
    command: str
    command_success: bool
    won_after: bool
    steps_after: int
    tries_after: int
    event_timestamp_ms: int


def _run_dir_name(seed: str, grid_size: int, n_blocks: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_seed = seed.replace("/", "_")
    return f"manual_seed-{safe_seed}_grid-{grid_size}_blocks-{n_blocks}_{ts}"


async def _install_command_hook(page: Page) -> None:
    await page.evaluate(
        """() => {
            if (window.__manualCollectorInstalled) return;
            window.__manualCollectorInstalled = true;
            window.__manualCollectorEvents = [];
            window.__manualCollectorSeq = 0;

            const originalParseAndExecute = window.parseAndExecute;
            window.parseAndExecute = function(raw) {
                const eventTs = Date.now();
                const command = raw == null ? "" : String(raw);
                const success = !!originalParseAndExecute.call(this, raw);
                const st = window.state || {};
                window.__manualCollectorEvents.push({
                    id: ++window.__manualCollectorSeq,
                    command,
                    success,
                    event_timestamp_ms: eventTs,
                    won_after: !!st.won,
                    steps_after: Number(st.totalSteps || 0),
                    tries_after: Number(st.totalTries || 0),
                });
                return success;
            };
        }"""
    )


async def _poll_events(page: Page, processed_count: int) -> list[dict[str, Any]]:
    return await page.evaluate(
        """(startIdx) => {
            const src = window.__manualCollectorEvents || [];
            return src.slice(startIdx);
        }""",
        processed_count,
    )


async def collect_manual_dataset(
    seed: str,
    n_blocks: int,
    grid_size: int,
    n_samples: int,
) -> dict[str, Any]:
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if n_blocks < 0:
        raise ValueError("n_blocks must be >= 0")

    run_dir = DATASET_ROOT / _run_dir_name(seed=seed, grid_size=grid_size, n_blocks=n_blocks)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    samples: list[SampleRecord] = []
    event_count = 0
    closed_reason = "unknown"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.set_viewport_size({"width": 256, "height": 256})
        await setup_game(page, grid_size=grid_size, seed=seed, blocks=n_blocks)
        await _install_command_hook(page)
        await page.bring_to_front()

        print("Manual collection started.")
        print(f"seed={seed} grid_size={grid_size} n_blocks={n_blocks} target_samples={n_samples}")
        print("Use the game UI to play. This window auto-closes on win or when sample target is reached.")

        current_frame = await capture_screenshot(page)

        try:
            while len(samples) < n_samples:
                state = await get_state(page)
                if state.get("won"):
                    closed_reason = "game_won"
                    break

                new_events = await _poll_events(page, event_count)
                if not new_events:
                    await page.wait_for_timeout(50)
                    continue

                for evt in new_events:
                    event_count += 1

                    command = str(evt.get("command", "")).strip()
                    if not command:
                        continue

                    sample_id = len(samples) + 1
                    frame_file = f"frame_{sample_id:06d}.png"
                    frame_path = frames_dir / frame_file
                    frame_path.write_bytes(current_frame)

                    record = SampleRecord(
                        sample_id=sample_id,
                        frame_file=f"frames/{frame_file}",
                        command=command,
                        command_success=bool(evt.get("success", False)),
                        won_after=bool(evt.get("won_after", False)),
                        steps_after=int(evt.get("steps_after", 0)),
                        tries_after=int(evt.get("tries_after", 0)),
                        event_timestamp_ms=int(evt.get("event_timestamp_ms", 0)),
                    )
                    samples.append(record)
                    print(f"[sample {sample_id}/{n_samples}] cmd='{record.command}' success={record.command_success}")

                    # Keep screenshot cadence aligned with eval helper.
                    current_frame = await capture_screenshot(page)

                    if record.won_after:
                        closed_reason = "game_won"
                        break
                    if len(samples) >= n_samples:
                        closed_reason = "sample_target_reached"
                        break

                if closed_reason in {"game_won", "sample_target_reached"}:
                    break
        finally:
            await browser.close()

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "grid_size": grid_size,
        "n_blocks": n_blocks,
        "target_samples": n_samples,
        "collected_samples": len(samples),
        "closed_reason": closed_reason,
        "frame_resolution": {"width": 256, "height": 256},
        "frame_source": "#canvas-container screenshot (eval-equivalent timing)",
    }

    records_payload = [asdict(s) for s in samples]

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "samples.json").write_text(json.dumps(records_payload, indent=2), encoding="utf-8")
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in records_payload:
            f.write(json.dumps(row) + "\n")

    return {
        "run_dir": str(run_dir),
        "metadata": metadata,
        "samples_file": str(run_dir / "samples.json"),
        "samples_jsonl": str(run_dir / "samples.jsonl"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect manual play data (frame + command) for Grid Walker."
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Game seed. If omitted, a random seed in [1, 100_000] is used.",
    )
    parser.add_argument(
        "--n-blocks",
        "--n_blocks",
        dest="n_blocks",
        type=int,
        required=True,
        help="Number of obstacle blocks",
    )
    parser.add_argument("--grid-size", type=int, required=True, help="Grid size")
    parser.add_argument("--samples", type=int, required=True, help="How many samples to collect")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = args.seed if args.seed is not None else str(random.randint(1, 100_000))
    if args.seed is None:
        print(f"No seed provided; using random seed: {seed}")

    result = asyncio.run(
        collect_manual_dataset(
            seed=seed,
            n_blocks=args.n_blocks,
            grid_size=args.grid_size,
            n_samples=args.samples,
        )
    )

    print("\nCollection complete.")
    print(f"Run dir: {result['run_dir']}")
    print(f"Samples: {result['metadata']['collected_samples']}/{result['metadata']['target_samples']}")
    print(f"Closed reason: {result['metadata']['closed_reason']}")
    print(f"Manifest: {result['samples_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
