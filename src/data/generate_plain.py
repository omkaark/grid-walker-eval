import argparse
import asyncio
import json
import random
import shutil
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.async_api import Page, async_playwright

from ..common.browser import capture_screenshot, execute_command, get_state, setup_game


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset"

DIRS = ["north", "east", "south", "west"]
DIR_TO_IDX = {name: idx for idx, name in enumerate(DIRS)}
DELTAS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}


@dataclass
class GeneratedSample:
    sample_id: int
    episode_id: int
    seed: str
    frame_file: str
    command: str
    command_success: bool
    won_after: bool
    steps_after: int
    tries_after: int
    event_timestamp_ms: int


def _run_dir_name(seed: str | None, grid_size: int, n_blocks: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed_tag = seed if seed is not None else "random"
    safe_seed = str(seed_tag).replace("/", "_")
    return f"generated_seed-{safe_seed}_grid-{grid_size}_blocks-{n_blocks}_{ts}"


def _neighbors(x: int, z: int, grid_size: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dx, dz in DELTAS.values():
        nx, nz = x + dx, z + dz
        if nx < 0 or nz < 0 or nx >= grid_size or nz >= grid_size:
            continue
        if (nx, nz) in blocked:
            continue
        out.append((nx, nz))
    return out


def _bfs_shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    grid_size: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]

    q: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in _neighbors(cur[0], cur[1], grid_size, blocked):
            if nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)

    if goal not in parent:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    dx = b[0] - a[0]
    dz = b[1] - a[1]
    if dx == 1 and dz == 0:
        return "east"
    if dx == -1 and dz == 0:
        return "west"
    if dx == 0 and dz == 1:
        return "south"
    if dx == 0 and dz == -1:
        return "north"
    raise ValueError(f"Non-adjacent path step: {a} -> {b}")


def _path_to_commands(path: list[tuple[int, int]], start_facing_idx: int) -> list[str]:
    if len(path) < 2:
        return []

    commands: list[str] = []
    facing = start_facing_idx
    segment_dir: str | None = None
    segment_len = 0

    for i in range(len(path) - 1):
        step_dir = _direction_between(path[i], path[i + 1])
        if segment_dir is None:
            segment_dir = step_dir
            segment_len = 1
            continue
        if step_dir == segment_dir:
            segment_len += 1
            continue

        target_idx = DIR_TO_IDX[segment_dir]
        diff = (target_idx - facing) % 4
        if diff == 3:
            commands.append("left")
        elif diff == 2:
            commands.append("right")
            commands.append("right")
        elif diff == 1:
            commands.append("right")
        facing = target_idx
        commands.append(f"forward {segment_len}")
        segment_dir = step_dir
        segment_len = 1

    if segment_dir is not None:
        target_idx = DIR_TO_IDX[segment_dir]
        diff = (target_idx - facing) % 4
        if diff == 3:
            commands.append("left")
        elif diff == 2:
            commands.append("right")
            commands.append("right")
        elif diff == 1:
            commands.append("right")
        commands.append(f"forward {segment_len}")

    return commands


async def _read_layout(page: Page, grid_size: int) -> dict[str, Any]:
    return await page.evaluate(
        """(boardSize) => {
            const st = (typeof state !== "undefined") ? state : {};
            const obstacles = [];
            if (st.obstacles && typeof st.obstacles.forEach === "function") {
                st.obstacles.forEach((key) => {
                    const [x, z] = String(key).split(",").map((v) => Number(v));
                    obstacles.push([x, z]);
                });
            }
            return {
                grid_size: boardSize,
                start_x: Number(st.gridX || 0),
                start_z: Number(st.gridZ || 0),
                goal_x: Number(st.goalX || 0),
                goal_z: Number(st.goalZ || 0),
                facing_idx: Number(st.facing || 0),
                obstacles,
            };
        }""",
        grid_size,
    )


async def _read_total_tries(page: Page) -> int:
    return int(
        await page.evaluate(
            """() => {
                const st = (typeof state !== "undefined") ? state : {};
                return Number(st.totalTries || 0);
            }"""
        )
    )


def _episode_seed(
    base_seed: str | None,
    episode_idx: int,
    worker_id: int = 0,
    total_workers: int = 1,
) -> str:
    if base_seed is None:
        return str(random.randint(1, 100_000))
    start = int(base_seed) + worker_id
    return str(start + (episode_idx - 1) * total_workers)


def _split_target_games(total_games: int, workers: int) -> list[int]:
    base = total_games // workers
    remainder = total_games % workers
    return [base + (1 if i < remainder else 0) for i in range(workers)]


async def _generate_single_worker(
    seed: str | None,
    n_blocks: int,
    grid_size: int,
    target_games: int,
    max_episodes: int,
    output_dir: Path,
    worker_id: int,
    total_workers: int,
    emit_logs: bool = True,
) -> dict[str, Any]:
    if target_games < 1:
        raise ValueError("target_games must be >= 1")
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if n_blocks < 0:
        raise ValueError("n_blocks must be >= 0")
    if max_episodes < 1:
        raise ValueError("max_episodes must be >= 1")
    if seed is not None:
        int(seed)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    won_games = 0
    samples: list[GeneratedSample] = []
    episodes_summary: list[dict[str, Any]] = []

    prefix = f"[worker {worker_id}]"
    if emit_logs:
        print(
            f"{prefix} start grid_size={grid_size}, n_blocks={n_blocks}, "
            f"target_games={target_games}, max_episodes={max_episodes}"
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_viewport_size({"width": 256, "height": 256})

        try:
            for episode_id in range(1, max_episodes + 1):
                if won_games >= target_games:
                    break

                episode_seed = _episode_seed(
                    base_seed=seed,
                    episode_idx=episode_id,
                    worker_id=worker_id,
                    total_workers=total_workers,
                )
                if emit_logs:
                    print(
                        f"{prefix} [episode {episode_id}] seed={episode_seed} "
                        f"(won_games={won_games}/{target_games}, samples={len(samples)})"
                    )
                await setup_game(page, grid_size=grid_size, seed=episode_seed, blocks=n_blocks)

                layout = await _read_layout(page, grid_size)
                start = (int(layout["start_x"]), int(layout["start_z"]))
                goal = (int(layout["goal_x"]), int(layout["goal_z"]))
                facing_idx = int(layout["facing_idx"])
                blocked = {(int(x), int(z)) for x, z in layout["obstacles"]}

                path = _bfs_shortest_path(start=start, goal=goal, grid_size=grid_size, blocked=blocked)
                if path is None:
                    if emit_logs:
                        print(f"{prefix} [episode {episode_id}] unreachable goal; skipping")
                    episodes_summary.append(
                        {"episode_id": episode_id, "seed": episode_seed, "status": "unreachable_goal"}
                    )
                    continue

                commands = _path_to_commands(path=path, start_facing_idx=facing_idx)
                if not commands:
                    state = await get_state(page)
                    if state.get("won"):
                        if emit_logs:
                            print(f"{prefix} [episode {episode_id}] already won at start")
                        episodes_summary.append(
                            {"episode_id": episode_id, "seed": episode_seed, "status": "already_won"}
                        )
                    else:
                        if emit_logs:
                            print(f"{prefix} [episode {episode_id}] empty command plan; skipping")
                        episodes_summary.append(
                            {
                                "episode_id": episode_id,
                                "seed": episode_seed,
                                "status": "empty_command_plan",
                            }
                        )
                    continue

                planned = len(commands)
                executed = 0
                won = False

                for command in commands:
                    screenshot = await capture_screenshot(page)
                    sample_id = len(samples) + 1
                    frame_file = f"frame_{sample_id:06d}.png"
                    (frames_dir / frame_file).write_bytes(screenshot)

                    command_success = await execute_command(page, command)
                    state_after = await get_state(page)
                    tries_after = await _read_total_tries(page)
                    event_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

                    samples.append(
                        GeneratedSample(
                            sample_id=sample_id,
                            episode_id=episode_id,
                            seed=episode_seed,
                            frame_file=f"frames/{frame_file}",
                            command=command,
                            command_success=command_success,
                            won_after=bool(state_after.get("won", False)),
                            steps_after=int(state_after.get("steps", 0)),
                            tries_after=tries_after,
                            event_timestamp_ms=event_ts,
                        )
                    )
                    executed += 1

                    if emit_logs and (sample_id <= 10 or sample_id % 50 == 0):
                        print(
                            f"{prefix}   [sample {sample_id}] cmd='{command}' "
                            f"ok={command_success} won={bool(state_after.get('won', False))}"
                        )

                    if state_after.get("won"):
                        won = True
                        won_games += 1
                        if emit_logs:
                            print(
                                f"{prefix}   [episode {episode_id}] reached goal "
                                f"(won_games={won_games}/{target_games})"
                            )
                        break

                if emit_logs:
                    print(
                        f"{prefix} [episode {episode_id}] status={'won' if won else 'incomplete'} "
                        f"executed={executed}/{planned} total_samples={len(samples)} "
                        f"won_games={won_games}/{target_games}"
                    )
                episodes_summary.append(
                    {
                        "episode_id": episode_id,
                        "seed": episode_seed,
                        "status": "won" if won else "incomplete",
                        "planned_commands": planned,
                        "executed_commands": executed,
                    }
                )
        finally:
            await browser.close()

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "auto_generate_shortest_path",
        "worker_id": worker_id,
        "total_workers": total_workers,
        "seed": seed,
        "seed_policy": "increment_from_seed_partitioned" if seed is not None else "random_1_to_100000_per_episode",
        "grid_size": grid_size,
        "n_blocks": n_blocks,
        "target_games": target_games,
        "won_games": won_games,
        "collected_samples": len(samples),
        "max_episodes": max_episodes,
        "episodes_ran": len(episodes_summary),
        "frame_resolution": {"width": 256, "height": 256},
        "frame_source": "#canvas-container screenshot (eval-equivalent timing)",
    }

    records = [asdict(s) for s in samples]
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "episodes.json").write_text(json.dumps(episodes_summary, indent=2), encoding="utf-8")
    (output_dir / "samples.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")

    if emit_logs:
        if won_games < target_games:
            print(
                f"{prefix} stopped before target won games were reached: "
                f"{won_games}/{target_games} (max_episodes hit)"
            )
        else:
            print(f"{prefix} target reached: won_games={won_games}/{target_games}, samples={len(samples)}")

    return {
        "run_dir": str(output_dir),
        "metadata": metadata,
        "samples_file": str(output_dir / "samples.json"),
        "episodes_file": str(output_dir / "episodes.json"),
    }


async def _run_worker_subprocess(
    worker_id: int,
    worker_output_dir: Path,
    worker_target_games: int,
    seed: str | None,
    n_blocks: int,
    grid_size: int,
    max_episodes: int,
    total_workers: int,
) -> tuple[int, int]:
    cmd = [
        sys.executable,
        "-m",
        "src.data.generate",
        "--n-blocks",
        str(n_blocks),
        "--grid-size",
        str(grid_size),
        "--samples",
        str(worker_target_games),
        "--max-episodes",
        str(max_episodes),
        "--workers",
        "1",
        "--_worker-output-dir",
        str(worker_output_dir),
        "--_worker-id",
        str(worker_id),
        "--_total-workers",
        str(total_workers),
    ]
    if seed is not None:
        cmd.extend(["--seed", seed])

    proc = await asyncio.create_subprocess_exec(*cmd)
    return_code = await proc.wait()
    return worker_id, return_code


async def _merge_worker_outputs(
    run_dir: Path,
    worker_dirs: list[Path],
    seed: str | None,
    n_blocks: int,
    grid_size: int,
    target_games: int,
    max_episodes: int,
    workers: int,
) -> dict[str, Any]:
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    merged_samples: list[dict[str, Any]] = []
    merged_episodes: list[dict[str, Any]] = []
    total_won_games = 0
    total_episodes_ran = 0

    for worker_id, worker_dir in enumerate(worker_dirs):
        meta_path = worker_dir / "metadata.json"
        samples_path = worker_dir / "samples.json"
        episodes_path = worker_dir / "episodes.json"
        if not meta_path.exists() or not samples_path.exists() or not episodes_path.exists():
            raise RuntimeError(f"Worker {worker_id} output missing in {worker_dir}")

        worker_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        worker_samples = json.loads(samples_path.read_text(encoding="utf-8"))
        worker_episodes = json.loads(episodes_path.read_text(encoding="utf-8"))

        total_won_games += int(worker_meta.get("won_games", 0))
        total_episodes_ran += int(worker_meta.get("episodes_ran", 0))

        episode_offset = len(merged_episodes)
        episode_id_map: dict[int, int] = {}

        for episode in worker_episodes:
            old_episode_id = int(episode.get("episode_id", 0))
            new_episode_id = episode_offset + len(episode_id_map) + 1
            episode_id_map[old_episode_id] = new_episode_id

            copied = dict(episode)
            copied["episode_id"] = new_episode_id
            copied["worker_id"] = worker_id
            merged_episodes.append(copied)

        for sample in worker_samples:
            old_frame_rel = str(sample.get("frame_file", ""))
            old_frame_path = worker_dir / old_frame_rel
            if not old_frame_path.exists():
                raise RuntimeError(f"Missing frame for worker {worker_id}: {old_frame_path}")

            new_sample_id = len(merged_samples) + 1
            new_frame_file = f"frame_{new_sample_id:06d}.png"
            shutil.copy2(old_frame_path, frames_dir / new_frame_file)

            old_episode_id = int(sample.get("episode_id", 0))
            new_episode_id = episode_id_map.get(old_episode_id, old_episode_id)

            copied = dict(sample)
            copied["sample_id"] = new_sample_id
            copied["episode_id"] = new_episode_id
            copied["frame_file"] = f"frames/{new_frame_file}"
            copied["worker_id"] = worker_id
            merged_samples.append(copied)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "auto_generate_shortest_path",
        "seed": seed,
        "seed_policy": "increment_from_seed_partitioned" if seed is not None else "random_1_to_100000_per_episode",
        "grid_size": grid_size,
        "n_blocks": n_blocks,
        "target_games": target_games,
        "won_games": total_won_games,
        "collected_samples": len(merged_samples),
        "max_episodes": max_episodes,
        "episodes_ran": total_episodes_ran,
        "workers": workers,
        "frame_resolution": {"width": 256, "height": 256},
        "frame_source": "#canvas-container screenshot (eval-equivalent timing)",
    }

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "episodes.json").write_text(json.dumps(merged_episodes, indent=2), encoding="utf-8")
    (run_dir / "samples.json").write_text(json.dumps(merged_samples, indent=2), encoding="utf-8")
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in merged_samples:
            f.write(json.dumps(row) + "\n")

    return {
        "run_dir": str(run_dir),
        "metadata": metadata,
        "samples_file": str(run_dir / "samples.json"),
        "episodes_file": str(run_dir / "episodes.json"),
    }


async def generate_dataset(
    seed: str | None,
    n_blocks: int,
    grid_size: int,
    n_samples: int,
    max_episodes: int,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be >= 1")

    run_dir = DATASET_ROOT / _run_dir_name(seed=seed, grid_size=grid_size, n_blocks=n_blocks)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Starting automated generation: "
        f"grid_size={grid_size}, n_blocks={n_blocks}, target_games={n_samples}, "
        f"max_episodes={max_episodes}, workers={workers}, seed_policy="
        f"{'increment_from_seed_partitioned' if seed is not None else 'random_1_to_100000_per_episode'}"
    )
    print(f"Output directory: {run_dir}")

    if workers == 1:
        return await _generate_single_worker(
            seed=seed,
            n_blocks=n_blocks,
            grid_size=grid_size,
            target_games=n_samples,
            max_episodes=max_episodes,
            output_dir=run_dir,
            worker_id=0,
            total_workers=1,
            emit_logs=True,
        )

    worker_targets = _split_target_games(n_samples, workers)
    worker_dirs = [run_dir / "workers" / f"worker_{idx:02d}" for idx in range(workers)]
    for worker_dir in worker_dirs:
        worker_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[asyncio.Task[tuple[int, int]]] = []
    for idx in range(workers):
        if worker_targets[idx] <= 0:
            continue
        print(
            f"Launching worker {idx}: target_games={worker_targets[idx]}, "
            f"output={worker_dirs[idx]}"
        )
        tasks.append(
            asyncio.create_task(
                _run_worker_subprocess(
                    worker_id=idx,
                    worker_output_dir=worker_dirs[idx],
                    worker_target_games=worker_targets[idx],
                    seed=seed,
                    n_blocks=n_blocks,
                    grid_size=grid_size,
                    max_episodes=max_episodes,
                    total_workers=workers,
                )
            )
        )

    worker_failures: list[tuple[int, int]] = []
    for task in asyncio.as_completed(tasks):
        worker_id, return_code = await task
        if return_code == 0:
            print(f"Worker {worker_id} finished successfully")
        else:
            print(f"Worker {worker_id} failed with exit code {return_code}")
            worker_failures.append((worker_id, return_code))

    if worker_failures:
        raise RuntimeError(f"One or more workers failed: {worker_failures}")

    merged = await _merge_worker_outputs(
        run_dir=run_dir,
        worker_dirs=worker_dirs,
        seed=seed,
        n_blocks=n_blocks,
        grid_size=grid_size,
        target_games=n_samples,
        max_episodes=max_episodes,
        workers=workers,
    )

    if merged["metadata"]["won_games"] < n_samples:
        print(
            "Generation stopped before target won games were reached: "
            f"{merged['metadata']['won_games']}/{n_samples}"
        )
    else:
        print(
            f"Target reached: won_games={merged['metadata']['won_games']}/{n_samples}, "
            f"samples={merged['metadata']['collected_samples']}"
        )

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate automated Grid Walker dataset using shortest-path play."
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Optional base seed. If omitted, each episode uses random seed in [1, 100_000].",
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
    parser.add_argument(
        "--samples",
        type=int,
        required=True,
        help="How many completed games to generate (won=True)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=5000,
        help="Max episodes per worker while trying to reach --samples won games",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to run in parallel",
    )

    parser.add_argument("--_worker-output-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_total-workers", type=int, default=1, help=argparse.SUPPRESS)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args._worker_output_dir is not None:
        result = asyncio.run(
            _generate_single_worker(
                seed=args.seed,
                n_blocks=args.n_blocks,
                grid_size=args.grid_size,
                target_games=args.samples,
                max_episodes=args.max_episodes,
                output_dir=Path(args._worker_output_dir),
                worker_id=args._worker_id,
                total_workers=args._total_workers,
                emit_logs=True,
            )
        )
    else:
        result = asyncio.run(
            generate_dataset(
                seed=args.seed,
                n_blocks=args.n_blocks,
                grid_size=args.grid_size,
                n_samples=args.samples,
                max_episodes=args.max_episodes,
                workers=args.workers,
            )
        )

    print("Generation complete.")
    print(f"Run dir: {result['run_dir']}")
    print(
        f"Won games: {result['metadata']['won_games']}/{result['metadata']['target_games']} "
        f"(episodes ran: {result['metadata']['episodes_ran']}, "
        f"samples: {result['metadata']['collected_samples']})"
    )
    print(f"Samples manifest: {result['samples_file']}")
    print(f"Episodes summary: {result['episodes_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
