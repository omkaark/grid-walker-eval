import argparse
import asyncio
import base64
import json
import os
import random
import re
import shutil
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright

from ..common.browser import capture_screenshot, execute_command, get_state, setup_game


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset"

DIRS = ["north", "east", "south", "west"]
DIR_TO_IDX = {name: idx for idx, name in enumerate(DIRS)}
DELTAS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
DEFAULT_REASON_MODEL = os.getenv("REASON_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
DEFAULT_REASON_BASE_URL = os.getenv("REASON_BASE_URL", "http://localhost:8000/v1")
DEFAULT_REASON_API_KEY = os.getenv("REASON_API_KEY", "EMPTY")
DEFAULT_VISIBILITY_MODEL = os.getenv("VISIBILITY_MODEL", DEFAULT_REASON_MODEL)
DEFAULT_VISIBILITY_BASE_URL = os.getenv("VISIBILITY_BASE_URL", DEFAULT_REASON_BASE_URL)
DEFAULT_VISIBILITY_API_KEY = os.getenv("VISIBILITY_API_KEY", DEFAULT_REASON_API_KEY)
REASON_BANNED_WORDS = re.compile(
    r"\b(north|south|east|west|northeast|northwest|southeast|southwest|coordinate|grid|x|z)\b",
    re.IGNORECASE,
)
REASONING_SYSTEM_PROMPT = """You generate short reasoning traces for Grid Walker.

Game context:
- This is a third-person 3D grid navigation game.
- The player sees only the camera view where the camera is placed behind them.
- Goal: step onto the GOLD square to win.
- Obstacles/walls block movement; you cannot move through them.
- Available commands are `left`, `right`, and `forward N`.
- `left` and `right` rotate in place by 90 degrees.
- `forward N` moves straight ahead N cells if the route is clear.
- If the goal is not visible, exploration by turning or moving is expected.

Your task:
- You will receive an image and the already-chosen command.
- Produce a brief reason that supports that command from visible evidence only.

Output rules (strict):
- Output exactly one <think>...</think> block and nothing else.
- Keep it very short: 5-12 words.
- Do not use compass words (north/south/east/west), coordinates, or hidden-map claims.
- Mention only observable cues (goal visible, blocked path, clear lane, exploration).
- Make the reason align with the provided command.
- Do not copy examples exactly; write a similar fresh line.

Style examples (do not necessarily repeat verbatim):
- Gold not visible, so I will explore left.
- Gold is front-right, so I turn right.
- Exploration worked, now I go straight.
- Gold is in front, so I move forward.
- Gold seems in front, so I keep moving.
- Gold is slightly left, turn to face it.
- Gold sits to the right, so I reorient.
- Gold not visible, trying the right side.
- Gold is not visible, so I turn left to scan.
- Front is blocked, rotating right to route around.
- Gold is front-left, turn then advance.
- Gold is front-right, aligning before moving.
- Gold stays in front, so I step forward.
- Gold remains in view, so I move forward again.
- Gold is still in front, continue forward.
- Exploration worked, gold is front-left, turn left.
- Exploration worked, gold is front-right, turn right.
- Exploration worked, now pushing straight ahead.
- Gold appears nearer, taking another forward burst.
- Exploration worked, goal is left, turning left.
- Exploration worked, goal is right, turning right.
"""
VISIBILITY_SYSTEM_PROMPT = """You are a binary visibility checker for Grid Walker.

Task:
- Look at the image and decide if the gold target tile is visible.

Output format (strict):
- Reply with exactly one token: VISIBLE or NOT_VISIBLE
- No punctuation
- No explanation

If uncertain, reply NOT_VISIBLE.
"""


@dataclass
class GeneratedSample:
    sample_id: int
    episode_id: int
    seed: str
    frame_file: str
    reason_text: str
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
    return f"generated_reason_seed-{safe_seed}_grid-{grid_size}_blocks-{n_blocks}_{ts}"


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
        commands.extend(_split_forward_segment(segment_len))
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
        commands.extend(_split_forward_segment(segment_len))

    return commands


def _split_forward_segment(length: int) -> list[str]:
    if length <= 0:
        return []
    if length == 1:
        return ["forward 1"]

    out: list[str] = []
    remaining = length
    while remaining > 0:
        if remaining <= 2:
            step = 1 if remaining == 1 else random.choices([1, 2], weights=[1, 3], k=1)[0]
        elif remaining == 3:
            step = random.choices([1, 2, 3], weights=[1, 2, 3], k=1)[0]
        else:
            # Prefer segmented bursts while still mixing in atomic moves.
            step = random.choices([1, 2, 3, 4], weights=[1, 2, 3, 5], k=1)[0]
        step = min(step, remaining)
        out.append(f"forward {step}")
        remaining -= step
    return out


def _fallback_reason(command: str, visibility: str, explore_mode: str | None) -> str:
    if visibility == "NOT_VISIBLE":
        # For hidden-goal frames, keep exploration behavior with 50/50 style randomness.
        mode = explore_mode if explore_mode in {"deterministic", "opposite"} else random.choice(
            ["deterministic", "opposite"]
        )
        if mode == "opposite":
            if command == "left":
                return random.choice(
                    [
                        "<think>Gold not visible, so I keep scanning left.</think>",
                        "<think>Gold not visible, turning left to scan.</think>",
                        "<think>Still scanning left to spot the gold.</think>",
                    ]
                )
            if command == "right":
                return random.choice(
                    [
                        "<think>Gold not visible, so I keep scanning right.</think>",
                        "<think>Gold not visible, turning right to scan.</think>",
                        "<think>Still scanning right to find the gold.</think>",
                    ]
                )
            return random.choice(
                [
                    "<think>Gold not visible, so I keep scanning.</think>",
                    "<think>Gold not visible, continuing to scan around.</think>",
                ]
            )
        if command == "left":
            return random.choice(
                [
                    "<think>Gold not visible, so I explore left.</think>",
                    "<think>Gold not visible, trying the left side.</think>",
                    "<think>Gold is not visible, so I turn left to scan.</think>",
                ]
            )
        if command == "right":
            return random.choice(
                [
                    "<think>Gold not visible, so I explore right.</think>",
                    "<think>Gold not visible, trying the right side.</think>",
                    "<think>Gold is not visible, so I turn right to scan.</think>",
                ]
            )
        return random.choice(
            [
                "<think>Gold not visible, so I keep turning to scan.</think>",
                "<think>Gold not visible, scanning before moving.</think>",
            ]
        )

    if command == "left":
        return random.choice(
            [
                "<think>Gold seems left, so I turn left.</think>",
                "<think>Gold is slightly left, turn to face it.</think>",
                "<think>Gold is front-left, turn then advance.</think>",
                "<think>Exploration worked, gold is front-left, turn left.</think>",
            ]
        )
    if command == "right":
        return random.choice(
            [
                "<think>Gold seems right, so I turn right.</think>",
                "<think>Gold sits to the right, so I reorient.</think>",
                "<think>Gold is front-right, aligning before moving.</think>",
                "<think>Exploration worked, gold is front-right, turn right.</think>",
            ]
        )
    return random.choice(
        [
            "<think>Gold is in front, so I move forward.</think>",
            "<think>Gold seems in front, so I keep moving.</think>",
            "<think>Gold stays in front, so I step forward.</think>",
            "<think>Gold remains in view, so I move forward again.</think>",
            "<think>Gold is still in front, continue forward.</think>",
        ]
    )


def _sanitize_think(text: str) -> str | None:
    if not text:
        return None

    match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        inner = match.group(1)
    else:
        # Relaxed parse: accept plain short text and wrap it into <think>...</think>.
        inner = text

    inner = re.sub(r"`[^`]*`", " ", inner)
    inner = re.sub(r"</?think>", " ", inner, flags=re.IGNORECASE)
    inner = re.sub(r"\s+", " ", inner).strip(" \n\r\t.,;:!?")
    if not inner:
        return None
    if REASON_BANNED_WORDS.search(inner):
        return None

    words = inner.split(" ")
    if len(words) < 3:
        return None
    if len(words) > 12:
        inner = " ".join(words[:12]).strip(" .,;:")
        if not inner:
            return None

    return f"<think>{inner}</think>"


class ReasoningVLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout_sec: float,
        max_retries: int,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec)

    async def _repair_reason(
        self,
        raw_text: str,
        command: str,
        visibility: str,
    ) -> str | None:
        if not raw_text:
            return None
        constraint = (
            "Gold is not visible; mention scanning/turning."
            if visibility == "NOT_VISIBLE"
            else "Gold is visible; mention moving/alignment."
        )
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Rewrite text into exactly one <think>...</think> reason.\n"
                            "5-12 words, short and concrete, no compass/coordinates."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Command: `{command}`. {constraint}\n"
                            f"Text: {raw_text}\n"
                            "Return only <think>...</think>."
                        ),
                    }
                ],
            },
        ]
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=40,
                temperature=0.3,
                top_p=0.9,
            )
            repaired = response.choices[0].message.content or ""
            return _sanitize_think(repaired)
        except Exception:
            return None

    async def generate_reason(
        self,
        screenshot: bytes,
        command: str,
        turn: int,
        visibility: str,
        explore_mode: str | None,
    ) -> str:
        b64_image = base64.b64encode(screenshot).decode("ascii")
        if visibility == "NOT_VISIBLE":
            visibility_hint = (
                "Gold is not visible in this frame. Mention scanning/turning briefly. "
                "Do not claim gold position in view. Do not describe moving forward."
            )
        else:
            visibility_hint = "Gold is visible in this frame. Briefly justify alignment or moving toward it."
        user_prompt = (
            f"Turn {turn}. Chosen command is `{command}`.\n"
            f"{visibility_hint}\n"
            "Write one short reason in <think>...</think> that supports this command."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": REASONING_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            },
        ]
        for _ in range(self.max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    max_tokens=48,
                    temperature=1.0 if visibility == "VISIBLE" else 0.8,
                    top_p=0.9,
                )
                raw = response.choices[0].message.content or ""
                cleaned = _sanitize_think(raw)
                if cleaned is not None:
                    return cleaned
                repaired = await self._repair_reason(
                    raw_text=raw,
                    command=command,
                    visibility=visibility,
                )
                if repaired is not None:
                    return repaired
            except Exception:
                continue
        return _fallback_reason(command=command, visibility=visibility, explore_mode=explore_mode)


def _normalize_visibility_label(text: str) -> str:
    if not text:
        return "NOT_VISIBLE"
    norm = re.sub(r"[^A-Z_ ]+", " ", text.upper())
    squashed = re.sub(r"\s+", " ", norm).strip()
    if "NOT VISIBLE" in squashed or "NOT_VISIBLE" in squashed or "NOTVISIBLE" in squashed:
        return "NOT_VISIBLE"
    if squashed == "VISIBLE" or " VISIBLE" in f" {squashed}":
        return "VISIBLE"
    return "NOT_VISIBLE"


class VisibilityVLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout_sec: float,
        max_retries: int,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec)

    async def classify_visibility(self, screenshot: bytes, turn: int) -> str:
        b64_image = base64.b64encode(screenshot).decode("ascii")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": VISIBILITY_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Turn {turn}. Is the gold tile visible?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            },
        ]
        for _ in range(self.max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    max_tokens=8,
                    temperature=0.2,
                    top_p=0.95,
                )
                raw = (response.choices[0].message.content or "").strip()
                label = _normalize_visibility_label(raw)
                if label in {"VISIBLE", "NOT_VISIBLE"}:
                    return label
            except Exception:
                continue
        return "NOT_VISIBLE"


def _opposite_turn(direction: str) -> str:
    return "right" if direction == "left" else "left"


def _is_opposite_turn(a: str, b: str) -> bool:
    if a not in {"left", "right"} or b not in {"left", "right"}:
        return False
    return _opposite_turn(a) == b


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


async def _read_pose(page: Page) -> tuple[int, int, int]:
    x, z, facing = await page.evaluate(
        """() => {
            const st = (typeof state !== "undefined") ? state : {};
            return [Number(st.gridX || 0), Number(st.gridZ || 0), Number(st.facing || 0)];
        }"""
    )
    return int(x), int(z), int(facing)


async def _heuristic_gold_visibility(page: Page) -> str:
    visible = await page.evaluate(
        """() => {
            const canvas = document.querySelector('#canvas-container canvas');
            if (!canvas || canvas.width === 0 || canvas.height === 0) return false;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            if (!ctx) return false;
            const w = canvas.width;
            const h = canvas.height;
            const img = ctx.getImageData(0, 0, w, h).data;

            // Gold tile in scene uses ~#ffd700 with emissive glow; this accepts nearby shades.
            let count = 0;
            const step = 3;
            for (let y = 0; y < h; y += step) {
              for (let x = 0; x < w; x += step) {
                const i = (y * w + x) * 4;
                const r = img[i];
                const g = img[i + 1];
                const b = img[i + 2];
                const a = img[i + 3];
                if (a < 200) continue;
                const isGold = r >= 175 && g >= 120 && b <= 140 && (r - b) >= 70 && (g - b) >= 30;
                if (isGold) count += 1;
              }
            }
            return count >= 12;
        }"""
    )
    return "VISIBLE" if bool(visible) else "NOT_VISIBLE"


def _aligned_turn_from_pose(
    cur_x: int,
    cur_z: int,
    facing_idx: int,
    goal: tuple[int, int],
    grid_size: int,
    blocked: set[tuple[int, int]],
    planner_command: str,
) -> str:
    path = _bfs_shortest_path(start=(cur_x, cur_z), goal=goal, grid_size=grid_size, blocked=blocked)
    if path is None or len(path) < 2:
        if planner_command in {"left", "right"}:
            return planner_command
        return random.choice(["left", "right"])

    next_dir = _direction_between(path[0], path[1])
    target_idx = DIR_TO_IDX[next_dir]
    diff = (target_idx - facing_idx) % 4
    if diff == 3:
        return "left"
    if diff == 1:
        return "right"
    if diff == 2:
        return random.choice(["left", "right"])

    # Already facing ideal heading; keep scanning with a small turn.
    return random.choice(["left", "right"])


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


def _write_samples_snapshot(base_dir: Path, records: list[dict[str, Any]], pct: int) -> None:
    snap_json = base_dir / f"samples.partial.{pct:03d}.json"
    snap_jsonl = base_dir / f"samples.partial.{pct:03d}.jsonl"
    snap_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    with snap_jsonl.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")


async def _generate_single_worker(
    seed: str | None,
    n_blocks: int,
    grid_size: int,
    target_games: int,
    max_episodes: int,
    output_dir: Path,
    worker_id: int,
    total_workers: int,
    reason_model: str,
    reason_base_url: str,
    reason_api_key: str,
    reason_timeout_sec: float,
    reason_max_retries: int,
    visibility_model: str,
    visibility_base_url: str,
    visibility_api_key: str,
    visibility_timeout_sec: float,
    visibility_max_retries: int,
    verbose: bool = False,
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
    next_progress_pct = 10

    prefix = f"[worker {worker_id}]"
    if emit_logs:
        print(
            f"{prefix} start grid_size={grid_size}, n_blocks={n_blocks}, "
            f"target_games={target_games}, max_episodes={max_episodes}"
        )

    reason_client = ReasoningVLMClient(
        model=reason_model,
        base_url=reason_base_url,
        api_key=reason_api_key,
        timeout_sec=reason_timeout_sec,
        max_retries=reason_max_retries,
    )
    visibility_client = VisibilityVLMClient(
        model=visibility_model,
        base_url=visibility_base_url,
        api_key=visibility_api_key,
        timeout_sec=visibility_timeout_sec,
        max_retries=visibility_max_retries,
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
                episode_log_lines: list[str] = []
                episode_header = (
                    f"{prefix} [episode {episode_id}] seed={episode_seed} "
                    f"(won_games={won_games}/{target_games}, samples={len(samples)})"
                )
                try:
                    await setup_game(page, grid_size=grid_size, seed=episode_seed, blocks=n_blocks)

                    layout = await _read_layout(page, grid_size)
                    start = (int(layout["start_x"]), int(layout["start_z"]))
                    goal = (int(layout["goal_x"]), int(layout["goal_z"]))
                    facing_idx = int(layout["facing_idx"])
                    blocked = {(int(x), int(z)) for x, z in layout["obstacles"]}

                    path = _bfs_shortest_path(start=start, goal=goal, grid_size=grid_size, blocked=blocked)
                    if path is None:
                        episodes_summary.append(
                            {"episode_id": episode_id, "seed": episode_seed, "status": "unreachable_goal"}
                        )
                        continue

                    commands = _path_to_commands(path=path, start_facing_idx=facing_idx)
                    if not commands:
                        state = await get_state(page)
                        if state.get("won"):
                            episodes_summary.append(
                                {"episode_id": episode_id, "seed": episode_seed, "status": "already_won"}
                            )
                        else:
                            episodes_summary.append(
                                {
                                    "episode_id": episode_id,
                                    "seed": episode_seed,
                                    "status": "empty_command_plan",
                                }
                            )
                        continue

                    planned_commands = list(commands)
                    planned = len(planned_commands)
                    executed = 0
                    won = False
                    episode_pending: list[dict[str, Any]] = []
                    planner_idx = 0
                    max_extra_explore = max(20, planned * 2)
                    extra_explore_steps = 0
                    explore_active = False
                    explore_mode: str | None = None
                    pending_opposite_correction = False
                    explore_turn_streak = 0
                    turn_no_move_streak = 0
                    last_turn_command: str | None = None

                    while planner_idx < planned and not won:
                        planner_command = planned_commands[planner_idx]
                        screenshot = await capture_screenshot(page)
                        preview_sample_id = len(samples) + len(episode_pending) + 1
                        model_visibility = await visibility_client.classify_visibility(
                            screenshot=screenshot,
                            turn=preview_sample_id,
                        )
                        heuristic_visibility = await _heuristic_gold_visibility(page)
                        visibility_label = (
                            "NOT_VISIBLE"
                            if model_visibility == "NOT_VISIBLE" and heuristic_visibility == "NOT_VISIBLE"
                            else "VISIBLE"
                        )
                        cur_x, cur_z, cur_facing = await _read_pose(page)
                        command = planner_command

                        if visibility_label == "VISIBLE":
                            if explore_active:
                                explore_active = False
                                explore_mode = None
                                pending_opposite_correction = False
                                explore_turn_streak = 0
                            planner_idx += 1
                        else:
                            aligned_turn = _aligned_turn_from_pose(
                                cur_x=cur_x,
                                cur_z=cur_z,
                                facing_idx=cur_facing,
                                goal=goal,
                                grid_size=grid_size,
                                blocked=blocked,
                                planner_command=planner_command,
                            )
                            if extra_explore_steps >= max_extra_explore:
                                planner_idx += 1
                                command = aligned_turn
                                explore_active = True
                                explore_mode = "deterministic"
                                pending_opposite_correction = False
                                explore_turn_streak = 1
                            else:
                                explore_active = True
                                if turn_no_move_streak >= 3:
                                    # Same-state turn cap: break oscillation while staying turn-only.
                                    command = aligned_turn
                                    explore_mode = "deterministic"
                                    pending_opposite_correction = False
                                elif explore_turn_streak >= 2:
                                    # Exploration streak limit: stop random turns and lock to aligned turn.
                                    command = aligned_turn
                                    explore_mode = "deterministic"
                                    pending_opposite_correction = False
                                elif pending_opposite_correction:
                                    command = aligned_turn
                                    explore_mode = "deterministic"
                                    pending_opposite_correction = False
                                else:
                                    if random.random() < 0.5:
                                        command = aligned_turn
                                        explore_mode = "deterministic"
                                    else:
                                        command = _opposite_turn(aligned_turn)
                                        explore_mode = "opposite"
                                        pending_opposite_correction = True
                                extra_explore_steps += 1

                        # Hard anti-oscillation: zero probability of immediate opposite turn.
                        if (
                            visibility_label == "NOT_VISIBLE"
                            and explore_active
                            and
                            command in {"left", "right"}
                            and last_turn_command is not None
                            and _is_opposite_turn(command, last_turn_command)
                        ):
                            command = last_turn_command
                            explore_mode = "deterministic"
                            pending_opposite_correction = False

                        reason_only = await reason_client.generate_reason(
                            screenshot=screenshot,
                            command=command,
                            turn=preview_sample_id,
                            visibility=visibility_label,
                            explore_mode=explore_mode,
                        )
                        reason_text = f"{reason_only}`{command}`"

                        command_success = await execute_command(page, command)
                        state_after = await get_state(page)
                        tries_after = await _read_total_tries(page)
                        event_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
                        post_x, post_z, _ = await _read_pose(page)

                        if visibility_label == "NOT_VISIBLE" and command in {"left", "right"}:
                            explore_turn_streak += 1
                        else:
                            explore_turn_streak = 0

                        if command in {"left", "right"} and post_x == cur_x and post_z == cur_z:
                            turn_no_move_streak += 1
                        else:
                            turn_no_move_streak = 0

                        if command in {"left", "right"}:
                            last_turn_command = command

                        episode_pending.append(
                            {
                                "screenshot": screenshot,
                                "reason_text": reason_text,
                                "command": command,
                                "command_success": command_success,
                                "won_after": bool(state_after.get("won", False)),
                                "steps_after": int(state_after.get("steps", 0)),
                                "tries_after": tries_after,
                                "event_timestamp_ms": event_ts,
                            }
                        )
                        executed += 1

                        if emit_logs and (verbose or preview_sample_id <= 10 or preview_sample_id % 50 == 0):
                            episode_log_lines.append(
                                f"{prefix}   [sample {preview_sample_id}] cmd='{command}' "
                                f"ok={command_success} won={bool(state_after.get('won', False))} "
                                f"vis={visibility_label} (model={model_visibility},heur={heuristic_visibility}) "
                                f"explore={explore_mode or 'off'} reason={reason_only}"
                            )

                        if state_after.get("won"):
                            won = True
                            won_games += 1
                            break

                    if won and episode_pending:
                        for pending in episode_pending:
                            sample_id = len(samples) + 1
                            frame_file = f"frame_{sample_id:06d}.png"
                            (frames_dir / frame_file).write_bytes(pending["screenshot"])
                            samples.append(
                                GeneratedSample(
                                    sample_id=sample_id,
                                    episode_id=episode_id,
                                    seed=episode_seed,
                                    frame_file=f"frames/{frame_file}",
                                    reason_text=str(pending["reason_text"]),
                                    command=str(pending["command"]),
                                    command_success=bool(pending["command_success"]),
                                    won_after=bool(pending["won_after"]),
                                    steps_after=int(pending["steps_after"]),
                                    tries_after=int(pending["tries_after"]),
                                    event_timestamp_ms=int(pending["event_timestamp_ms"]),
                                )
                            )
                        while won_games * 100 >= target_games * next_progress_pct and next_progress_pct <= 100:
                            records_now = [asdict(s) for s in samples]
                            _write_samples_snapshot(output_dir, records_now, next_progress_pct)
                            if emit_logs:
                                print(
                                    f"{prefix} checkpoint {next_progress_pct}%: "
                                    f"won_games={won_games}/{target_games}, samples={len(records_now)}"
                                )
                            next_progress_pct += 10

                    if emit_logs and won:
                        print(episode_header)
                        for line in episode_log_lines:
                            print(line)
                        print(
                            f"{prefix}   [episode {episode_id}] reached goal "
                            f"(won_games={won_games}/{target_games})"
                        )
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
                except (PlaywrightTimeoutError, TimeoutError) as exc:
                    if emit_logs:
                        print(
                            f"{prefix} [episode {episode_id}] timeout; skipping episode "
                            f"(seed={episode_seed}, error={type(exc).__name__})"
                        )
                    episodes_summary.append(
                        {
                            "episode_id": episode_id,
                            "seed": episode_seed,
                            "status": "timeout_error",
                        }
                    )
                    continue
        finally:
            await browser.close()

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "auto_generate_shortest_path",
        "reason_mode": "vlm_image_short_think_plus_action",
        "reason_model": reason_model,
        "reason_base_url": reason_base_url,
        "visibility_mode": "vlm_binary_visible_not_visible",
        "visibility_heuristic": "gold_pixel_threshold_agreement_gate",
        "visibility_model": visibility_model,
        "visibility_base_url": visibility_base_url,
        "not_visible_policy": "explore_only_if_model_and_heuristic_not_visible",
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
    reason_model: str,
    reason_base_url: str,
    reason_api_key: str,
    reason_timeout_sec: float,
    reason_max_retries: int,
    visibility_model: str,
    visibility_base_url: str,
    visibility_api_key: str,
    visibility_timeout_sec: float,
    visibility_max_retries: int,
    verbose: bool,
) -> tuple[int, int]:
    cmd = [
        sys.executable,
        "-m",
        "src.data.generate_with_reasoning",
        "--n-blocks",
        str(n_blocks),
        "--grid-size",
        str(grid_size),
        "--samples",
        str(worker_target_games),
        "--max-episodes",
        str(max_episodes),
        "--n-workers",
        "1",
        "--reason-model",
        reason_model,
        "--reason-base-url",
        reason_base_url,
        "--reason-api-key",
        reason_api_key,
        "--reason-timeout-sec",
        str(reason_timeout_sec),
        "--reason-max-retries",
        str(reason_max_retries),
        "--visibility-model",
        visibility_model,
        "--visibility-base-url",
        visibility_base_url,
        "--visibility-api-key",
        visibility_api_key,
        "--visibility-timeout-sec",
        str(visibility_timeout_sec),
        "--visibility-max-retries",
        str(visibility_max_retries),
        "--verbose" if verbose else "",
        "--_worker-output-dir",
        str(worker_output_dir),
        "--_worker-id",
        str(worker_id),
        "--_total-workers",
        str(total_workers),
    ]
    if seed is not None:
        cmd.extend(["--seed", seed])

    cmd = [part for part in cmd if part != ""]
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
    reason_model: str,
    reason_base_url: str,
    visibility_model: str,
    visibility_base_url: str,
) -> dict[str, Any]:
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    merged_samples: list[dict[str, Any]] = []
    merged_episodes: list[dict[str, Any]] = []
    total_won_games = 0
    total_episodes_ran = 0

    for worker_id, worker_dir in enumerate(worker_dirs):
        samples_path = worker_dir / "samples.json"
        if not samples_path.exists():
            print(f"[merge] skipping worker {worker_id}; output missing in {worker_dir}")
            continue

        worker_samples = json.loads(samples_path.read_text(encoding="utf-8"))
        worker_won_episodes = {
            int(s.get("episode_id", -1))
            for s in worker_samples
            if int(s.get("episode_id", -1)) > 0
        }
        total_won_games += len(worker_won_episodes)
        total_episodes_ran += len(worker_won_episodes)

        episode_offset = len(merged_episodes)
        episode_id_map: dict[int, int] = {}

        for old_episode_id in sorted(worker_won_episodes):
            new_episode_id = episode_offset + len(episode_id_map) + 1
            episode_id_map[old_episode_id] = new_episode_id
            merged_episodes.append(
                {
                    "episode_id": new_episode_id,
                    "worker_id": worker_id,
                    "status": "won",
                }
            )

        for sample in worker_samples:
            old_frame_rel = str(sample.get("frame_file", ""))
            old_frame_path = worker_dir / old_frame_rel
            if not old_frame_path.exists():
                print(f"[merge] skipping sample with missing frame: worker={worker_id} path={old_frame_path}")
                continue

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
        "reason_mode": "vlm_image_short_think_plus_action",
        "reason_model": reason_model,
        "reason_base_url": reason_base_url,
        "visibility_mode": "vlm_binary_visible_not_visible",
        "visibility_heuristic": "gold_pixel_threshold_agreement_gate",
        "visibility_model": visibility_model,
        "visibility_base_url": visibility_base_url,
        "not_visible_policy": "explore_only_if_model_and_heuristic_not_visible",
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

    (run_dir / "samples.json").write_text(json.dumps(merged_samples, indent=2), encoding="utf-8")
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in merged_samples:
            f.write(json.dumps(row) + "\n")

    return {
        "run_dir": str(run_dir),
        "metadata": metadata,
        "samples_file": str(run_dir / "samples.json"),
    }


async def generate_dataset(
    seed: str | None,
    n_blocks: int,
    grid_size: int,
    n_samples: int,
    max_episodes: int,
    workers: int,
    reason_model: str,
    reason_base_url: str,
    reason_api_key: str,
    reason_timeout_sec: float,
    reason_max_retries: int,
    visibility_model: str,
    visibility_base_url: str,
    visibility_api_key: str,
    visibility_timeout_sec: float,
    visibility_max_retries: int,
    verbose: bool = False,
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
    print(
        f"Reasoning: model={reason_model}, base_url={reason_base_url}, "
        f"max_retries={reason_max_retries}, timeout_sec={reason_timeout_sec}"
    )
    print(
        f"Visibility: model={visibility_model}, base_url={visibility_base_url}, "
        f"max_retries={visibility_max_retries}, timeout_sec={visibility_timeout_sec}"
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
            reason_model=reason_model,
            reason_base_url=reason_base_url,
            reason_api_key=reason_api_key,
            reason_timeout_sec=reason_timeout_sec,
            reason_max_retries=reason_max_retries,
            visibility_model=visibility_model,
            visibility_base_url=visibility_base_url,
            visibility_api_key=visibility_api_key,
            visibility_timeout_sec=visibility_timeout_sec,
            visibility_max_retries=visibility_max_retries,
            verbose=verbose,
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
                    reason_model=reason_model,
                    reason_base_url=reason_base_url,
                    reason_api_key=reason_api_key,
                    reason_timeout_sec=reason_timeout_sec,
                    reason_max_retries=reason_max_retries,
                    visibility_model=visibility_model,
                    visibility_base_url=visibility_base_url,
                    visibility_api_key=visibility_api_key,
                    visibility_timeout_sec=visibility_timeout_sec,
                    visibility_max_retries=visibility_max_retries,
                    verbose=verbose,
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
        print(f"Continuing despite worker failures: {worker_failures}")

    merged = await _merge_worker_outputs(
        run_dir=run_dir,
        worker_dirs=worker_dirs,
        seed=seed,
        n_blocks=n_blocks,
        grid_size=grid_size,
        target_games=n_samples,
        max_episodes=max_episodes,
        workers=workers,
        reason_model=reason_model,
        reason_base_url=reason_base_url,
        visibility_model=visibility_model,
        visibility_base_url=visibility_base_url,
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
        "--n-workers",
        "--n_workers",
        dest="workers",
        type=int,
        default=1,
        help="Number of worker processes to run in parallel",
    )
    parser.add_argument(
        "--reason-model",
        type=str,
        default=DEFAULT_REASON_MODEL,
        help="Vision-capable model ID exposed by the OpenAI-compatible server",
    )
    parser.add_argument(
        "--reason-base-url",
        type=str,
        default=DEFAULT_REASON_BASE_URL,
        help="OpenAI-compatible base URL for reasoning generation",
    )
    parser.add_argument(
        "--reason-api-key",
        type=str,
        default=DEFAULT_REASON_API_KEY,
        help="API key for reasoning server",
    )
    parser.add_argument(
        "--reason-timeout-sec",
        type=float,
        default=30.0,
        help="Per-reasoning request timeout in seconds",
    )
    parser.add_argument(
        "--reason-max-retries",
        type=int,
        default=2,
        help="Retries for malformed/failed reasoning responses",
    )
    parser.add_argument(
        "--visibility-model",
        type=str,
        default=DEFAULT_VISIBILITY_MODEL,
        help="Model ID for binary gold visibility classification (defaults to --reason-model)",
    )
    parser.add_argument(
        "--visibility-base-url",
        type=str,
        default=DEFAULT_VISIBILITY_BASE_URL,
        help="OpenAI-compatible base URL for visibility classification (defaults to --reason-base-url)",
    )
    parser.add_argument(
        "--visibility-api-key",
        type=str,
        default=DEFAULT_VISIBILITY_API_KEY,
        help="API key for visibility server (defaults to --reason-api-key)",
    )
    parser.add_argument(
        "--visibility-timeout-sec",
        type=float,
        default=15.0,
        help="Per-visibility request timeout in seconds",
    )
    parser.add_argument(
        "--visibility-max-retries",
        type=int,
        default=1,
        help="Retries for malformed/failed visibility responses",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample logs for every sample",
    )

    parser.add_argument("--_worker-output-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_total-workers", type=int, default=1, help=argparse.SUPPRESS)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visibility_model = args.visibility_model or args.reason_model
    visibility_base_url = args.visibility_base_url or args.reason_base_url
    visibility_api_key = args.visibility_api_key or args.reason_api_key

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
                reason_model=args.reason_model,
                reason_base_url=args.reason_base_url,
                reason_api_key=args.reason_api_key,
                reason_timeout_sec=args.reason_timeout_sec,
                reason_max_retries=args.reason_max_retries,
                visibility_model=visibility_model,
                visibility_base_url=visibility_base_url,
                visibility_api_key=visibility_api_key,
                visibility_timeout_sec=args.visibility_timeout_sec,
                visibility_max_retries=args.visibility_max_retries,
                verbose=args.verbose,
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
                reason_model=args.reason_model,
                reason_base_url=args.reason_base_url,
                reason_api_key=args.reason_api_key,
                reason_timeout_sec=args.reason_timeout_sec,
                reason_max_retries=args.reason_max_retries,
                visibility_model=visibility_model,
                visibility_base_url=visibility_base_url,
                visibility_api_key=visibility_api_key,
                visibility_timeout_sec=args.visibility_timeout_sec,
                visibility_max_retries=args.visibility_max_retries,
                verbose=args.verbose,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
