import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.prompts import SYSTEM_PROMPT


MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
LOCAL_FILES_ONLY = os.getenv("GRID_WALKER_LOCAL_FILES_ONLY", "1") == "1"
DEFAULT_DATASET_DIR = Path("dataset/simpleds")


@dataclass
class TurnSample:
    sample_id: int
    episode_id: int
    frame_path: Path
    command: str


def _resolve_local_model_source(model_name: str) -> str:
    if os.path.isdir(model_name):
        return model_name

    hf_home = os.getenv("HF_HOME")
    if not hf_home:
        return model_name

    model_dir = Path(hf_home) / "hub" / f"models--{model_name.replace('/', '--')}"
    ref_main = model_dir / "refs" / "main"
    if not ref_main.exists():
        return model_name

    try:
        revision = ref_main.read_text(encoding="utf-8").strip()
    except OSError:
        return model_name

    snapshot_dir = model_dir / "snapshots" / revision
    return str(snapshot_dir) if snapshot_dir.exists() else model_name


def _load_episode_turns(dataset_dir: Path) -> list[list[TurnSample]]:
    samples_file = dataset_dir / "samples.json"
    if not samples_file.exists():
        raise FileNotFoundError(f"Missing dataset file: {samples_file}")

    rows = json.loads(samples_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No rows found in {samples_file}")

    grouped: dict[int, list[TurnSample]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            sample_id = int(row["sample_id"])
            episode_id = int(row["episode_id"])
            frame_rel = str(row["frame_file"])
            command = str(row["command"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not command:
            continue

        frame_path = dataset_dir / frame_rel
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame referenced in dataset: {frame_path}")

        grouped.setdefault(episode_id, []).append(
            TurnSample(
                sample_id=sample_id,
                episode_id=episode_id,
                frame_path=frame_path,
                command=command,
            )
        )

    episodes: list[list[TurnSample]] = []
    for episode_id in sorted(grouped):
        turns = sorted(grouped[episode_id], key=lambda t: t.sample_id)
        if turns:
            episodes.append(turns)
    if not episodes:
        raise ValueError("No valid episodes found.")
    return episodes


def _build_messages(turns: list[TurnSample]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    for idx, turn in enumerate(turns, start=1):
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Turn {idx}. What is your next move?"},
                    {"type": "image", "image": str(turn.frame_path.resolve())},
                ],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"`{turn.command}`"}],
            }
        )
    return messages


def _encode_messages(
    messages: list[dict[str, Any]], processor: AutoProcessor
) -> tuple[dict[str, torch.Tensor], str]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    image_inputs, _ = process_vision_info(messages)
    if image_inputs is None:
        image_inputs = []

    model_inputs = processor(
        text=[text],
        images=[image_inputs],
        return_tensors="pt",
        padding=True,
    )
    return model_inputs, text


def _sequence_length(model_inputs: dict[str, torch.Tensor]) -> int:
    if "attention_mask" in model_inputs:
        return int(model_inputs["attention_mask"][0].sum().item())
    return int(model_inputs["input_ids"].shape[1])


def _find_supervised_spans(
    turns: list[TurnSample], processor: AutoProcessor
) -> tuple[dict[str, torch.Tensor], str, list[tuple[int, int, int]]]:
    full_messages = _build_messages(turns)
    full_inputs, full_text = _encode_messages(full_messages, processor)
    full_len = _sequence_length(full_inputs)

    spans: list[tuple[int, int, int]] = []
    for turn_idx in range(len(turns)):
        prefix_without_assistant = _build_messages(turns[: turn_idx + 1])[:-1]
        prefix_with_assistant = _build_messages(turns[: turn_idx + 1])

        inputs_a, _ = _encode_messages(prefix_without_assistant, processor)
        inputs_b, _ = _encode_messages(prefix_with_assistant, processor)
        start = _sequence_length(inputs_a)
        end = _sequence_length(inputs_b)
        if end > full_len:
            end = full_len
        if end > start:
            spans.append((turn_idx + 1, start, end))

    return full_inputs, full_text, spans


def _fit_turns_to_max_seq_len(
    episode_turns: list[TurnSample],
    processor: AutoProcessor,
    max_seq_len: int,
    max_turns: int,
) -> list[TurnSample]:
    selected: list[TurnSample] = []
    for turn in episode_turns[:max_turns]:
        candidate = selected + [turn]
        candidate_msgs = _build_messages(candidate)
        candidate_inputs, _ = _encode_messages(candidate_msgs, processor)
        candidate_len = _sequence_length(candidate_inputs)
        if candidate_len > max_seq_len:
            break
        selected = candidate
    return selected


def _labels_from_spans(
    input_ids: torch.Tensor, spans: list[tuple[int, int, int]]
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:] = -100
    for _, start, end in spans:
        labels[0, start:end] = input_ids[0, start:end]
    return labels


def _print_debug_sample(
    turns: list[TurnSample],
    full_text: str,
    input_ids: torch.Tensor,
    spans: list[tuple[int, int, int]],
    labels: torch.Tensor,
    processor: AutoProcessor,
) -> None:
    seq_len = int(input_ids.shape[1])
    supervised_count = int((labels != -100).sum().item())
    print("\n=== Debug Sample ===")
    print(f"episode_id={turns[0].episode_id} turns={len(turns)} seq_len={seq_len}")
    print(f"supervised_token_count={supervised_count}")
    print("\nChat template (first 1200 chars):")
    print(full_text[:1200])

    print("\nAssistant spans and decoded supervised text:")
    for turn_idx, start, end in spans:
        token_ids = input_ids[0, start:end].tolist()
        decoded = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
        expected = turns[turn_idx - 1].command
        print(
            f"- turn={turn_idx} span=[{start},{end}) token_count={end-start} "
            f"expected_command={expected!r}"
        )
        print(f"  decoded={decoded!r}")


def validate(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    episodes = _load_episode_turns(dataset_dir)
    model_source = _resolve_local_model_source(args.model)
    processor = AutoProcessor.from_pretrained(model_source, local_files_only=LOCAL_FILES_ONLY)

    checked = 0
    zero_supervised = 0
    total_supervised = 0
    max_seq_len_seen = 0
    total_turns = 0

    debug_printed = False
    for episode in episodes[: args.num_episodes]:
        turns = _fit_turns_to_max_seq_len(
            episode_turns=episode,
            processor=processor,
            max_seq_len=args.max_seq_len,
            max_turns=args.max_turns,
        )
        if not turns:
            continue
        full_inputs, full_text, spans = _find_supervised_spans(turns, processor)
        input_ids = full_inputs["input_ids"]
        labels = _labels_from_spans(input_ids, spans)
        supervised_count = int((labels != -100).sum().item())

        if supervised_count == 0:
            zero_supervised += 1
        total_supervised += supervised_count
        max_seq_len_seen = max(max_seq_len_seen, int(input_ids.shape[1]))
        total_turns += len(turns)
        checked += 1

        if not debug_printed:
            _print_debug_sample(turns, full_text, input_ids, spans, labels, processor)
            debug_printed = True

    print("\n=== Validation Summary ===")
    print(f"episodes_checked={checked}")
    print(f"zero_supervised_episodes={zero_supervised}")
    print(f"avg_turns_per_episode={(total_turns / checked if checked else 0.0):.2f}")
    print(
        f"avg_supervised_tokens_per_episode="
        f"{(total_supervised / checked if checked else 0.0):.2f}"
    )
    print(f"max_seq_len_seen={max_seq_len_seen}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate multi-turn episode formatting and supervised token spans."
    )
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
