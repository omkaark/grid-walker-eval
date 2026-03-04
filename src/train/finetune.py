import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.common.prompts import SYSTEM_PROMPT


MODEL_NAME ="Qwen/Qwen3-VL-2B-Instruct"
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1") == "1"
DEFAULT_DATASET_DIR = Path("dataset/simpleds")
DEFAULT_OUTPUT_DIR = Path("adapter_policy")

TEXT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class EpisodeTurn:
    sample_id: int
    episode_id: int
    frame_path: Path
    command: str


@dataclass
class EpisodeSample:
    episode_id: int
    turns: list[EpisodeTurn]


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


def _is_text_lora_param(name: str) -> bool:
    if "lora_" not in name:
        return False
    if any(token in name for token in ("visual", "vision", "image", "img")):
        return False
    return any(module_name in name for module_name in TEXT_LORA_TARGET_MODULES)


def _load_dataset(dataset_dir: Path) -> list[EpisodeSample]:
    samples_file = dataset_dir / "samples.json"
    if not samples_file.exists():
        raise FileNotFoundError(f"Missing dataset file: {samples_file}")

    rows = json.loads(samples_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No rows found in {samples_file}")

    grouped: dict[int, list[EpisodeTurn]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue

        sample_id = row.get("sample_id")
        episode_id = row.get("episode_id")
        frame_rel = row.get("frame_file")
        command = row.get("command")
        if (
            not isinstance(sample_id, int)
            or not isinstance(episode_id, int)
            or not isinstance(frame_rel, str)
            or not isinstance(command, str)
        ):
            continue
        if not command.strip():
            continue

        frame_path = dataset_dir / frame_rel
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame referenced in dataset: {frame_path}")

        grouped.setdefault(episode_id, []).append(
            EpisodeTurn(
                sample_id=sample_id,
                episode_id=episode_id,
                frame_path=frame_path,
                command=command.strip(),
            )
        )

    out: list[EpisodeSample] = []
    for episode_id in sorted(grouped):
        turns = sorted(grouped[episode_id], key=lambda t: t.sample_id)
        if turns:
            out.append(EpisodeSample(episode_id=episode_id, turns=turns))
    if not out:
        raise ValueError(f"No valid episodes found in {samples_file}")
    return out


def _build_messages(turns: list[EpisodeTurn]) -> list[dict[str, Any]]:
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
        # Keep target format aligned with deployment prompt style.
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": f"`{turn.command}`"}]}
        )
    return messages


def _encode_messages(
    messages: list[dict[str, Any]],
    processor: AutoProcessor,
) -> tuple[dict[str, torch.Tensor], str, list[Any]]:
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
    return model_inputs, text, image_inputs


def _sequence_length(model_inputs: dict[str, torch.Tensor]) -> int:
    if "attention_mask" in model_inputs:
        return int(model_inputs["attention_mask"][0].sum().item())
    return int(model_inputs["input_ids"].shape[1])


def _fit_turns_to_max_seq_len(
    episode_turns: list[EpisodeTurn],
    processor: AutoProcessor,
    max_turns: int,
    max_seq_len: int,
) -> list[EpisodeTurn]:
    selected: list[EpisodeTurn] = []
    for turn in episode_turns[:max_turns]:
        candidate = selected + [turn]
        candidate_messages = _build_messages(candidate)
        candidate_inputs, _, _ = _encode_messages(candidate_messages, processor)
        if _sequence_length(candidate_inputs) > max_seq_len:
            break
        selected = candidate
    return selected


def _compute_supervised_spans(
    turns: list[EpisodeTurn],
    processor: AutoProcessor,
) -> tuple[str, list[Any], list[tuple[int, int]]]:
    full_messages = _build_messages(turns)
    full_inputs, full_text, full_images = _encode_messages(full_messages, processor)
    full_len = _sequence_length(full_inputs)

    spans: list[tuple[int, int]] = []
    for turn_idx in range(len(turns)):
        prefix_without_assistant = _build_messages(turns[: turn_idx + 1])[:-1]
        prefix_with_assistant = _build_messages(turns[: turn_idx + 1])

        start_inputs, _, _ = _encode_messages(prefix_without_assistant, processor)
        end_inputs, _, _ = _encode_messages(prefix_with_assistant, processor)

        start = _sequence_length(start_inputs)
        end = min(_sequence_length(end_inputs), full_len)
        if end > start:
            spans.append((start, end))

    return full_text, full_images, spans


def _prepare_batch(
    batch: list[EpisodeSample],
    processor: AutoProcessor,
    max_turns_per_sample: int,
    max_seq_len: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    texts: list[str] = []
    all_images: list[Any] = []
    assistant_spans: list[list[tuple[int, int]]] = []

    for episode in batch:
        turns = _fit_turns_to_max_seq_len(
            episode_turns=episode.turns,
            processor=processor,
            max_turns=max_turns_per_sample,
            max_seq_len=max_seq_len,
        )
        if not turns:
            continue

        text, images, spans = _compute_supervised_spans(turns, processor)
        if not spans:
            continue

        texts.append(text)
        all_images.append(images)
        assistant_spans.append(spans)

    if not texts:
        raise ValueError("Batch produced no valid episode windows with supervised spans.")

    model_inputs = processor(
        text=texts,
        images=all_images,
        return_tensors="pt",
        padding=True,
    )

    input_ids = model_inputs["input_ids"]
    labels = input_ids.clone()
    labels[:] = -100

    batch_size, max_len = input_ids.shape
    for i in range(batch_size):
        for start, end in assistant_spans[i]:
            if start >= max_len:
                continue
            end = min(end, max_len)
            if end > start:
                labels[i, start:end] = input_ids[i, start:end]

    if "attention_mask" in model_inputs:
        labels = labels.masked_fill(model_inputs["attention_mask"] == 0, -100)

    return model_inputs, labels


def _trainable_param_stats(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(dataset_dir)
    model_source = _resolve_local_model_source(args.model)
    processor = AutoProcessor.from_pretrained(model_source, local_files_only=LOCAL_FILES_ONLY)

    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=LOCAL_FILES_ONLY,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=list(TEXT_LORA_TARGET_MODULES),
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        param.requires_grad = _is_text_lora_param(name)

    trainable, total = _trainable_param_stats(model)
    total_turns = sum(len(ep.turns) for ep in dataset)
    print(
        f"Loaded {len(dataset)} episodes ({total_turns} turns) from {dataset_dir}. "
        f"Trainable params: {trainable}/{total} ({100.0 * trainable / total:.4f}%)"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model.train()
    grad_accum = max(1, args.grad_accum)
    step = 0
    running_loss = 0.0
    updates = 0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(dataset)
        n_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
        print(f"Epoch {epoch}/{args.epochs} - batches: {n_batches}")

        for batch_idx, i in enumerate(range(0, len(dataset), args.batch_size), start=1):
            batch_should_log = batch_idx == 1 or batch_idx % max(1, args.batch_log_every) == 0
            is_first_batch = epoch == 1 and batch_idx == 1
            batch_start = time.perf_counter()
            if batch_should_log:
                print(
                    f"  [epoch {epoch}] batch {batch_idx}/{n_batches} "
                    f"(global_step={step + 1}, updates={updates}) start"
                )

            prep_t0 = time.perf_counter()
            batch = dataset[i : i + args.batch_size]
            model_inputs, labels = _prepare_batch(
                batch=batch,
                processor=processor,
                max_turns_per_sample=args.max_turns_per_sample,
                max_seq_len=args.max_seq_len,
            )
            prep_dt = time.perf_counter() - prep_t0

            transfer_t0 = time.perf_counter()
            model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
            labels = labels.to("cuda")
            transfer_dt = time.perf_counter() - transfer_t0

            fwbw_t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**model_inputs, labels=labels)
                loss = outputs.loss / grad_accum

            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwbw_dt = time.perf_counter() - fwbw_t0
            running_loss += float(loss.item()) * grad_accum

            opt_dt = 0.0
            if (step + 1) % grad_accum == 0:
                opt_t0 = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                opt_dt = time.perf_counter() - opt_t0
                updates += 1

                if updates % args.log_every == 0:
                    avg_loss = running_loss / args.log_every
                    print(f"  update {updates}: loss={avg_loss:.5f}")
                    running_loss = 0.0
                if args.save_every_updates > 0 and updates % args.save_every_updates == 0:
                    ckpt_dir = output_dir / f"checkpoint_update_{updates}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    print(f"  saved adapter checkpoint at update {updates}: {ckpt_dir}")

            batch_dt = time.perf_counter() - batch_start
            if batch_should_log:
                print(
                    f"  [epoch {epoch}] batch {batch_idx}/{n_batches} done "
                    f"in {batch_dt:.2f}s (updates={updates})"
                )
            if is_first_batch:
                print(
                    "  [first-batch timing] "
                    f"prepare={prep_dt:.2f}s, to_cuda={transfer_dt:.2f}s, "
                    f"forward+backward={fwbw_dt:.2f}s, optimizer={opt_dt:.2f}s"
                )

            step += 1

    # Flush remainder gradients if steps were not divisible by grad_accum.
    if step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised finetune of text-only LoRA on Grid Walker dataset.")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--max-turns-per-sample",
        type=int,
        default=64,
        help="Max turns packed from one episode into a single training sample.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum tokenized sequence length per training sample.",
    )
    parser.add_argument(
        "--batch-log-every",
        type=int,
        default=1,
        help="Print batch progress every N batches (per epoch).",
    )
    parser.add_argument(
        "--save-every-updates",
        type=int,
        default=10,
        help="Save adapter checkpoint every N optimizer updates (0 disables periodic saves).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
