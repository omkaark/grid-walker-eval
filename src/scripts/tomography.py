#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..common.prompts import SYSTEM_PROMPT


def _resolve_image_token_id(processor, model) -> int | None:
    for attr in ("image_token_id", "vision_token_id", "img_token_id"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value >= 0:
            return value

    token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if isinstance(token_id, int) and token_id >= 0:
        return token_id
    return None


def _make_messages(image_path: str, turn: int) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Turn {turn}. What is your next move?"},
                {"type": "image", "image": image_path},
            ],
        }
    ]


def _to_device(batch: dict, device: str) -> dict:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def _normalize_01(values: torch.Tensor) -> torch.Tensor:
    min_v = values.min()
    max_v = values.max()
    if float(max_v - min_v) < 1e-12:
        return torch.zeros_like(values)
    return (values - min_v) / (max_v - min_v)


def _reshape_image_scores(image_scores: torch.Tensor, model_inputs: dict) -> torch.Tensor:
    grid = model_inputs.get("image_grid_thw")
    if torch.is_tensor(grid) and grid.numel() >= 3:
        t, h, w = [int(x) for x in grid[0].tolist()]
        expected = t * h * w
        if expected == image_scores.numel() and t > 0 and h > 0 and w > 0:
            return image_scores.reshape(t, h, w).mean(dim=0)

    n = int(image_scores.numel())
    side = int(torch.ceil(torch.sqrt(torch.tensor(float(n)))).item())
    padded = torch.full((side * side,), float("nan"))
    padded[:n] = image_scores
    return padded.reshape(side, side)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate attention over all layers/heads/query tokens for image tokens "
            "and export a normalized [0,1] heatmap."
        )
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--turn",
        type=int,
        default=1,
        help="Turn number used in the user message (default: 1).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", "Qwen/Qwen3-VL-2B-Instruct"),
        help="HF model id or local path.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("heatmap.png")),
        help="Output heatmap path.",
    )
    args = parser.parse_args()

    image_path = str(Path(args.image).expanduser().resolve())
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()

    if args.turn <= 0:
        raise ValueError("--turn must be a positive integer.")

    messages = _make_messages(image_path=image_path, turn=args.turn)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, _ = process_vision_info(messages)
    model_inputs = processor(
        text=[text],
        images=[image_inputs],
        return_tensors="pt",
        padding=True,
    )
    model_inputs = _to_device(model_inputs, device=device)

    with torch.no_grad():
        outputs = model(
            **model_inputs,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )

    if not outputs.attentions:
        raise RuntimeError("Model did not return attentions. Disable flash attention/eager mode for analysis.")

    image_token_id = _resolve_image_token_id(processor, model)
    if image_token_id is None:
        raise RuntimeError("Could not determine image token id from config/tokenizer.")

    input_ids = model_inputs["input_ids"][0]
    image_positions = torch.where(input_ids == image_token_id)[0]
    if image_positions.numel() == 0:
        raise RuntimeError("No image tokens found in sequence; cannot build image-token heatmap.")

    # [layers, heads, q_len, k_len]
    attn = torch.stack([layer_attn[0].float().cpu() for layer_attn in outputs.attentions], dim=0)
    # Sum over layers, heads, and all query positions -> score per key token.
    token_scores = attn.sum(dim=(0, 1, 2))
    image_scores = token_scores[image_positions.cpu()]
    image_scores = _normalize_01(image_scores)
    heatmap = _reshape_image_scores(image_scores, model_inputs)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(heatmap.numpy(), cmap="inferno", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Normalized attention [0,1]")
    plt.title("Summed Attention Over Image Tokens")
    plt.xlabel("Patch X")
    plt.ylabel("Patch Y")
    plt.tight_layout()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    print(f"Saved heatmap to: {output_path}")


if __name__ == "__main__":
    main()
