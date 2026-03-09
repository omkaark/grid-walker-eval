#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import re

_default_cache_root = Path(__file__).resolve().parent / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_default_cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ["XDG_CACHE_HOME"]) / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..common.prompts import SYSTEM_PROMPT

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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


def _natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name.lower())
    key: list[int | str] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in _IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]

    if input_path.is_dir():
        images = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
        images.sort(key=_natural_sort_key)
        if not images:
            raise ValueError(f"No image files found in directory: {input_path}")
        return images

    raise FileNotFoundError(f"Path not found: {input_path}")


def _reshape_image_scores(
    image_scores: torch.Tensor,
    model_inputs: dict,
    spatial_merge_size: int | None = None,
) -> torch.Tensor:
    grid = model_inputs.get("image_grid_thw")
    if torch.is_tensor(grid) and grid.numel() >= 3:
        t, h, w = [int(x) for x in grid[0].tolist()]
        if t > 0 and h > 0 and w > 0:
            n_tokens = int(image_scores.numel())
            expected_raw = t * h * w
            if expected_raw == n_tokens:
                return image_scores.reshape(t, h, w).mean(dim=0)

            # Qwen3-VL exposes image_grid_thw before spatial token merge.
            if spatial_merge_size is not None and spatial_merge_size > 1:
                if h % spatial_merge_size == 0 and w % spatial_merge_size == 0:
                    merged_h = h // spatial_merge_size
                    merged_w = w // spatial_merge_size
                    expected_merged = t * merged_h * merged_w
                    if expected_merged == n_tokens:
                        return image_scores.reshape(t, merged_h, merged_w).mean(dim=0)

            # Fallback: infer merge factor from token-count ratio when possible.
            if expected_raw % n_tokens == 0:
                ratio = expected_raw // n_tokens
                merge = int(round(float(ratio) ** 0.5))
                if merge > 1 and merge * merge == ratio and h % merge == 0 and w % merge == 0:
                    merged_h = h // merge
                    merged_w = w // merge
                    if t * merged_h * merged_w == n_tokens:
                        return image_scores.reshape(t, merged_h, merged_w).mean(dim=0)

    n = int(image_scores.numel())
    side = int(torch.ceil(torch.sqrt(torch.tensor(float(n)))).item())
    padded = torch.full((side * side,), float("nan"))
    padded[:n] = image_scores
    return padded.reshape(side, side)


def _resolve_cached_hf_model_path(model_id: str) -> str:
    model_path = Path(model_id).expanduser()
    if model_path.exists():
        return str(model_path.resolve())

    if "/" not in model_id:
        return model_id

    hub_model_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    if not hub_model_dir.exists():
        return model_id

    main_ref = hub_model_dir / "refs" / "main"
    if main_ref.exists():
        revision = main_ref.read_text(encoding="utf-8").strip()
        if revision:
            snapshot_path = hub_model_dir / "snapshots" / revision
            if snapshot_path.exists():
                return str(snapshot_path.resolve())

    snapshots_dir = hub_model_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0].resolve())

    return model_id


def _load_processor_with_fallback(model_id: str):
    try:
        return AutoProcessor.from_pretrained(model_id)
    except Exception:
        # In restricted environments, Hub metadata checks can fail even with a warm cache.
        return AutoProcessor.from_pretrained(model_id, local_files_only=True)


def _load_model_with_fallback(model_id: str, dtype: torch.dtype):
    model_kwargs = {
        "dtype": dtype,
        "device_map": None,
        "attn_implementation": "eager",
    }
    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    except Exception:
        return AutoModelForImageTextToText.from_pretrained(
            model_id,
            local_files_only=True,
            **model_kwargs,
        )


def _compute_heatmap_for_image(
    image_path: str,
    turn: int,
    processor,
    model,
    device: str,
    spatial_merge_size: int | None,
) -> torch.Tensor:
    messages = _make_messages(image_path=image_path, turn=turn)
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
        raise RuntimeError(f"No image tokens found in sequence for: {image_path}")

    # [layers, heads, q_len, k_len]
    attn = torch.stack([layer_attn[0].float().cpu() for layer_attn in outputs.attentions], dim=0)
    # Sum over layers, heads, and all query positions -> score per key token.
    token_scores = attn.sum(dim=(0, 1, 2))
    image_scores = token_scores[image_positions.cpu()]
    image_scores = _normalize_01(image_scores)
    return _reshape_image_scores(
        image_scores,
        model_inputs,
        spatial_merge_size=spatial_merge_size,
    )


def _ensure_rgb01(base_image: np.ndarray) -> np.ndarray:
    if base_image.ndim == 2:
        base_image = np.stack([base_image] * 3, axis=-1)
    elif base_image.ndim == 3 and base_image.shape[2] == 4:
        base_image = base_image[:, :, :3]
    elif base_image.ndim != 3 or base_image.shape[2] < 3:
        raise ValueError(f"Unsupported image shape: {base_image.shape}")

    if np.issubdtype(base_image.dtype, np.integer):
        base = base_image.astype(np.float32) / 255.0
    else:
        base = base_image.astype(np.float32)
        if base.max() > 1.0:
            base = np.clip(base / 255.0, 0.0, 1.0)
        else:
            base = np.clip(base, 0.0, 1.0)
    return base


def _make_overlay_frame(base_image: np.ndarray, heatmap: torch.Tensor, alpha: float) -> np.ndarray:
    base = _ensure_rgb01(base_image)
    h_px, w_px = base.shape[:2]

    hm = heatmap.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    hm_resized = F.interpolate(hm, size=(h_px, w_px), mode="bilinear", align_corners=False)[0, 0].numpy()
    hm_resized = np.nan_to_num(hm_resized, nan=0.0, posinf=1.0, neginf=0.0)
    hm_resized = np.clip(hm_resized, 0.0, 1.0)
    heat_rgb = plt.get_cmap("inferno")(hm_resized)[..., :3].astype(np.float32)

    blended = (1.0 - alpha) * base + alpha * heat_rgb
    return (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)


def _make_base_frame(base_image: np.ndarray) -> np.ndarray:
    return (_ensure_rgb01(base_image) * 255.0).astype(np.uint8)


def _add_step_label(frame_rgb: np.ndarray, step: int, text_color: tuple[int, int, int]) -> Image.Image:
    image = Image.fromarray(frame_rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"Step {step}"
    pad_x = max(8, image.width // 60)
    pad_y = max(6, image.height // 80)
    draw.text((pad_x, pad_y), text, fill=text_color, font=font)
    return image


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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap opacity when overlaid on the input image (default: 0.45).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="GIF frame rate when --image points to a directory (default: 2.0).",
    )
    parser.add_argument(
        "--no-tomo",
        action="store_true",
        help="For directory input, skip tomography and export a labeled GIF from the input images.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    image_paths = _collect_images(image_path)
    is_batch = image_path.is_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if args.turn <= 0:
        raise ValueError("--turn must be a positive integer.")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be between 0 and 1.")
    if args.fps <= 0:
        raise ValueError("--fps must be a positive number.")
    if args.no_tomo and not is_batch:
        raise ValueError("--no-tomo requires --image to be a directory.")

    processor = None
    model = None
    spatial_merge_size = None
    if not args.no_tomo:
        model_source = _resolve_cached_hf_model_path(args.model)
        processor = _load_processor_with_fallback(model_source)
        model = _load_model_with_fallback(model_source, dtype=dtype).to(device)
        model.eval()
        spatial_merge_size = getattr(getattr(model.config, "vision_config", None), "spatial_merge_size", None)
        if not isinstance(spatial_merge_size, int) or spatial_merge_size <= 0:
            spatial_merge_size = None

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_batch:
        if output_path.suffix.lower() != ".gif":
            output_path = output_path.with_suffix(".gif")

        frames: list[Image.Image] = []
        for frame_index, frame_path in enumerate(image_paths):
            turn = args.turn + frame_index
            base_image = plt.imread(str(frame_path))
            if args.no_tomo:
                frame_rgb = _make_base_frame(base_image)
                text_color = (255, 255, 255)
            else:
                if processor is None or model is None:
                    raise RuntimeError("Tomography model is not initialized.")
                heatmap = _compute_heatmap_for_image(
                    image_path=str(frame_path),
                    turn=turn,
                    processor=processor,
                    model=model,
                    device=device,
                    spatial_merge_size=spatial_merge_size,
                )
                frame_rgb = _make_overlay_frame(base_image=base_image, heatmap=heatmap, alpha=args.alpha)
                text_color = (0, 0, 0)
            frames.append(_add_step_label(frame_rgb=frame_rgb, step=turn, text_color=text_color))
            print(f"[{frame_index + 1}/{len(image_paths)}] processed {frame_path.name}")

        duration_ms = int(round(1000.0 / args.fps))
        if not frames:
            raise RuntimeError("No frames produced for GIF.")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"Saved overlay GIF to: {output_path}")
    else:
        if processor is None or model is None:
            raise RuntimeError("Tomography model is not initialized.")
        single_image = image_paths[0]
        heatmap = _compute_heatmap_for_image(
            image_path=str(single_image),
            turn=args.turn,
            processor=processor,
            model=model,
            device=device,
            spatial_merge_size=spatial_merge_size,
        )
        base_image = plt.imread(str(single_image))
        heatmap_np = np.ma.masked_invalid(heatmap.numpy())

        h_px, w_px = base_image.shape[:2]
        extent = (0, w_px, h_px, 0)
        plt.figure(figsize=(7, 6))
        plt.imshow(base_image, extent=extent, interpolation="nearest")
        im = plt.imshow(
            heatmap_np,
            cmap="inferno",
            interpolation="bilinear",
            alpha=args.alpha,
            extent=extent,
        )
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Normalized attention [0,1]")
        plt.title("Summed Attention Over Image Tokens (Overlay)")
        plt.xlabel("Image X")
        plt.ylabel("Image Y")
        plt.tight_layout()
        plt.savefig(output_path, dpi=220)
        plt.close()
        print(f"Saved heatmap to: {output_path}")


if __name__ == "__main__":
    main()
