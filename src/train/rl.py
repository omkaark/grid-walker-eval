import asyncio
import gc
import json
import os
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from .rollout import run_rollouts
from .vllm_utils import (
    ADAPTER_NAME,
    adapter_exists,
    vllm_is_lora_loaded,
    vllm_reload_weights,
    vllm_reload_with_lora,
    vllm_sleep,
)

MODEL_NAME = os.getenv("GRID_WALKER_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./lora_adapter")
MAX_TURN_NUMBER = int(os.getenv("GRID_WALKER_MAX_TURNS", "10"))
N_ROLLOUTS = int(os.getenv("GRID_WALKER_N_ROLLOUTS", "8"))
N_ITERS = int(os.getenv("GRID_WALKER_N_ITERS", "4"))
VAR_EPS = 1e-6
CLIP_EPS = 0.2
KL_COEF = 0.04
LR = 1e-6

GRADIENT_CHECKPOINTING = os.getenv("GRID_WALKER_GRADIENT_CHECKPOINTING", "1") == "1"
USE_FLASH_ATTN = os.getenv("GRID_WALKER_USE_FLASH_ATTN", "0") == "1"

device = "cuda"
LOCAL_FILES_ONLY = os.getenv("GRID_WALKER_LOCAL_FILES_ONLY", "1") == "1"
N_STEPS = int(os.getenv("GRID_WALKER_N_STEPS", "100"))

# Limit LoRA to language-model transformer blocks (text path only).
TEXT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _is_text_lora_param(name: str) -> bool:
    if "lora_" not in name:
        return False
    if any(token in name for token in ("visual", "vision", "image", "img")):
        return False
    return any(module_name in name for module_name in TEXT_LORA_TARGET_MODULES)


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


MODEL_SOURCE = _resolve_local_model_source(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_SOURCE, local_files_only=LOCAL_FILES_ONLY)


def get_per_token_logps(model, model_inputs: dict[str, Any]) -> torch.Tensor:
    """Compute per-token log probabilities from a multimodal forward pass."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**model_inputs, use_cache=False)
        logits = outputs.logits

    input_ids = model_inputs["input_ids"]
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def free_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _validate_image_objects(messages: list[dict[str, Any]]) -> None:
    """Canonical training format: image objects only, no image_url blocks."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "image_url":
                raise ValueError(
                    "Training expects canonical image objects in messages: "
                    "{'type': 'image', 'image': ...}. image_url blocks are not supported in rl.py."
                )
            if block_type == "image" and "image" not in block:
                raise ValueError("Found an image content block without an 'image' field.")


def _to_int_token_ids(tokenized: Any) -> list[int]:
    """Convert chat-template tokenized output to a flat token id list."""
    if isinstance(tokenized, torch.Tensor):
        values = tokenized.tolist()
    elif isinstance(tokenized, dict):
        if "input_ids" not in tokenized:
            return []
        return _to_int_token_ids(tokenized["input_ids"])
    elif hasattr(tokenized, "input_ids"):
        return _to_int_token_ids(tokenized.input_ids)
    else:
        values = list(tokenized)

    # Many processors return batched outputs even for one conversation: [[...ids...]]
    if isinstance(values, list) and values and isinstance(values[0], (list, tuple)):
        values = list(values[0])

    token_ids: list[int] = []
    for v in values:
        try:
            token_ids.append(int(v))
        except (TypeError, ValueError):
            continue
    return token_ids


def _normalize_messages_for_processor(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure every message uses block-based content expected by chat templates."""
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            content_blocks: list[dict[str, Any]] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content_blocks = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    content_blocks.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image" and "image" in block:
                    content_blocks.append({"type": "image", "image": block["image"]})
                elif block.get("type") == "image_url":
                    # Preserve for validation to raise a clear canonical-format error.
                    content_blocks.append(block)
        else:
            content_blocks = [{"type": "text", "text": str(content)}]

        normalized.append({"role": role, "content": content_blocks})
    return normalized


def prepare_batch(rollouts):
    """Build a multimodal batch and aligned completion masks for PPO-style loss."""
    texts: list[str] = []
    all_images = []
    all_completion_masks = []

    for rollout in rollouts:
        messages = _normalize_messages_for_processor(rollout.request["messages"])
        _validate_image_objects(messages)

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs, _ = process_vision_info(messages)
        if image_inputs is None:
            image_inputs = []

        texts.append(text)
        all_images.append(image_inputs)

        token_ids = _to_int_token_ids(
            processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        seq_len = len(token_ids)

        completion_mask = [0.0] * seq_len

        turn_idx = 0
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue

            tokens_with = _to_int_token_ids(
                processor.apply_chat_template(
                    messages[: i + 1],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
            tokens_without = _to_int_token_ids(
                processor.apply_chat_template(
                    messages[:i],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )

            start_pos = min(len(tokens_without), seq_len)
            end_pos = min(len(tokens_with), seq_len)

            if turn_idx < len(rollout.turns):
                span_len = end_pos - start_pos

                # Always train over assistant tokens in the span.
                for j in range(span_len):
                    pos = start_pos + j
                    if pos < seq_len:
                        completion_mask[pos] = 1.0

                turn_idx += 1

        all_completion_masks.append(torch.tensor(completion_mask, dtype=torch.float32))

    processor_kwargs: dict[str, Any] = {
        "text": texts,
        "images": all_images,
        "return_tensors": "pt",
        "padding": True,
    }

    model_inputs = processor(**processor_kwargs)

    n = len(rollouts)
    max_len = model_inputs["input_ids"].shape[1]
    completion_mask = torch.zeros(n, max_len, dtype=torch.float32)

    for i in range(n):
        seq_len = min(len(all_completion_masks[i]), max_len)
        completion_mask[i, :seq_len] = all_completion_masks[i][:seq_len]

    return {
        "model_inputs": model_inputs,
        "completion_mask": completion_mask[:, 1:],
    }


def _rollout_has_image(rollout: Any) -> bool:
    messages = rollout.request.get("messages", [])
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image" and block.get("image"):
                return True
    return False


def load_model_optimized(adapter_path: str = None, for_training: bool = True):
    """Load Qwen VL model with memory optimizations and optional LoRA adapter."""
    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
    }

    if USE_FLASH_ATTN:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")

    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_SOURCE, local_files_only=LOCAL_FILES_ONLY, **model_kwargs
    )

    if GRADIENT_CHECKPOINTING and for_training:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled")

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        print(f"Loaded existing adapter from {adapter_path}")
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=list(TEXT_LORA_TARGET_MODULES),
            lora_dropout=0.05,
        )
        model = get_peft_model(base_model, lora_config)
        print("Created new LoRA adapter")

    # Freeze all non-text LoRA parameters so only text adapter weights are updated.
    for name, param in model.named_parameters():
        param.requires_grad = _is_text_lora_param(name)

    return model


def _adapter_base_model(adapter_path: str) -> str | None:
    adapter_config = os.path.join(os.path.abspath(adapter_path), "adapter_config.json")
    if not os.path.exists(adapter_config):
        return None
    try:
        with open(adapter_config, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = data.get("base_model_name_or_path")
        return str(base) if isinstance(base, str) else None
    except (OSError, json.JSONDecodeError):
        return None


def initialize_policy_adapter_if_missing() -> None:
    """
    Ensure a policy LoRA adapter exists and is loaded into vLLM.
    If missing, create a default-initialized adapter (no-op at start), save it, then load in vLLM.
    """
    existing_base = _adapter_base_model(ADAPTER_PATH)
    current_model_aliases = {
        str(MODEL_NAME),
        str(MODEL_SOURCE),
        os.path.abspath(str(MODEL_SOURCE)),
    }
    adapter_incompatible = existing_base is not None and existing_base not in current_model_aliases

    if adapter_incompatible:
        print(
            "Existing adapter base model does not match current MODEL_NAME. "
            f"Reinitializing adapter. adapter_base={existing_base}, model={MODEL_NAME}"
        )

    if (not adapter_exists(ADAPTER_PATH)) or adapter_incompatible:
        print("No policy adapter found on disk. Initializing default LoRA adapter...")
        init_model = load_model_optimized(adapter_path=None, for_training=False)
        init_model.save_pretrained(ADAPTER_PATH)
        del init_model
        free_memory()
        print(f"Initialized default adapter at {ADAPTER_PATH}")

    print("Loading policy adapter in vLLM...")
    loaded = vllm_reload_with_lora(adapter_path=ADAPTER_PATH, adapter_name=ADAPTER_NAME)
    print(f"Policy adapter loaded in vLLM: {loaded}")


def train_step(step_idx: int, adapter_exists_on_disk: bool):
    # === 1. ROLLOUTS (vLLM awake) ===
    use_lora = adapter_exists_on_disk and vllm_is_lora_loaded(ADAPTER_NAME)
    print(
        f"[step {step_idx}] Running rollouts with "
        f"{'LoRA adapter' if use_lora else 'base model'}..."
    )

    rollouts, rewards = asyncio.run(
        run_rollouts(N_ROLLOUTS, max_turn_number=MAX_TURN_NUMBER, use_lora=use_lora)
    )
    reward_mean = statistics.mean(rewards)
    reward_std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    print(f"[step {step_idx}] Rewards: mean={reward_mean:.3f}, std={reward_std:.3f}")

    valid_pairs = [(r, rw) for r, rw in zip(rollouts, rewards) if _rollout_has_image(r)]
    if len(valid_pairs) != len(rollouts):
        print(
            f"Filtered invalid rollouts without images: "
            f"{len(rollouts) - len(valid_pairs)}/{len(rollouts)}"
        )

    # === 2. SLEEP vLLM ===
    print(f"[step {step_idx}] Putting vLLM to sleep...")
    vllm_sleep(level=1)
    free_memory()

    if not valid_pairs:
        print("No valid rollouts for batching; skipping optimization for this step.")
        print("Waking vLLM and reloading LoRA adapter...")
        vllm_reload_with_lora()
        wandb.log(
            {
                "rollout/reward_mean": reward_mean,
                "rollout/reward_std": reward_std,
                "rollout/reward_min": min(rewards),
                "rollout/reward_max": max(rewards),
                "rollout/valid_count": 0,
                "step": step_idx,
            }
        )
        return reward_mean

    # === 3. PREPARE DATA (CPU) ===
    rollouts = [r for r, _ in valid_pairs]
    rewards = [rw for _, rw in valid_pairs]
    batch = prepare_batch(rollouts)
    mean_r = statistics.mean(rewards)
    std_r = statistics.stdev(rewards) if len(rewards) > 1 else 1.0
    advantages = torch.tensor([(r - mean_r) / (std_r + VAR_EPS) for r in rewards])

    # === 4. LOAD BASE MODEL FOR REF LOGPS ===
    ref_model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
    }
    if USE_FLASH_ATTN:
        ref_model_kwargs["attn_implementation"] = "flash_attention_2"

    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_SOURCE, local_files_only=LOCAL_FILES_ONLY, **ref_model_kwargs
    )
    base_model.eval()

    # === 5. COMPUTE REF LOGPS ===
    print(f"[step {step_idx}] Computing reference log probs...")
    model_inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch["model_inputs"].items()
    }
    with torch.no_grad():
        ref_logps = get_per_token_logps(base_model, model_inputs).cpu()

    del base_model
    free_memory()

    # === 6. LOAD POLICY MODEL WITH OPTIMIZATIONS ===
    policy_model = load_model_optimized(
        adapter_path=ADAPTER_PATH if adapter_exists_on_disk else None,
        for_training=True,
    )
    policy_model.train()

    optimizer = torch.optim.AdamW(
        (p for p in policy_model.parameters() if p.requires_grad),
        lr=LR,
        fused=True,
    )

    # === 7. TRAINING ITERS ===
    completion_mask = batch["completion_mask"].to(device)
    ref_logps = ref_logps.to(device)
    adv = advantages.to(device).unsqueeze(1)
    mask_tokens = int(completion_mask.sum().item())
    print(f"[step {step_idx}] Training tokens in completion mask: {mask_tokens}")
    if mask_tokens == 0:
        print(
            "No completion tokens selected for training. "
            "No assistant token spans were detected; skipping optimization."
        )
        policy_model.save_pretrained(ADAPTER_PATH)
        del policy_model, optimizer
        del model_inputs, completion_mask, ref_logps, adv
        free_memory()
        print("Waking vLLM and loading LoRA adapter...")
        vllm_reload_with_lora()
        return reward_mean

    # Old policy logprobs are from the frozen snapshot before optimizer updates.
    policy_model.eval()
    with torch.no_grad():
        old_logps = get_per_token_logps(policy_model, model_inputs).detach()
    policy_model.train()

    for it in range(N_ITERS):
        optimizer.zero_grad(set_to_none=True)

        current_logps = get_per_token_logps(policy_model, model_inputs)

        min_len = min(
            current_logps.shape[1],
            old_logps.shape[1],
            ref_logps.shape[1],
            completion_mask.shape[1],
        )
        cur = current_logps[:, :min_len]
        old = old_logps[:, :min_len]
        ref = ref_logps[:, :min_len]
        mask = completion_mask[:, :min_len]

        ratio = torch.exp(cur - old)
        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        policy_loss = -torch.min(ratio * adv, clipped * adv)
        kl = cur - ref
        loss = ((policy_loss + KL_COEF * kl) * mask).sum() / (mask.sum() + 1e-8)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            kl_mean = (kl * mask).sum() / (mask.sum() + 1e-8)
            policy_loss_mean = (policy_loss * mask).sum() / (mask.sum() + 1e-8)
            ratio_mean = (ratio * mask).sum() / (mask.sum() + 1e-8)

        wandb.log(
            {
                "train/loss": loss.item(),
                "train/policy_loss": policy_loss_mean.item(),
                "train/kl": kl_mean.item(),
                "train/ratio_mean": ratio_mean.item(),
                "train/iter": step_idx * N_ITERS + it,
            }
        )

        print(f"[step {step_idx}] Iter {it}: loss={loss.item():.4f}")

        del current_logps, cur, ratio, clipped, policy_loss, kl, loss
        if it < N_ITERS - 1:
            free_memory()

    # === 8. SAVE & CLEANUP ===
    policy_model.save_pretrained(ADAPTER_PATH)
    del policy_model, optimizer
    del model_inputs, completion_mask, old_logps, ref_logps, adv
    free_memory()

    # === 9. WAKE VLLM WITH UPDATED LORA ===
    print(f"[step {step_idx}] Waking vLLM and loading updated LoRA adapter...")
    vllm_reload_with_lora()

    wandb.log(
        {
            "rollout/reward_mean": reward_mean,
            "rollout/reward_std": reward_std,
            "rollout/reward_min": min(rewards),
            "rollout/reward_max": max(rewards),
            "step": step_idx,
        }
    )

    return reward_mean


def main():
    n_steps = N_STEPS

    wandb.init(
        project="wordle-grpo",
        mode='disabled',
        config={
            "model": MODEL_NAME,
            "n_rollouts": N_ROLLOUTS,
            "n_iters": N_ITERS,
            "clip_eps": CLIP_EPS,
            "kl_coef": KL_COEF,
            "lr": LR,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "flash_attn": USE_FLASH_ATTN,
        },
    )

    print("Memory optimizations enabled:")
    print(f"  - Model name: {MODEL_NAME}")
    print(f"  - Model source: {MODEL_SOURCE}")
    print(f"  - Flash Attention 2: {USE_FLASH_ATTN}")
    print(f"  - Gradient Checkpointing: {GRADIENT_CHECKPOINTING}")
    print(f"  - Adapter path: {ADAPTER_PATH}")
    print()

    print("Initializing vLLM with base model...")
    vllm_reload_weights()
    initialize_policy_adapter_if_missing()

    for step in range(n_steps):
        print(f"\n{'=' * 50}")
        print(f"=== Step {step} ({step + 1}/{n_steps}) ===")
        print(f"{'=' * 50}")
        print(f"Progress: completed {step}/{n_steps}, now running step {step + 1}/{n_steps}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            wandb.log(
                {
                    "gpu/memory_allocated_gb": allocated,
                    "gpu/memory_reserved_gb": reserved,
                    "step": step,
                }
            )

        adapter_on_disk = adapter_exists(ADAPTER_PATH)
        print(f"Adapter exists on disk: {adapter_on_disk}")

        mean_reward = train_step(step, adapter_exists_on_disk=adapter_on_disk)
        print(f"Mean reward: {mean_reward:.3f}")

    wandb.finish()


if __name__ == "__main__":
    main()
