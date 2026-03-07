import asyncio
import gc
import os
import statistics
import time
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
    vllm_wake_up,
    vllm_reload_with_lora,
    vllm_sleep,
)

MODEL_NAME = os.getenv("MODEL", "Qwen/Qwen3-VL-2B-Instruct")
ADAPTER_PATH = os.path.abspath(os.getenv("ADAPTER_PATH", "./lora_adapter"))
MAX_TURN_NUMBER = int(os.getenv("MAX_TURNS", "10"))
N_ROLLOUTS = int(os.getenv("N_ROLLOUTS", "8"))
N_ITERS = int(os.getenv("N_ITERS", "4"))
ZERO_WIN_RETRIES = int(os.getenv("ZERO_WIN_RETRIES", "10"))
MAX_ROLLOUT_ERROR_RETRIES = int(os.getenv("MAX_ROLLOUT_ERROR_RETRIES", "10"))
ROLLOUT_WAIT_FOR_S = float(os.getenv("ROLLOUT_WAIT_FOR_S", "600"))
CLIP_EPS = 0.2
KL_COEF = 0.04
LR = 5e-7
LOG_RATIO_CLAMP = 20.0
GROUP_ADV_EPS = 1e-4

GRADIENT_CHECKPOINTING = os.getenv("GRADIENT_CHECKPOINTING", "1") == "1"
USE_FLASH_ATTN = os.getenv("USE_FLASH_ATTN", "0") == "1"

N_STEPS = int(os.getenv("N_STEPS", "100"))

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
    if any(token in name for token in ("visual", "vision", "image", "img")):
        return False
    if "lora_" not in name:
        return False
    return any(module_name in name for module_name in TEXT_LORA_TARGET_MODULES)

processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer
ASSISTANT_START_IDS = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
ASSISTANT_END_IDS = tokenizer.encode("<|im_end|>", add_special_tokens=False)
ASSISTANT_END_WITH_NEWLINE_IDS = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

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


def _find_subsequence(token_ids: list[int], pattern: list[int], start: int) -> int:
    """Return first index >= start where pattern appears, else -1."""
    if not pattern:
        return -1
    last = len(token_ids) - len(pattern) + 1
    for i in range(start, max(start, last)):
        if token_ids[i : i + len(pattern)] == pattern:
            return i
    return -1


def _build_assistant_completion_mask(token_ids: list[int]) -> torch.Tensor:
    """
    Mark assistant content spans via chat-template delimiters:
    <|im_start|>assistant\\n ... <|im_end|>\\n.
    """
    seq_len = len(token_ids)
    mask = torch.zeros(seq_len, dtype=torch.float32)
    if seq_len == 0:
        return mask
    if not ASSISTANT_START_IDS or not ASSISTANT_END_IDS:
        return mask

    pos = 0
    while pos < seq_len:
        start_idx = _find_subsequence(token_ids, ASSISTANT_START_IDS, pos)
        if start_idx < 0:
            break
        content_start = start_idx + len(ASSISTANT_START_IDS)
        end_ids = ASSISTANT_END_WITH_NEWLINE_IDS or ASSISTANT_END_IDS
        end_idx = _find_subsequence(token_ids, end_ids, content_start)
        if end_idx < 0:
            end_ids = ASSISTANT_END_IDS
            end_idx = _find_subsequence(token_ids, end_ids, content_start)
        if end_idx < 0:
            break
        if content_start < end_idx:
            mask[content_start:end_idx] = 1.0
        pos = end_idx + len(end_ids)

    return mask


def prepare_batch(rollouts):
    """Build a multimodal batch and aligned completion masks for PPO-style loss."""
    texts: list[str] = []
    all_images = []

    for rollout in rollouts:
        messages = rollout.request["messages"]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs, _ = process_vision_info(messages)

        texts.append(text)
        all_images.append(image_inputs)

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
        seq_len = int(model_inputs["attention_mask"][i].sum().item())
        token_ids = model_inputs["input_ids"][i, :seq_len].tolist()
        mask = _build_assistant_completion_mask(token_ids)
        completion_mask[i, :seq_len] = mask[:seq_len]

    return {
        "model_inputs": model_inputs,
        "completion_mask": completion_mask[:, 1:],
    }


def _group_normalized_advantages(
    rewards: list[float],
    rollouts: list[Any],
    eps: float = GROUP_ADV_EPS,
) -> tuple[torch.Tensor, int, int]:
    """Dr. GRPO: center rewards per group (no std normalization)."""
    grouped_indices: dict[int, list[int]] = {}
    for idx, rollout in enumerate(rollouts):
        group_id = rollout.group_id if getattr(rollout, "group_id", None) is not None else idx
        grouped_indices.setdefault(int(group_id), []).append(idx)

    advantages = torch.zeros(len(rewards), dtype=torch.float32)
    nonzero_signal_groups = 0

    for _, indices in grouped_indices.items():
        group_rewards = [rewards[i] for i in indices]
        mean_r = statistics.mean(group_rewards)
        centered = [r - mean_r for r in group_rewards]
        if max((abs(v) for v in centered), default=0.0) >= eps:
            nonzero_signal_groups += 1
        for i in indices:
            advantages[i] = float(rewards[i] - mean_r)

    return advantages, nonzero_signal_groups, len(grouped_indices)


def initialize_policy_adapter_if_missing() -> None:
    """
    Ensure a policy LoRA adapter exists and is loaded into vLLM.
    If missing, create a default-initialized adapter (no-op at start), save it, then load in vLLM.
    """
    if not adapter_exists(ADAPTER_PATH):
        print("No policy adapter found on disk. Initializing default LoRA adapter...")
        init_model_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "cuda",
        }
        if USE_FLASH_ATTN:
            init_model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        base_model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME, **init_model_kwargs
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=list(TEXT_LORA_TARGET_MODULES),
            lora_dropout=0.05,
        )
        init_model = get_peft_model(base_model, lora_config)
        init_model.save_pretrained(ADAPTER_PATH)
        del init_model
        free_memory()
        print(f"Initialized default adapter at {ADAPTER_PATH}")

    print("Loading policy adapter in vLLM...")
    loaded = vllm_reload_with_lora(adapter_path=ADAPTER_PATH, adapter_name=ADAPTER_NAME)
    print(f"Policy adapter loaded in vLLM: {loaded}")


def train_step(step_idx: int):
    # === 1. ROLLOUTS (vLLM awake) ===
    use_lora = True
    rollouts = []
    rewards = []
    reward_mean = 0.0
    reward_std = 0.0
    win_ratio = 0.0
    wins = 0
    advantages = torch.tensor([], dtype=torch.float32)
    nonzero_signal_groups = 0
    n_groups = 0
    adv_abs_mean = 0.0
    rollout_error_retries = 0 # count

    for attempt in range(ZERO_WIN_RETRIES + 1):
        print(
            f"[step {step_idx}] Running rollouts attempt {attempt + 1}/{ZERO_WIN_RETRIES + 1} with "
            f"{'LoRA adapter' if use_lora else 'base model'}..."
        )
        try:
            rollouts, rewards = asyncio.run(
                asyncio.wait_for(
                    run_rollouts(N_ROLLOUTS, max_turn_number=MAX_TURN_NUMBER, use_lora=use_lora),
                    timeout=ROLLOUT_WAIT_FOR_S,
                )
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[step {step_idx}] rollout attempt {rollout_error_retries}/{MAX_ROLLOUT_ERROR_RETRIES} failed: {e}")
            rollout_error_retries += 1
            if rollout_error_retries >= MAX_ROLLOUT_ERROR_RETRIES:
                raise
            continue
        reward_mean = statistics.mean(rewards)
        reward_std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        wins = sum(1 for r in rollouts if r.won)
        win_ratio = wins / len(rollouts)
        advantages, nonzero_signal_groups, n_groups = _group_normalized_advantages(rewards, rollouts)
        adv_abs_mean = float(advantages.abs().mean().item()) if len(advantages) > 0 else 0.0

        if wins > 0:
            break
        if attempt < ZERO_WIN_RETRIES:
            print(f"[step {step_idx}] 0/{len(rollouts)} wins. Retrying rollouts...")
            continue

        print(
            f"[step {step_idx}] 0/{len(rollouts)} wins after {ZERO_WIN_RETRIES + 1} attempts. "
            "Skipping training for this step."
        )
        return reward_mean

    print(f"[step {step_idx}] Rewards: mean={reward_mean:.3f}, std={reward_std:.3f}")
    if adv_abs_mean < 1e-6:
        print(
            "All group-normalized advantages are ~0 (likely zero reward variance per group). "
            "Skipping optimization to avoid KL-only drift."
        )
        return reward_mean

    # === 2. SLEEP vLLM ===
    print(f"[step {step_idx}] Putting vLLM to sleep...")
    vllm_sleep(level=1)
    free_memory()

    # === 3. PREPARE DATA (CPU) ===
    print(f"[step {step_idx}] Preparing data...")
    batch = prepare_batch(rollouts)

    # === 4. LOAD BASE MODEL FOR REF LOGPS ===
    ref_model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
    }
    if USE_FLASH_ATTN:
        ref_model_kwargs["attn_implementation"] = "flash_attention_2"

    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, **ref_model_kwargs
    )
    base_model.eval()

    # === 5. COMPUTE REF LOGPS ===
    print(f"[step {step_idx}] Computing reference log probs...")
    model_inputs = {
        k: v.to("cuda") if torch.is_tensor(v) else v
        for k, v in batch["model_inputs"].items()
    }
    with torch.no_grad():
        ref_logps = get_per_token_logps(base_model, model_inputs).cpu()

    del base_model
    free_memory()

    # === 6. LOAD POLICY MODEL WITH OPTIMIZATIONS ===
    policy_model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
    }
    if USE_FLASH_ATTN:
        policy_model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    base_policy_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, **policy_model_kwargs
    )
    if GRADIENT_CHECKPOINTING:
        base_policy_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled")
    policy_model = PeftModel.from_pretrained(base_policy_model, ADAPTER_PATH, is_trainable=True)
    print(f"Loaded existing adapter from {ADAPTER_PATH}")
    for name, param in policy_model.named_parameters():
        param.requires_grad = _is_text_lora_param(name)
    policy_model.train()

    optimizer = torch.optim.AdamW(
        (p for p in policy_model.parameters() if p.requires_grad),
        lr=LR,
        fused=True,
    )

    # === 7. TRAINING ITERS ===
    completion_mask = batch["completion_mask"].to("cuda")
    ref_logps = ref_logps.to("cuda")
    adv = advantages.to("cuda").unsqueeze(1)
    mask_tokens = int(completion_mask.sum().item())
    print(f"[step {step_idx}] Training tokens in completion mask: {mask_tokens}")
    print(
        f"[step {step_idx}] Dr.GRPO groups with non-zero centered reward signal: "
        f"{nonzero_signal_groups}/{n_groups}"
    )

    # Old policy logprobs are from the frozen snapshot before optimizer updates.
    policy_model.eval()
    with torch.no_grad():
        old_logps = get_per_token_logps(policy_model, model_inputs).detach()
    policy_model.train()

    for it in range(N_ITERS):
        optimizer.zero_grad(set_to_none=True)

        current_logps = get_per_token_logps(policy_model, model_inputs)

        cur = current_logps
        old = old_logps
        ref = ref_logps
        mask = completion_mask

        log_ratio = (cur - old).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        policy_loss = -torch.min(ratio * adv, clipped * adv)
        log_ref_over_cur = (ref - cur).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
        approx_kl = torch.exp(log_ref_over_cur) - log_ref_over_cur - 1.0
        # Dr. GRPO: no token-length normalization.
        loss = ((policy_loss + KL_COEF * approx_kl) * mask).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            kl_mean = (approx_kl * mask).sum() / (mask.sum() + 1e-8)

        wandb.log(
            {
                "train/loss": loss.item(),
                "train/kl": kl_mean.item(),
                "train/step": step_idx,
            },
            step=step_idx * N_ITERS + it,
        )

        print(f"[step {step_idx}] Iter {it}: loss={loss.item():.4f}")

        del current_logps, cur, ratio, clipped, policy_loss, approx_kl, loss
        if it < N_ITERS - 1:
            free_memory()

    # === 8. SAVE & CLEANUP ===
    policy_model.save_pretrained(ADAPTER_PATH)
    del policy_model, base_policy_model, optimizer
    del model_inputs, completion_mask, old_logps, ref_logps, adv
    del batch, advantages, rollouts, rewards
    free_memory()

    # === 9. WAKE VLLM WITH UPDATED LORA ===
    print(f"[step {step_idx}] Waking vLLM and loading updated LoRA adapter from {ADAPTER_PATH}...")
    loaded = vllm_reload_with_lora(adapter_path=ADAPTER_PATH, adapter_name=ADAPTER_NAME)
    print(f"[step {step_idx}] Adapter reload status after update: {loaded}")

    wandb.log(
        {
            "rollout/reward_mean": reward_mean,
            "rollout/reward_std": reward_std,
            "rollout/win_ratio": win_ratio
        },
        step=step_idx * N_ITERS + (N_ITERS - 1),
    )

    return reward_mean


def main():
    n_steps = N_STEPS

    wandb.init(
        project="grid-walker",
        name="basic",
        # mode='disabled',
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

    print("Initializing vLLM with base model...")
    vllm_wake_up()
    initialize_policy_adapter_if_missing()

    for step in range(n_steps):
        print(f"\n{'=' * 50}")
        print(f"=== Step {step} ({step + 1}/{n_steps}) ===")
        print(f"{'=' * 50}")

        adapter_on_disk = adapter_exists(ADAPTER_PATH)
        print(f"Adapter exists on disk: {adapter_on_disk}")

        mean_reward = train_step(step)
        print(f"Mean reward: {mean_reward:.3f}")

        print(f"Progress: completed {step}/{n_steps}, now running step {step + 1}/{n_steps}")

    wandb.finish()


if __name__ == "__main__":
    main()
