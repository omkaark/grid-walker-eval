# Run-Stability Plan (No Code Changes Yet)

## What is failing (root causes)
1. **Runtime LoRA API mismatch**
- Your server returns `404` for:
  - `POST /v1/unload_lora_adapter`
  - `POST /v1/load_lora_adapter`
- This means runtime load/unload endpoints are not available in your current vLLM deployment mode.
- Result: `Policy adapter loaded in vLLM: False` can be printed even if requests still run with a static alias.

2. **Engine dies after sleep/wake cycle**
- Reproduction shows:
  - Baseline chat works.
  - After `sleep(level=1)` + `wake_up(weights)` + load/unload attempts, first post-wake multimodal calls return `error` payloads.
  - `/health` becomes `503` and engine transitions to `EngineDeadError` / CUDA illegal address.
- This is the immediate blocker for run stability.

3. **Config drift from env var names (secondary issue)**
- Current code reads short env vars (`MODEL`, `N_ROLLOUTS`, etc.), while older commands used `GRID_WALKER_*` and `MODEL_NAME`.
- This can make runs appear inconsistent (wrong rollout count/model settings).

## Goal constraints
- Keep **sleep/wake** behavior.
- Do only changes that directly improve run reliability.

## Proposed solution paths (pick one)

### Option A (recommended): Sleep/Wake with static LoRA only
Use sleep/wake, but **do not call runtime LoRA load/unload endpoints** when unsupported.

Steps:
1. Add a one-time capability check at startup:
   - If `/v1/load_lora_adapter` is 404, set `runtime_lora_supported = False`.
2. Keep `vllm_sleep(level=1)` and `wake_up(weights)`.
3. If `runtime_lora_supported == False`, skip unload/load calls entirely.
4. Keep querying model alias that already works (e.g., `policy`) without runtime reload.
5. Add a post-wake health/checkpoint call before rollouts; fail fast if unhealthy.

Why this helps:
- Preserves sleep/wake.
- Removes the unsupported API calls that currently happen every step.
- Minimizes code churn and keeps your current serving mode.

### Option B: Sleep/Wake + runtime LoRA endpoints fully enabled
If you need true runtime adapter reload each step, run vLLM in mode that exposes load/unload APIs.

Steps:
1. Start vLLM with runtime LoRA endpoint support enabled (deployment config change).
2. Verify both endpoints return non-404 before training starts.
3. Keep current reload logic.

Why this helps:
- Restores original intended reload behavior.
- Requires server launch/config changes, not just trainer code.

## Additional hardening (small, high impact)
1. Treat OpenAI `chat/completions` responses with top-level `error` as failures even when HTTP is 200.
2. Use one consistent env var namespace for run commands (either short names or `GRID_WALKER_*`, not mixed).
3. Add explicit log line each step:
   - sleep status
   - wake status
   - runtime_lora_supported flag

## Recommended execution order
1. Implement Option A first (fastest path to stable run).
2. Run 5-step smoke test with `N_ROLLOUTS=8`.
3. If stable, scale back up.
4. Only then evaluate Option B if runtime adapter hot-reload is strictly required.
