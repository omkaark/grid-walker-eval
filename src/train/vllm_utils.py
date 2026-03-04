import os

import requests

VLLM_BASE = "http://localhost:8000"
ADAPTER_NAME = "policy"
ADAPTER_PATH = os.path.abspath(os.getenv("ADAPTER_PATH", "./lora_adapter"))


def vllm_sleep(level: int = 1):
    return requests.post(f"{VLLM_BASE}/sleep?level={level}", timeout=30)

def vllm_reload_weights() -> None:
    requests.post(f"{VLLM_BASE}/wake_up?tags=weights", timeout=30)


def adapter_exists(adapter_path: str = ADAPTER_PATH) -> bool:
    adapter_config = os.path.join(os.path.abspath(adapter_path), "adapter_config.json")
    return os.path.exists(adapter_config)

def vllm_reload_with_lora(
    adapter_path: str = ADAPTER_PATH,
    adapter_name: str = ADAPTER_NAME,
) -> bool:
    vllm_reload_weights()

    adapter_path = os.path.abspath(adapter_path)
    if not adapter_exists(adapter_path):
        return False

    load_resp = requests.post(
        f"{VLLM_BASE}/v1/load_lora_adapter",
        headers={"Content-Type": "application/json"},
        json={
            "lora_name": adapter_name,
            "lora_path": adapter_path,
            "load_inplace": True,
        },
        timeout=60,
    )
    return load_resp.status_code == 200
