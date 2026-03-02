import os
import time

import requests

VLLM_BASE = "http://localhost:8000"
ADAPTER_NAME = "policy"
ADAPTER_PATH = os.path.abspath("./lora_adapter")


def vllm_sleep(level: int = 1):
    return requests.post(f"{VLLM_BASE}/sleep?level={level}", timeout=30)


def _vllm_wake(tags: list[str] | None = None):
    url = f"{VLLM_BASE}/wake_up"
    if tags:
        url += f"?tags={','.join(tags)}"
    return requests.post(url, timeout=30)


def vllm_reload_weights() -> None:
    _vllm_wake(tags=["weights"])
    _vllm_wake(tags=["kv_cache"])


def _vllm_health_check() -> bool:
    try:
        return requests.get(f"{VLLM_BASE}/health", timeout=5).status_code == 200
    except requests.exceptions.RequestException:
        return False


def _vllm_wait_ready(timeout: int = 60) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if _vllm_health_check():
            return True
        time.sleep(1)
    return False


def _vllm_list_models() -> list[str]:
    try:
        resp = requests.get(f"{VLLM_BASE}/v1/models", timeout=10)
        if resp.status_code != 200:
            return []
        return [m["id"] for m in resp.json().get("data", [])]
    except requests.exceptions.RequestException:
        return []


def _vllm_unload_lora(adapter_name: str) -> bool:
    try:
        resp = requests.post(
            f"{VLLM_BASE}/v1/unload_lora_adapter",
            headers={"Content-Type": "application/json"},
            json={"lora_name": adapter_name},
            timeout=30,
        )
        return resp.status_code in (200, 404)
    except requests.exceptions.RequestException:
        return False


def _vllm_load_lora(adapter_name: str, adapter_path: str) -> bool:
    try:
        resp = requests.post(
            f"{VLLM_BASE}/v1/load_lora_adapter",
            headers={"Content-Type": "application/json"},
            json={"lora_name": adapter_name, "lora_path": adapter_path},
            timeout=60,
        )
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def adapter_exists(adapter_path: str = ADAPTER_PATH) -> bool:
    adapter_config = os.path.join(os.path.abspath(adapter_path), "adapter_config.json")
    return os.path.exists(adapter_config)


def vllm_is_lora_loaded(adapter_name: str = ADAPTER_NAME) -> bool:
    return adapter_name in _vllm_list_models()


def vllm_reload_with_lora(
    adapter_path: str = ADAPTER_PATH,
    adapter_name: str = ADAPTER_NAME,
) -> bool:
    vllm_reload_weights()
    if not _vllm_wait_ready(timeout=60):
        return False

    adapter_path = os.path.abspath(adapter_path)
    if not adapter_exists(adapter_path):
        return False

    _vllm_unload_lora(adapter_name)
    time.sleep(1.0)

    if not _vllm_load_lora(adapter_name, adapter_path):
        return False

    time.sleep(1.0)
    return adapter_name in _vllm_list_models()
