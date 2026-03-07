import argparse
import base64
import mimetypes
import os
from pathlib import Path
import re

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat with a local vLLM-served model.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Model ID exposed by vLLM (/v1/models).")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY", "none"),
        help="API key for OpenAI-compatible server.",
    )
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()

def _image_path_to_data_url(image_path: str) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"

def _extract_image_path(user_text: str) -> tuple[str | None, str]:
    # Match .png/.jpg/.jpeg path-like tokens (supports quoted paths).
    match = re.search(r'["\']?([^\s"\']+\.(?:png|jpg|jpeg))["\']?', user_text, flags=re.IGNORECASE)
    if not match:
        return None, user_text
    raw_path = match.group(1)
    clean_text = (user_text[: match.start()] + user_text[match.end() :]).strip()
    return raw_path, clean_text


def main() -> int:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print(f"Chatting with model={args.model} via {args.base_url}")
    print("Include a .png/.jpg path in your message to attach that image for the turn.")
    print("Type /reset to clear history, /exit to quit.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not user_text:
            continue
        if user_text in {"/exit", "exit", "quit"}:
            print("Exiting.")
            return 0
        if user_text == "/reset":
            messages = [{"role": "system", "content": args.system}] if args.system else []
            print("History reset.")
            continue

        image_path, clean_user_text = _extract_image_path(user_text)
        if image_path:
            try:
                image_data_url = _image_path_to_data_url(image_path)
            except Exception as exc:  # noqa: BLE001
                print(f"Image load failed ({image_path}): {exc}")
                continue
            if not clean_user_text:
                clean_user_text = "Please analyze this image."
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": clean_user_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": user_text})

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Request failed: {exc}")
            continue

        assistant_text = (resp.choices[0].message.content or "").strip()
        print(f"Assistant: {assistant_text}")
        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    raise SystemExit(main())
