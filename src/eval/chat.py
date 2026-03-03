import argparse
import os

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat with a local vLLM-served model.")
    parser.add_argument("--model", type=str, default="gw", help="Model ID exposed by vLLM (/v1/models).")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "none"),
        help="API key for OpenAI-compatible server.",
    )
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print(f"Chatting with model={args.model} via {args.base_url}")
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
