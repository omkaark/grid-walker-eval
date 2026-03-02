import base64
import re
from openai import OpenAI

from .prompts import SYSTEM_PROMPT


class VLMClient:
    def __init__(self, model: str, base_url: str = "https://openrouter.ai/api/v1", api_key: str = None):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.messages: list[dict] = []

    def reset(self) -> None:
        """Reset conversation history for a new episode."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def query(self, screenshot: bytes, turn: int) -> str:
        """Send screenshot to VLM and get the next action. Maintains conversation history."""
        b64_image = base64.b64encode(screenshot).decode()

        self.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Turn {turn}. What is your next move?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=2000
        )

        assistant_content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_content})

        return assistant_content


def parse_response(response: str) -> str | None:
    if response is None:
        return None

    text = response.strip().lower()

    # Prefer strict backtick format when present.
    match = re.search(r"`([^`]+)`", text)
    candidates: list[str] = []
    if match:
        candidates.append(match.group(1).strip())
    candidates.append(text)

    for candidate in candidates:
        # Accept exact control commands anywhere in text.
        if re.search(r"\bleft\b", candidate):
            return "left"
        if re.search(r"\bright\b", candidate):
            return "right"

        m = re.search(r"\bforward\s+(\d+)\b", candidate)
        if m:
            return f"forward {int(m.group(1))}"

    return None
