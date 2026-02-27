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

    match = re.search(r'`([^`]+)`', response)
    if match:
        cmd = match.group(1).strip().lower()
        # Validate format
        if cmd in ['left', 'right'] or re.match(r'^forward\s+\d+$', cmd):
            return cmd

    return None
