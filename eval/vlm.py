import base64
import re
from openai import AsyncOpenAI

from .prompts import SYSTEM_PROMPT


class VLMClient:
    def __init__(self, model: str, api_key: str):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.messages: list[dict] = []

    def reset(self) -> None:
        """Reset conversation history for a new episode."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def query(self, screenshot: bytes, turn: int) -> str:
        """Send screenshot to VLM and get the next action. Maintains conversation history."""
        b64_image = base64.b64encode(screenshot).decode()

        self.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Turn {turn}. What is your next move?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        })

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=500
        )

        assistant_content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_content})

        return assistant_content


def parse_response(response: str) -> str | None:
    """Extract command from backticks in VLM response."""
    if response is None:
        return None

    match = re.search(r'`([^`]+)`', response)
    if match:
        cmd = match.group(1).strip().lower()
        # Validate format
        if cmd in ['left', 'right']:
            return cmd
        if re.match(r'^forward\s+\d+$', cmd):
            return cmd

    return None
