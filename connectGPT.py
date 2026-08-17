"""Shared, environment-configured OpenAI helpers."""

from __future__ import annotations

import os
import json
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client() -> OpenAI:
    """Create a client without ever embedding credentials in source code."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env")
    return OpenAI(api_key=api_key)


def send_to_chatgpt(text: str, system: str = "You are a helpful assistant.") -> str:
    response = get_client().chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    return response.choices[0].message.content or ""


def chat_json(system: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Ask the model for a JSON object and parse it."""
    response = get_client().chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("OpenAI response must be a JSON object")
    return result


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello!"
    print(send_to_chatgpt(prompt))
