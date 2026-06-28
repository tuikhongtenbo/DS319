"""
Inference predictor for Qwen via DashScope (0-shot).
"""

import base64
import os
from pathlib import Path
from typing import List

from openai import OpenAI


DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _read_env_file_value(name: str) -> str:
    search_dirs = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    for directory in dict.fromkeys(search_dirs):
        env_path = directory / ".env"
        if not env_path.exists():
            continue
        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() == name:
                    return value.strip().strip('"').strip("'")
    return ""


def resolve_qwen_api_key(api_key: str = "") -> str:
    resolved_api_key = api_key or os.getenv("QWEN_API_KEY", "") or _read_env_file_value("QWEN_API_KEY")
    if not resolved_api_key:
        raise ValueError("Qwen API key is required. Pass --api_key or set QWEN_API_KEY in the environment or .env file.")
    return resolved_api_key


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class QwenZeroShotPredictor:
    def __init__(self, model_name: str, api_key: str = ""):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=resolve_qwen_api_key(api_key),
            base_url=DASHSCOPE_BASE_URL,
        )

    def predict(self, image_path: str, question: str, options: List[str]) -> str:
        base64_image = encode_image(image_path)

        system_prompt = (
            "You are currently a senior expert in spatial relation reasoning. "
            "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
            "Note that you only need to choose one option from all options without explaining any reason."
        )

        user_prompt = f"Input: Question: {question}, Options: {'; '.join(options)}.\nOutput:"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=20,
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )

        return response.choices[0].message.content.strip()