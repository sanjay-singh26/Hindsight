"""Minimal OpenAI-chat-completions-shaped adapter over a local mlx-vlm model.

Lets the same call sites used for the LLMGateway-backed rescore_*.py
scripts (``client.chat.completions.create(model=..., messages=[...])``)
run against a local open-weights vision-language model instead, with
zero API cost. Supports both vision (image_url content blocks) and
text-only messages, and multiple images per call (for multi-page PDFs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class _Usage:
    cost: float = 0.0


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list
    usage: _Usage


def _extract_images_and_text(content) -> tuple[list[str], str]:
    """Pull out any base64 data-URI images and the text from an OpenAI-style
    multi-part content list, writing images to temp PNG files (mlx-vlm's
    generate() takes file paths, not raw bytes)."""
    import base64
    import io
    import tempfile

    if isinstance(content, str):
        return [], content

    image_paths = []
    text_parts = []
    for part in content:
        if part.get("type") == "image_url":
            url = part["image_url"]["url"]
            # data:image/png;base64,<b64>
            header, b64data = url.split(",", 1)
            data = base64.b64decode(b64data)
            ext = ".jpg" if "jpeg" in header else ".png"
            f = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            f.write(data)
            f.close()
            image_paths.append(f.name)
        elif part.get("type") == "text":
            text_parts.append(part["text"])
    return image_paths, " ".join(text_parts)


class _Completions:
    def __init__(self, model, processor, config):
        self._model = model
        self._processor = processor
        self._config = config

    def create(self, model: str, messages: list, max_tokens: int = 1024, **kwargs) -> _Response:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        chat_messages = []
        all_images: list[str] = []
        for m in messages:
            imgs, text = _extract_images_and_text(m["content"])
            all_images.extend(imgs)
            chat_messages.append({"role": m["role"], "content": text})

        formatted = apply_chat_template(
            self._processor, self._config, chat_messages, num_images=len(all_images)
        )
        result = generate(
            self._model,
            self._processor,
            formatted,
            all_images if all_images else None,
            max_tokens=max_tokens,
            verbose=False,
        )
        text = result.text if hasattr(result, "text") else str(result)
        return _Response(choices=[_Choice(message=_Message(content=text))], usage=_Usage(cost=0.0))


class _Chat:
    def __init__(self, completions: _Completions):
        self.completions = completions


class LocalQwenClient:
    """Drop-in replacement for `openai.OpenAI(...)` backed by a local mlx-vlm model."""

    _model = None
    _processor = None
    _config = None

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            import os

            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "Qwen2-VL-7B-Instruct-4bit",
            )
        if LocalQwenClient._model is None:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            LocalQwenClient._model, LocalQwenClient._processor = load(model_path)
            LocalQwenClient._config = load_config(model_path)
        self.chat = _Chat(_Completions(LocalQwenClient._model, LocalQwenClient._processor, LocalQwenClient._config))
