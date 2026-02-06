"""Streaming support for LLM responses.

This module exposes a `stream` function compatible with the agent
loop's `StreamFn` type. It accepts `(model, context, options)` and
returns an async iterable of events with `type` keys, plus a
`result()` method for the final assistant message.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from .api_registry import get_client, get_default_model


def _message_to_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item.get("text") or ""))
                elif "thinking" in item:
                    parts.append(str(item.get("thinking") or ""))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _context_to_messages(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    system_prompt = context.get("system_prompt") or context.get("system")
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    for msg in context.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        text = _message_to_text(msg)
        messages.append({"role": role, "content": text})
    return messages


def _call_llm(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]) -> str:
    provider = model.get("provider") or model.get("api")
    if not provider:
        raise ValueError("Model provider is required")
    client = get_client(provider)
    if client is None:
        raise ValueError(f"Unknown provider: {provider}")
    model_id = model.get("id") or get_default_model(provider)
    messages = _context_to_messages(context)
    call_kwargs = dict(options or {})
    if "tools" not in call_kwargs and context.get("tools") is not None:
        call_kwargs["tools"] = context.get("tools")
    response = client.chat_completion(messages=messages, model=model_id, **call_kwargs)
    try:
        return response.choices[0].message.content
    except Exception:
        return str(response)


class MessageEventStream:
    def __init__(self, model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]) -> None:
        self._model = model
        self._context = context
        self._options = options
        self._final_message: Optional[Dict[str, Any]] = None
        self._started = False

    def _build_partial(self) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "stopReason": "stop",
            "content": [{"type": "text", "text": ""}],
            "api": self._model.get("api"),
            "provider": self._model.get("provider"),
            "model": self._model.get("id"),
            "usage": {},
            "timestamp": time.time(),
        }

    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        if self._started:
            return
        self._started = True
        partial = self._build_partial()
        yield {"type": "start", "partial": dict(partial)}
        yield {"type": "text_start", "contentIndex": 0, "partial": dict(partial)}
        try:
            text = await asyncio.to_thread(_call_llm, self._model, self._context, self._options)
        except Exception as exc:
            error_msg = dict(partial)
            error_msg["stopReason"] = "error"
            error_msg["errorMessage"] = str(exc)
            self._final_message = error_msg
            yield {"type": "error", "error": error_msg}
            return

        partial["content"][0]["text"] = text
        yield {"type": "text_delta", "contentIndex": 0, "delta": text, "partial": dict(partial)}
        yield {"type": "text_end", "contentIndex": 0, "content": text, "partial": dict(partial)}
        final = dict(partial)
        final["stopReason"] = "stop"
        self._final_message = final
        yield {"type": "done", "message": final}

    async def result(self) -> Dict[str, Any]:
        return self._final_message or {}


async def stream(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]) -> MessageEventStream:
    """Stream an LLM response using the StreamFn signature."""
    return MessageEventStream(model, context, options)


__all__ = ["stream", "MessageEventStream"]
