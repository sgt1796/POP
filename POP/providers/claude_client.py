"""Claude API client implementation via Anthropic's OpenAI compatibility layer.

This module provides a :class:`LLMClient` implementation that uses the
OpenAI Python SDK against Anthropic's OpenAI-compatible endpoint.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from os import getenv
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient


try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


CLAUDE_OPENAI_BASE_URL = "https://api.anthropic.com/v1/"

_CLAUDE_SUPPORTED_FIELDS = {
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "top_p",
    "parallel_tool_calls",
    "stop",
    "n",
    "extra_body",
    "extra_headers",
    "extra_query",
    "timeout",
}

_CLAUDE_IGNORED_FIELDS = {
    "logprobs",
    "metadata",
    "response_format",
    "prediction",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "service_tier",
    "audio",
    "logit_bias",
    "store",
    "user",
    "modalities",
    "top_logprobs",
    "reasoning_effort",
}


def _warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=3)


def _content_to_text(content: Any) -> str:
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


def normalize_claude_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Hoist all system/developer messages into one leading system message."""
    normalized: List[Dict[str, Any]] = []
    system_parts: List[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user"))
        if role in {"system", "developer"}:
            system_parts.append(_content_to_text(msg.get("content", "")))
            continue
        normalized.append(dict(msg))
    if system_parts:
        normalized.insert(0, {"role": "system", "content": "\n".join(system_parts)})
    return normalized


def _normalize_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    normalized: List[Dict[str, Any]] = []
    saw_strict = False
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue

        function_decl: Optional[Dict[str, Any]] = None
        tool_type = tool.get("type")
        nested_function = tool.get("function")

        if tool_type in (None, "function") and isinstance(nested_function, dict):
            function_decl = deepcopy(nested_function)
        elif tool_type in (None, "function"):
            function_decl = deepcopy(tool)
            function_decl.pop("type", None)
            function_decl.pop("function", None)

        if not function_decl or not function_decl.get("name"):
            continue

        if function_decl.pop("strict", None) is not None:
            saw_strict = True

        normalized.append({"type": "function", "function": function_decl})

    if saw_strict:
        _warn(
            "Claude OpenAI compatibility ignores tool strictness; POP removed `strict` from the request."
        )
    return normalized or None


def _normalize_temperature(temperature: float) -> float:
    try:
        parsed = float(temperature)
    except (TypeError, ValueError):
        return 0.0
    clamped = max(0.0, min(1.0, parsed))
    if clamped != parsed:
        _warn("Claude OpenAI compatibility only supports temperature values between 0 and 1; POP clamped it.")
    return clamped


def _normalize_stop_sequences(stop: Any) -> Any:
    if isinstance(stop, str):
        if stop.strip():
            return stop
        _warn("Claude only supports non-whitespace stop sequences; POP removed an empty stop sequence.")
        return None
    if isinstance(stop, list):
        filtered = [item for item in stop if isinstance(item, str) and item.strip()]
        if len(filtered) != len(stop):
            _warn("Claude only supports non-whitespace stop sequences; POP removed unsupported entries.")
        return filtered or None
    return stop


def prepare_claude_request(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build one Claude-compatible OpenAI SDK request plus usage metadata."""
    request_payload: Dict[str, Any] = {
        "model": model,
        "messages": [],
        "temperature": _normalize_temperature(temperature),
    }

    normalized_messages = normalize_claude_messages(messages)
    images = kwargs.pop("images", None)
    if images and not isinstance(images, list):
        images = [images]

    for msg in normalized_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if images and role == "user":
            if isinstance(content, list):
                multi: List[Any] = list(content)
            else:
                multi = [{"type": "text", "text": _content_to_text(content)}]
            for img in images:
                if isinstance(img, str) and img.startswith("http"):
                    multi.append({"type": "image_url", "image_url": {"url": img}})
                else:
                    multi.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
            content = multi
        payload_msg: Dict[str, Any] = {"role": role, "content": content}
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            payload_msg["tool_calls"] = tool_calls
        tool_call_id = msg.get("tool_call_id") or msg.get("toolCallId")
        if tool_call_id:
            payload_msg["tool_call_id"] = tool_call_id
        request_payload["messages"].append(payload_msg)

    tools = _normalize_tools(kwargs.pop("tools", None))
    tool_choice = kwargs.pop("tool_choice", None)
    if tools:
        request_payload["tools"] = tools
        if tool_choice is not None:
            request_payload["tool_choice"] = tool_choice
    elif tool_choice is not None:
        _warn("POP ignored `tool_choice` because no tools were supplied for Claude.")

    response_format = kwargs.pop("response_format", None)
    if response_format is not None:
        _warn("Claude OpenAI compatibility ignores `response_format`; POP omitted it from the request.")

    if "n" in kwargs and kwargs["n"] not in (None, 1):
        _warn("Claude OpenAI compatibility requires `n=1`; POP normalized the request to `n=1`.")
    if "n" in kwargs:
        request_payload["n"] = 1
        kwargs.pop("n", None)

    stop = _normalize_stop_sequences(kwargs.pop("stop", None))
    if stop is not None:
        request_payload["stop"] = stop

    for key in list(kwargs.keys()):
        value = kwargs[key]
        if value is None:
            kwargs.pop(key, None)
            continue
        if key in _CLAUDE_IGNORED_FIELDS:
            _warn(f"Claude OpenAI compatibility ignores `{key}`; POP omitted it from the request.")
            kwargs.pop(key, None)
            continue
        if key in _CLAUDE_SUPPORTED_FIELDS:
            request_payload[key] = kwargs.pop(key)

    for key in list(kwargs.keys()):
        _warn(f"POP omitted unsupported Claude compatibility field `{key}`.")
        kwargs.pop(key, None)

    return {
        "payload": request_payload,
        "messages": request_payload["messages"],
        "tools": request_payload.get("tools"),
        "response_format": None,
    }


class ClaudeClient(LLMClient):
    """Client for Anthropic Claude models via OpenAI compatibility."""

    provider_name = "claude"
    supports_response_format = False

    def __init__(self, model: str = "claude-haiku-4-5", base_url: str = CLAUDE_OPENAI_BASE_URL) -> None:
        if OpenAI is None:
            raise ImportError("openai package is not installed. Install it to use ClaudeClient.")
        self.client = OpenAI(api_key=getenv("ANTHROPIC_API_KEY"), base_url=base_url)
        self.model_name = model
        self.last_request_meta: Optional[Dict[str, Any]] = None

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        prepared = prepare_claude_request(
            messages=messages,
            model=model or self.model_name,
            temperature=temperature,
            **kwargs,
        )
        self.last_request_meta = {
            "messages": prepared["messages"],
            "tools": prepared["tools"],
            "response_format": prepared["response_format"],
        }
        try:
            response = self.client.chat.completions.create(**prepared["payload"])
        except Exception as exc:
            raise RuntimeError(f"Claude chat_completion error: {exc}") from exc
        return response


__all__ = ["ClaudeClient", "CLAUDE_OPENAI_BASE_URL", "normalize_claude_messages", "prepare_claude_request"]
