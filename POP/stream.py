"""Streaming support for LLM responses.

This module exposes a `stream` function compatible with the agent
loop's `StreamFn` type. It accepts `(model, context, options)` and
returns an async iterable of events with `type` keys, plus a
`result()` method for the final assistant message.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

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


def _context_to_messages(context: Dict[str, Any], provider: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    provider = (provider or "").lower()
    # Gemini uses the OpenAI-compatible chat completions API, so treat it like OpenAI.
    openai_like = {"openai", "deepseek", "doubao", "gemini"}
    system_prompt = context.get("system_prompt") or context.get("system")
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    for msg in context.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        if provider in openai_like:
            if role == "toolResult":
                tool_call_id = msg.get("toolCallId") or msg.get("tool_call_id")
                text = _message_to_text(msg)
                tool_msg = {"role": "tool", "content": text}
                if tool_call_id:
                    tool_msg["tool_call_id"] = tool_call_id
                messages.append(tool_msg)
                continue
            if role == "assistant":
                content_items = msg.get("content") or []
                text_parts: List[str] = []
                tool_calls: List[Dict[str, Any]] = []
                if isinstance(content_items, list):
                    for item in content_items:
                        if isinstance(item, dict) and item.get("type") == "toolCall":
                            call_id = item.get("id") or f"call_{len(tool_calls)}"
                            name = item.get("name") or ""
                            arguments = item.get("arguments") or {}
                            extra_content = item.get("extra_content")
                            if extra_content is None:
                                extra_content = item.get("extraContent")
                            if extra_content is None:
                                thought_sig = item.get("thought_signature") or item.get("thoughtSignature")
                                if thought_sig:
                                    extra_content = {"google": {"thought_signature": thought_sig}}
                            try:
                                arg_str = json.dumps(arguments)
                            except Exception:
                                arg_str = "{}"
                            tool_call: Dict[str, Any] = {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arg_str},
                            }
                            if extra_content is not None:
                                tool_call["extra_content"] = extra_content
                            tool_calls.append(tool_call)
                        elif isinstance(item, dict) and "text" in item:
                            text_parts.append(str(item.get("text") or ""))
                        else:
                            text_parts.append(str(item))
                else:
                    text_parts.append(_message_to_text(msg))
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": "".join(text_parts),
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)
                continue
        text = _message_to_text(msg)
        messages.append({"role": role, "content": text})
    return messages


def _call_llm(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]) -> Any:
    provider = model.get("provider") or model.get("api")
    if not provider:
        raise ValueError("Model provider is required")
    client = get_client(provider)
    if client is None:
        raise ValueError(f"Unknown provider: {provider}")
    model_id = model.get("id") or get_default_model(provider)
    messages = _context_to_messages(context, provider)
    call_kwargs = dict(options or {})
    if "tools" not in call_kwargs and context.get("tools") is not None:
        call_kwargs["tools"] = context.get("tools")
    response = client.chat_completion(messages=messages, model=model_id, **call_kwargs)
    return response


def _extract_text_and_tool_calls(response: Any) -> Tuple[str, List[Dict[str, Any]]]:
    text = ""
    tool_calls: List[Dict[str, Any]] = []
    try:
        msg = response.choices[0].message
    except Exception:
        return str(response), tool_calls

    content = getattr(msg, "content", "")
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        text = str(content)

    def _extract_extra_content(call: Any) -> Optional[Dict[str, Any]]:
        if isinstance(call, dict):
            extra = call.get("extra_content") or call.get("extraContent")
            if extra is None:
                thought_sig = call.get("thought_signature") or call.get("thoughtSignature")
                if thought_sig:
                    extra = {"google": {"thought_signature": thought_sig}}
            return extra
        extra = getattr(call, "extra_content", None) or getattr(call, "extraContent", None)
        if extra is None:
            thought_sig = getattr(call, "thought_signature", None) or getattr(call, "thoughtSignature", None)
            if thought_sig:
                extra = {"google": {"thought_signature": thought_sig}}
        if extra is None:
            model_extra = getattr(call, "model_extra", None)
            if isinstance(model_extra, dict):
                extra = model_extra.get("extra_content") or model_extra.get("extraContent")
                if extra is None:
                    thought_sig = model_extra.get("thought_signature") or model_extra.get("thoughtSignature")
                    if thought_sig:
                        extra = {"google": {"thought_signature": thought_sig}}
        if extra is None and hasattr(call, "model_dump"):
            try:
                data = call.model_dump()
            except Exception:
                data = None
            if isinstance(data, dict):
                extra = data.get("extra_content") or data.get("extraContent")
                if extra is None:
                    thought_sig = data.get("thought_signature") or data.get("thoughtSignature")
                    if thought_sig:
                        extra = {"google": {"thought_signature": thought_sig}}
        return extra

    raw_calls = getattr(msg, "tool_calls", None)
    if raw_calls:
        for idx, call in enumerate(raw_calls):
            if isinstance(call, dict):
                call_id = call.get("id") or f"call_{idx}"
                func = call.get("function") or {}
                name = func.get("name") or call.get("name") or ""
                args_raw = func.get("arguments") or call.get("arguments") or ""
            else:
                call_id = getattr(call, "id", None) or f"call_{idx}"
                func = getattr(call, "function", None)
                if func is not None:
                    name = getattr(func, "name", "") or ""
                    args_raw = getattr(func, "arguments", "") or ""
                else:
                    name = getattr(call, "name", "") or ""
                    args_raw = getattr(call, "arguments", "") or ""
            if isinstance(args_raw, dict):
                args = args_raw
            else:
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except Exception:
                    args = {"_raw": args_raw}
            tool_call: Dict[str, Any] = {"type": "toolCall", "id": call_id, "name": name, "arguments": args}
            extra_content = _extract_extra_content(call)
            if extra_content is not None:
                tool_call["extra_content"] = extra_content
            tool_calls.append(tool_call)
    return text, tool_calls


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
            response = await asyncio.to_thread(_call_llm, self._model, self._context, self._options)
        except Exception as exc:
            error_msg = dict(partial)
            error_msg["stopReason"] = "error"
            error_msg["errorMessage"] = str(exc)
            self._final_message = error_msg
            yield {"type": "error", "error": error_msg}
            return

        text, tool_calls = _extract_text_and_tool_calls(response)
        if text:
            partial["content"][0]["text"] = text
            yield {"type": "text_delta", "contentIndex": 0, "delta": text, "partial": dict(partial)}
            yield {"type": "text_end", "contentIndex": 0, "content": text, "partial": dict(partial)}
        final = dict(partial)
        content_items: List[Dict[str, Any]] = []
        if text:
            content_items.append({"type": "text", "text": text})
        content_items.extend(tool_calls)
        final["content"] = content_items
        final["stopReason"] = "stop"
        self._final_message = final
        yield {"type": "done", "message": final}

    async def result(self) -> Dict[str, Any]:
        return self._final_message or {}


async def stream(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]) -> MessageEventStream:
    """Stream an LLM response using the StreamFn signature."""
    return MessageEventStream(model, context, options)


__all__ = ["stream", "MessageEventStream"]
