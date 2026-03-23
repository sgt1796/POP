"""Streaming support for LLM responses.

This module exposes a `stream` function compatible with the agent
loop's `StreamFn` type. It accepts `(model, context, options)` and
returns an async iterable of events with `type` keys, plus a
`result()` method for the final assistant message.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from .api_registry import get_client, get_default_model
from .usage_tracking import build_usage_record


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
    openai_like = {"openai", "deepseek", "doubao", "gemini", "claude"}
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


def _prepare_call(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    provider = model.get("provider") or model.get("api")
    if not provider:
        raise ValueError("Model provider is required")
    client = get_client(provider)
    if client is None:
        raise ValueError(f"Unknown provider: {provider}")
    model_id = model.get("id") or get_default_model(provider)
    messages = _context_to_messages(context, provider)
    call_options = dict(options or {})
    if "tools" not in call_options and context.get("tools") is not None:
        call_options["tools"] = context.get("tools")
    return {
        "provider": provider,
        "client": client,
        "model_id": model_id,
        "messages": messages,
        "call_options": call_options,
        "tools": call_options.get("tools"),
        "response_format": call_options.get("response_format"),
    }


def _invoke_prepared_call(prepared_call: Dict[str, Any]) -> Any:
    return prepared_call["client"].chat_completion(
        messages=prepared_call["messages"],
        model=prepared_call["model_id"],
        **prepared_call["call_options"],
    )


def _call_llm(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]) -> Any:
    return _invoke_prepared_call(_prepare_call(model, context, options))


def _get_effective_usage_inputs(
    client: Any,
    messages: List[Dict[str, Any]],
    tools: Any,
    response_format: Any,
) -> Tuple[List[Dict[str, Any]], Any, Any]:
    request_meta = getattr(client, "last_request_meta", None)
    if not isinstance(request_meta, dict):
        return messages, tools, response_format
    usage_messages = request_meta.get("messages", messages)
    usage_tools = request_meta["tools"] if "tools" in request_meta else tools
    usage_response_format = request_meta["response_format"] if "response_format" in request_meta else response_format
    return usage_messages, usage_tools, usage_response_format


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


def _detect_empty_reply_error(
    provider: str,
    text: str,
    tool_calls: List[Dict[str, Any]],
    requested_tools: Any,
) -> Optional[str]:
    if (provider or "").lower() != "gemini":
        return None
    if not requested_tools:
        return None
    if text.strip() or tool_calls:
        return None
    return (
        "Gemini returned an empty response without text or tool calls for a tool-enabled request. "
        "POP is treating this as an error so Gemini compatibility issues do not fail silently."
    )


def _choice_delta(choice: Any) -> Any:
    if isinstance(choice, dict):
        return choice.get("delta")
    return getattr(choice, "delta", None)


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _extract_chunk_text(chunk: Any) -> str:
    pieces: List[str] = []
    for choice in _value(chunk, "choices", []) or []:
        delta = _choice_delta(choice)
        content = _value(delta, "content", None)
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    pieces.append(str(item.get("text") or ""))
                elif hasattr(item, "text"):
                    pieces.append(str(getattr(item, "text") or ""))
    return "".join(pieces)


def _merge_tool_call_delta(tool_state: Dict[int, Dict[str, Any]], chunk: Any) -> None:
    for choice in _value(chunk, "choices", []) or []:
        delta = _choice_delta(choice)
        raw_calls = _value(delta, "tool_calls", None) or []
        for fallback_index, raw_call in enumerate(raw_calls):
            index = _value(raw_call, "index", fallback_index)
            entry = tool_state.setdefault(
                int(index),
                {"id": None, "name": "", "arguments_text": "", "extra_content": None},
            )
            call_id = _value(raw_call, "id", None)
            if call_id:
                entry["id"] = call_id
            function = _value(raw_call, "function", None)
            name = _value(function, "name", "") or _value(raw_call, "name", "")
            if name:
                entry["name"] = name
            arguments = _value(function, "arguments", None)
            if arguments is None:
                arguments = _value(raw_call, "arguments", None)
            if isinstance(arguments, dict):
                entry["arguments_text"] += json.dumps(arguments, ensure_ascii=False, sort_keys=True)
            elif arguments:
                entry["arguments_text"] += str(arguments)
            extra_content = _extract_extra_content(raw_call)
            if extra_content is not None:
                entry["extra_content"] = extra_content


def _finalize_stream_tool_calls(tool_state: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    final_calls: List[Dict[str, Any]] = []
    for index in sorted(tool_state):
        state = tool_state[index]
        args_raw = state.get("arguments_text", "")
        if args_raw:
            try:
                arguments = json.loads(args_raw)
            except Exception:
                arguments = {"_raw": args_raw}
        else:
            arguments = {}
        tool_call: Dict[str, Any] = {
            "type": "toolCall",
            "id": state.get("id") or f"call_{index}",
            "name": state.get("name") or "",
            "arguments": arguments,
        }
        if state.get("extra_content") is not None:
            tool_call["extra_content"] = state["extra_content"]
        final_calls.append(tool_call)
    return final_calls


def _extract_chunk_usage(chunk: Any) -> Any:
    return _value(chunk, "usage", None)


def _should_stream_natively(provider: str, options: Dict[str, Any]) -> bool:
    return (provider or "").lower() == "claude" and options.get("stream", True) is not False


def _build_native_stream_options(call_options: Dict[str, Any]) -> Dict[str, Any]:
    stream_call_options = dict(call_options)
    stream_call_options["stream"] = True
    stream_options = stream_call_options.get("stream_options")
    if isinstance(stream_options, dict):
        merged_stream_options = dict(stream_options)
    else:
        merged_stream_options = {}
    merged_stream_options.setdefault("include_usage", True)
    stream_call_options["stream_options"] = merged_stream_options
    return stream_call_options


async def _iterate_sync_stream(prepared_call: Dict[str, Any], stream_options: Dict[str, Any]) -> AsyncIterator[Any]:
    queue: "asyncio.Queue[Tuple[str, Any]]" = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker() -> None:
        stream_obj: Any = None
        try:
            stream_obj = prepared_call["client"].chat_completion(
                messages=prepared_call["messages"],
                model=prepared_call["model_id"],
                **stream_options,
            )
            for chunk in stream_obj:
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            if stream_obj is not None:
                close = getattr(stream_obj, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        kind, payload = await queue.get()
        if kind == "chunk":
            yield payload
        elif kind == "error":
            raise payload
        else:
            break


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

    def _default_usage_inputs(self) -> Tuple[str, str, List[Dict[str, Any]], Any, Any]:
        provider = self._model.get("provider") or self._model.get("api") or ""
        model_id = self._model.get("id") or get_default_model(provider) or ""
        messages = _context_to_messages(self._context, provider)
        tools = self._options.get("tools") if isinstance(self._options, dict) else None
        if tools is None:
            tools = self._context.get("tools")
        response_format = self._options.get("response_format") if isinstance(self._options, dict) else None
        return provider, model_id, messages, tools, response_format

    def _usage_inputs(
        self,
        prepared_call: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, List[Dict[str, Any]], Any, Any]:
        if prepared_call is None:
            return self._default_usage_inputs()
        usage_messages, usage_tools, usage_response_format = _get_effective_usage_inputs(
            prepared_call["client"],
            prepared_call["messages"],
            prepared_call["tools"],
            prepared_call["response_format"],
        )
        return (
            prepared_call["provider"],
            prepared_call["model_id"] or "",
            usage_messages,
            usage_tools,
            usage_response_format,
        )

    def _serialize_reply_for_usage(self, text: str, tool_calls: List[Dict[str, Any]]) -> str:
        reply_for_usage = text
        if tool_calls:
            serialized_calls = json.dumps(
                tool_calls,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            reply_for_usage = f"{text}\n{serialized_calls}" if text else serialized_calls
        return reply_for_usage

    def _build_usage_record(
        self,
        response: Any,
        reply_text: str,
        request_start: float,
        prepared_call: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        provider, model_id, messages, tools, response_format = self._usage_inputs(prepared_call)
        return build_usage_record(
            response=response,
            messages=messages,
            reply_text=reply_text,
            provider=provider,
            model=model_id,
            tools=tools,
            response_format=response_format,
            latency_ms=int((time.time() - request_start) * 1000),
            timestamp=time.time(),
        )

    async def _emit_native_stream(
        self,
        prepared_call: Dict[str, Any],
        partial: Dict[str, Any],
        request_start: float,
    ) -> AsyncIterator[Dict[str, Any]]:
        text_parts: List[str] = []
        tool_state: Dict[int, Dict[str, Any]] = {}
        provider_usage: Any = None
        stream_options = _build_native_stream_options(prepared_call["call_options"])

        try:
            async for chunk in _iterate_sync_stream(prepared_call, stream_options):
                delta_text = _extract_chunk_text(chunk)
                if delta_text:
                    text_parts.append(delta_text)
                    partial["content"][0]["text"] = "".join(text_parts)
                    yield {
                        "type": "text_delta",
                        "contentIndex": 0,
                        "delta": delta_text,
                        "partial": dict(partial),
                    }
                _merge_tool_call_delta(tool_state, chunk)
                chunk_usage = _extract_chunk_usage(chunk)
                if chunk_usage is not None:
                    provider_usage = chunk_usage
        except Exception as exc:
            text = "".join(text_parts)
            tool_calls = _finalize_stream_tool_calls(tool_state)
            error_msg = dict(partial)
            error_msg["stopReason"] = "error"
            error_msg["errorMessage"] = str(exc)
            error_msg["content"] = ([{"type": "text", "text": text}] if text else []) + tool_calls
            usage_response = {"usage": provider_usage} if provider_usage is not None else None
            error_msg["usage"] = self._build_usage_record(
                response=usage_response,
                reply_text=self._serialize_reply_for_usage(text, tool_calls),
                request_start=request_start,
                prepared_call=prepared_call,
            )
            self._final_message = error_msg
            yield {"type": "error", "error": error_msg}
            return

        text = "".join(text_parts)
        if text:
            yield {"type": "text_end", "contentIndex": 0, "content": text, "partial": dict(partial)}
        final = dict(partial)
        tool_calls = _finalize_stream_tool_calls(tool_state)
        content_items: List[Dict[str, Any]] = []
        if text:
            content_items.append({"type": "text", "text": text})
        content_items.extend(tool_calls)
        final["content"] = content_items
        final["stopReason"] = "stop"
        usage_response = {"usage": provider_usage} if provider_usage is not None else None
        empty_reply_error = _detect_empty_reply_error(
            prepared_call["provider"],
            text,
            tool_calls,
            prepared_call.get("tools"),
        )
        if empty_reply_error is not None:
            final["stopReason"] = "error"
            final["errorMessage"] = empty_reply_error
            final["usage"] = self._build_usage_record(
                response=usage_response,
                reply_text=self._serialize_reply_for_usage(text, tool_calls),
                request_start=request_start,
                prepared_call=prepared_call,
            )
            self._final_message = final
            yield {"type": "error", "error": final}
            return
        final["usage"] = self._build_usage_record(
            response=usage_response,
            reply_text=self._serialize_reply_for_usage(text, tool_calls),
            request_start=request_start,
            prepared_call=prepared_call,
        )
        self._final_message = final
        yield {"type": "done", "message": final}

    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        if self._started:
            return
        self._started = True
        request_start = time.time()
        partial = self._build_partial()
        yield {"type": "start", "partial": dict(partial)}
        yield {"type": "text_start", "contentIndex": 0, "partial": dict(partial)}
        prepared_call: Optional[Dict[str, Any]] = None
        try:
            prepared_call = _prepare_call(self._model, self._context, self._options)
            if _should_stream_natively(prepared_call["provider"], self._options):
                async for event in self._emit_native_stream(prepared_call, partial, request_start):
                    yield event
                return
            response = await asyncio.to_thread(_invoke_prepared_call, prepared_call)
        except Exception as exc:
            error_msg = dict(partial)
            error_msg["stopReason"] = "error"
            error_msg["errorMessage"] = str(exc)
            error_msg["usage"] = self._build_usage_record(
                response=None,
                reply_text="",
                request_start=request_start,
                prepared_call=prepared_call,
            )
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
        empty_reply_error = _detect_empty_reply_error(
            prepared_call["provider"],
            text,
            tool_calls,
            prepared_call.get("tools"),
        )
        if empty_reply_error is not None:
            error_msg = dict(partial)
            error_msg["content"] = content_items
            error_msg["stopReason"] = "error"
            error_msg["errorMessage"] = empty_reply_error
            error_msg["usage"] = self._build_usage_record(
                response=response,
                reply_text=self._serialize_reply_for_usage(text, tool_calls),
                request_start=request_start,
                prepared_call=prepared_call,
            )
            self._final_message = error_msg
            yield {"type": "error", "error": error_msg}
            return
        final["content"] = content_items
        final["stopReason"] = "stop"
        final["usage"] = self._build_usage_record(
            response=response,
            reply_text=self._serialize_reply_for_usage(text, tool_calls),
            request_start=request_start,
            prepared_call=prepared_call,
        )
        self._final_message = final
        yield {"type": "done", "message": final}

    async def result(self) -> Dict[str, Any]:
        return self._final_message or {}


async def stream(model: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]) -> MessageEventStream:
    """Stream an LLM response using the StreamFn signature."""
    return MessageEventStream(model, context, options)


__all__ = ["stream", "MessageEventStream"]
