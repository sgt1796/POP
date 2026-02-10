"""Proxy transport for routing LLM calls through an intermediary server.

The functions in this module implement a simple proxy protocol
compatible with the TypeScript `streamProxy` in the original
`pi‑agent` implementation.  When calling the LLM through a proxy
server the client does not send authentication headers directly to
the provider; instead it authenticates with the proxy which in
turn forwards requests to the underlying provider.  The proxy
strips the `partial` field from delta events to reduce bandwidth
and reconstructs messages on the client side.

The main entry point is :func:`stream_proxy`.  It accepts a model
definition, a context dictionary and a :class:`ProxyStreamOptions`
instance.  It returns an asynchronous event stream yielding
assistant message events as dictionaries.  The returned object
implements ``__aiter__`` and ``result()`` so that it behaves like
the transport objects returned by ``POP.stream_simple``.

Example usage::

    from pop_agent.proxy import stream_proxy, ProxyStreamOptions
    stream = stream_proxy(model, context, ProxyStreamOptions(
        auth_token="…", proxy_url="https://myproxy.example.com"
    ))
    async for event in stream:
        ...
    final_message = await stream.result()

"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .event_stream import EventStream


###############################################################################
# Data classes
###############################################################################

@dataclass
class ProxyStreamOptions:
    """Options for the proxy transport.

    Parameters
    ----------
    auth_token : str
        Bearer token used to authenticate with the proxy.  This
        token should be obtained via your own authentication flow.
    proxy_url : str
        Base URL of the proxy server (e.g. ``"https://genai.example.com"``).
    temperature : float, optional
        Sampling temperature forwarded to the LLM provider.  Omit to
        use the provider default.
    max_tokens : int, optional
        Maximum number of tokens to generate.  Omit to let the
        provider choose.
    reasoning : str, optional
        Reasoning level requested from the provider.  See the
        documentation of your provider for supported values.
    signal : asyncio.Event, optional
        Cancellation signal that aborts the HTTP request when set.
    """

    auth_token: str
    proxy_url: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning: Optional[str] = None
    signal: Optional[asyncio.Event] = None


###############################################################################
# Proxy event stream implementation
###############################################################################

class ProxyMessageEventStream:
    """Asynchronous event stream for proxy messages.

    This class wraps an underlying :class:`EventStream` but exposes
    ``__aiter__`` and ``result()`` methods so that it behaves like
    the objects returned by ``POP.stream_simple``.  Events
    pushed into this stream should be dictionaries conforming to
    the assistant message event protocol described in the original
    agent implementation.
    """

    def __init__(self) -> None:
        # Underlying event stream closes when it receives a done/error event
        self._stream: EventStream[Dict[str, Any], Dict[str, Any]] = EventStream(
            done_predicate=lambda e: e.get("type") in {"done", "error"},
            result_selector=lambda e: e.get("message") or e.get("error"),
        )

    def push(self, event: Dict[str, Any]) -> None:
        self._stream.push(event)

    def end(self) -> None:
        self._stream.end()

    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        async for event in self._stream:
            yield event

    async def result(self) -> Dict[str, Any]:
        return (await self._stream.result()) or {}


###############################################################################
# Main proxy function
###############################################################################

async def stream_proxy(model: Dict[str, Any], context: Dict[str, Any], options: ProxyStreamOptions) -> ProxyMessageEventStream:
    """Call the LLM through a proxy server and stream events.

    Parameters
    ----------
    model : dict
        LLM model definition.  Passed verbatim to the proxy.
    context : dict
        Conversation context including system prompt, messages and tools.
    options : ProxyStreamOptions
        Configuration for the proxy call including authentication and
        sampling parameters.

    Returns
    -------
    ProxyMessageEventStream
        Asynchronous stream of assistant message events.  Use
        ``async for`` to iterate over events and await ``result()`` to
        obtain the final assistant message.
    """
    stream = ProxyMessageEventStream()

    async def _run() -> None:
        # Initialise partial message for reconstruction
        partial: Dict[str, Any] = {
            "role": "assistant",
            "stopReason": "stop",
            "content": [],
            "api": model.get("api"),
            "provider": model.get("provider"),
            "model": model.get("id"),
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "total": 0,
                },
            },
            "timestamp": time.time(),
        }

        def process_proxy_event(proxy_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Process a single proxy event and update the partial message."""
            etype = proxy_event.get("type")
            # Content index may be used to map deltas into the content list
            if etype == "start":
                return {"type": "start", "partial": dict(partial)}
            if etype == "text_start":
                idx = proxy_event.get("contentIndex", 0)
                # Ensure the list is long enough
                while len(partial["content"]) <= idx:
                    partial["content"].append(None)
                partial["content"][idx] = {"type": "text", "text": ""}
                return {"type": "text_start", "contentIndex": idx, "partial": dict(partial)}
            if etype == "text_delta":
                idx = proxy_event.get("contentIndex", 0)
                delta = proxy_event.get("delta", "")
                item = partial["content"][idx]
                if item and item.get("type") == "text":
                    item["text"] = item.get("text", "") + delta
                    return {"type": "text_delta", "contentIndex": idx, "delta": delta, "partial": dict(partial)}
                return None
            if etype == "text_end":
                idx = proxy_event.get("contentIndex", 0)
                content_sig = proxy_event.get("contentSignature")
                item = partial["content"][idx]
                if item and item.get("type") == "text":
                    if content_sig is not None:
                        item["textSignature"] = content_sig
                    return {"type": "text_end", "contentIndex": idx, "content": item.get("text"), "partial": dict(partial)}
                return None
            if etype == "thinking_start":
                idx = proxy_event.get("contentIndex", 0)
                while len(partial["content"]) <= idx:
                    partial["content"].append(None)
                partial["content"][idx] = {"type": "thinking", "thinking": ""}
                return {"type": "thinking_start", "contentIndex": idx, "partial": dict(partial)}
            if etype == "thinking_delta":
                idx = proxy_event.get("contentIndex", 0)
                delta = proxy_event.get("delta", "")
                item = partial["content"][idx]
                if item and item.get("type") == "thinking":
                    item["thinking"] = item.get("thinking", "") + delta
                    return {"type": "thinking_delta", "contentIndex": idx, "delta": delta, "partial": dict(partial)}
                return None
            if etype == "thinking_end":
                idx = proxy_event.get("contentIndex", 0)
                content_sig = proxy_event.get("contentSignature")
                item = partial["content"][idx]
                if item and item.get("type") == "thinking":
                    if content_sig is not None:
                        item["thinkingSignature"] = content_sig
                    return {"type": "thinking_end", "contentIndex": idx, "content": item.get("thinking"), "partial": dict(partial)}
                return None
            if etype == "toolcall_start":
                idx = proxy_event.get("contentIndex", 0)
                while len(partial["content"]) <= idx:
                    partial["content"].append(None)
                # Start with empty arguments and accumulate JSON deltas
                partial["content"][idx] = {
                    "type": "toolCall",
                    "id": proxy_event.get("id"),
                    "name": proxy_event.get("toolName"),
                    "arguments": {},
                    "partialJson": "",
                }
                return {"type": "toolcall_start", "contentIndex": idx, "partial": dict(partial)}
            if etype == "toolcall_delta":
                idx = proxy_event.get("contentIndex", 0)
                delta = proxy_event.get("delta", "")
                item = partial["content"][idx]
                if item and item.get("type") == "toolCall":
                    partial_json = item.get("partialJson", "") + delta
                    item["partialJson"] = partial_json
                    # Try to parse as JSON
                    try:
                        item["arguments"] = json.loads(partial_json)
                    except Exception:
                        # Incomplete JSON; ignore until complete
                        pass
                    return {"type": "toolcall_delta", "contentIndex": idx, "delta": delta, "partial": dict(partial)}
                return None
            if etype == "toolcall_end":
                idx = proxy_event.get("contentIndex", 0)
                item = partial["content"][idx]
                if item and item.get("type") == "toolCall":
                    # Remove partialJson helper
                    item.pop("partialJson", None)
                    return {"type": "toolcall_end", "contentIndex": idx, "toolCall": item, "partial": dict(partial)}
                return None
            if etype == "done":
                # Populate stop reason and usage, return final message
                partial["stopReason"] = proxy_event.get("reason")
                usage = proxy_event.get("usage")
                if usage is not None:
                    partial["usage"] = usage
                return {"type": "done", "reason": proxy_event.get("reason"), "message": dict(partial)}
            if etype == "error":
                partial["stopReason"] = proxy_event.get("reason")
                err_msg = proxy_event.get("errorMessage")
                if err_msg:
                    partial["errorMessage"] = err_msg
                usage = proxy_event.get("usage")
                if usage is not None:
                    partial["usage"] = usage
                return {"type": "error", "reason": proxy_event.get("reason"), "error": dict(partial)}
            # Unknown event types are ignored to avoid breaking the stream
            return None

        # Compose HTTP request body
        body = {
            "model": model,
            "context": context,
            "options": {
                # Map optional parameters to provider expected names
                **({"temperature": options.temperature} if options.temperature is not None else {}),
                **({"maxTokens": options.max_tokens} if options.max_tokens is not None else {}),
                **({"reasoning": options.reasoning} if options.reasoning is not None else {}),
            },
        }
        headers = {
            "Authorization": f"Bearer {options.auth_token}",
            "Content-Type": "application/json",
        }
        # Use httpx for asynchronous streaming
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{options.proxy_url.rstrip('/')}/api/stream",
                    json=body,
                    headers=headers,
                ) as resp:
                    if resp.status_code >= 400:
                        # Attempt to read error message from JSON body
                        error_message = f"Proxy error: {resp.status_code} {resp.reason_phrase}"
                        try:
                            data = await resp.json()
                            if isinstance(data, dict) and "error" in data:
                                error_message = f"Proxy error: {data['error']}"
                        except Exception:
                            pass
                        raise RuntimeError(error_message)
                    # Read streaming lines
                    async for line in resp.aiter_lines():
                        if options.signal and options.signal.is_set():
                            # Abort the request
                            raise asyncio.CancelledError("Request aborted by user")
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            proxy_event = json.loads(data_str)
                        except Exception:
                            continue
                        event = process_proxy_event(proxy_event)
                        if event:
                            stream.push(event)
                    # End of stream
                    stream.end()
            except asyncio.CancelledError:
                # User cancelled: mark partial as aborted
                partial["stopReason"] = "aborted"
                event = {
                    "type": "error",
                    "reason": "aborted",
                    "error": dict(partial),
                }
                stream.push(event)
                stream.end()
            except Exception as exc:
                # Unexpected error: emit as error event
                partial["stopReason"] = "error"
                partial["errorMessage"] = str(exc)
                event = {
                    "type": "error",
                    "reason": "error",
                    "error": dict(partial),
                }
                stream.push(event)
                stream.end()

    # Schedule the proxy call
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        raise RuntimeError(
            "stream_proxy must be called within an asyncio event loop. "
            "Use asyncio.run() or create a loop manually."
        )
    loop.create_task(_run())
    return stream


__all__ = ["stream_proxy", "ProxyStreamOptions"]