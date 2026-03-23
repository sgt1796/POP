from __future__ import annotations

import asyncio
from types import SimpleNamespace

import POP.stream as stream_mod


def _fake_response(text: str, usage: object | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])
    if usage is not None:
        response.usage = usage
    return response


class _FakeClient:
    def __init__(self, response: object, request_meta: dict | None = None) -> None:
        self.response = response
        self.request_meta = request_meta or {}
        self.last_request_meta = request_meta
        self.last_call = None

    def chat_completion(self, messages, model, **kwargs):
        self.last_call = {"messages": messages, "model": model, **kwargs}
        return self.response


class _FakeSyncStream:
    def __init__(self, chunks: list[object], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.closed = False

    def __iter__(self):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.closed = True


def _fake_text_chunk(text: str, usage: object | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(delta=delta)
    chunk = SimpleNamespace(choices=[choice])
    if usage is not None:
        chunk.usage = usage
    return chunk


def _fake_tool_chunk(
    *,
    name: str | None = None,
    arguments: str = "",
    index: int = 0,
    call_id: str | None = None,
    usage: object | None = None,
) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments)
    tool_call = SimpleNamespace(index=index, id=call_id, function=function)
    delta = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(delta=delta)
    chunk = SimpleNamespace(choices=[choice])
    if usage is not None:
        chunk.usage = usage
    return chunk


def _prepared_call(
    *,
    provider: str,
    client: object,
    model_id: str,
    messages: list[dict] | None = None,
    tools: object = None,
    response_format: object = None,
    call_options: dict | None = None,
) -> dict:
    return {
        "provider": provider,
        "client": client,
        "model_id": model_id,
        "messages": messages or [{"role": "user", "content": "Say hi"}],
        "call_options": call_options or {},
        "tools": tools,
        "response_format": response_format,
    }


def test_stream_done_event_includes_provider_usage(monkeypatch):
    prepared = _prepared_call(
        provider="openai",
        client=_FakeClient(
            _fake_response(
                "hello",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        ),
        model_id="gpt-4o",
    )
    monkeypatch.setattr(stream_mod, "_prepare_call", lambda model, context, options: prepared)

    async def run_stream():
        event_stream = await stream_mod.stream(
            {"provider": "openai", "id": "gpt-4o"},
            {"messages": [{"role": "user", "content": "Say hi"}]},
            {},
        )
        events = []
        async for event in event_stream:
            events.append(event)
        return events

    events = asyncio.run(run_stream())
    done_event = next(event for event in events if event["type"] == "done")
    usage = done_event["message"]["usage"]

    assert usage["source"] == "provider"
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 5
    assert usage["total_tokens"] == 15


def test_stream_done_event_includes_estimated_usage_when_missing(monkeypatch):
    prepared = _prepared_call(
        provider="openai",
        client=_FakeClient(_fake_response("hello without provider usage")),
        model_id="gpt-4o",
    )
    monkeypatch.setattr(stream_mod, "_prepare_call", lambda model, context, options: prepared)

    async def run_stream():
        event_stream = await stream_mod.stream(
            {"provider": "openai", "id": "gpt-4o"},
            {"messages": [{"role": "user", "content": "Say hi"}]},
            {},
        )
        events = []
        async for event in event_stream:
            events.append(event)
        return events

    events = asyncio.run(run_stream())
    done_event = next(event for event in events if event["type"] == "done")
    usage = done_event["message"]["usage"]

    assert usage["source"] == "estimate"
    assert usage["total_tokens"] is not None
    assert usage["estimate_total_tokens"] == usage["total_tokens"]


def test_gemini_empty_tool_response_emits_error_instead_of_done(monkeypatch):
    prepared = _prepared_call(
        provider="gemini",
        client=_FakeClient(
            _fake_response(
                "",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=0, total_tokens=10),
            )
        ),
        model_id="gemini-2.5-pro",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(stream_mod, "_prepare_call", lambda model, context, options: prepared)

    async def run_stream():
        event_stream = await stream_mod.stream(
            {"provider": "gemini", "id": "gemini-2.5-pro"},
            {"messages": [{"role": "user", "content": "Call ping"}]},
            {},
        )
        events = []
        async for event in event_stream:
            events.append(event)
        return events

    events = asyncio.run(run_stream())
    assert not any(event["type"] == "done" for event in events)
    error_event = next(event for event in events if event["type"] == "error")
    usage = error_event["error"]["usage"]

    assert "empty response" in error_event["error"]["errorMessage"]
    assert usage["source"] == "provider"
    assert usage["output_tokens"] == 0
    assert usage["total_tokens"] == 10


def test_claude_stream_emits_incremental_text_and_provider_usage(monkeypatch):
    stream_obj = _FakeSyncStream(
        [
            _fake_text_chunk("hel"),
            _fake_text_chunk("lo"),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]
    )
    client = _FakeClient(stream_obj, request_meta={"messages": [{"role": "user", "content": "Say hi"}], "tools": None, "response_format": None})
    prepared = _prepared_call(provider="claude", client=client, model_id="claude-haiku-4-5")
    monkeypatch.setattr(stream_mod, "_prepare_call", lambda model, context, options: prepared)

    async def run_stream():
        event_stream = await stream_mod.stream(
            {"provider": "claude", "id": "claude-haiku-4-5"},
            {"messages": [{"role": "user", "content": "Say hi"}]},
            {},
        )
        events = []
        async for event in event_stream:
            events.append(event)
        return events

    events = asyncio.run(run_stream())
    text_deltas = [event["delta"] for event in events if event["type"] == "text_delta"]
    done_event = next(event for event in events if event["type"] == "done")
    usage = done_event["message"]["usage"]

    assert text_deltas == ["hel", "lo"]
    assert done_event["message"]["content"][0] == {"type": "text", "text": "hello"}
    assert usage["source"] == "provider"
    assert usage["total_tokens"] == 15
    assert client.last_call is not None
    assert client.last_call["stream"] is True
    assert client.last_call["stream_options"]["include_usage"] is True


def test_claude_stream_assembles_tool_calls(monkeypatch):
    stream_obj = _FakeSyncStream(
        [
            _fake_tool_chunk(name="read_file", arguments='{"path"', call_id="call_1"),
            _fake_tool_chunk(arguments=':"/tmp/example.txt"}', call_id="call_1"),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=8, completion_tokens=4, total_tokens=12),
            ),
        ]
    )
    client = _FakeClient(stream_obj, request_meta={"messages": [{"role": "user", "content": "Read a file"}], "tools": None, "response_format": None})
    prepared = _prepared_call(provider="claude", client=client, model_id="claude-haiku-4-5")
    monkeypatch.setattr(stream_mod, "_prepare_call", lambda model, context, options: prepared)

    async def run_stream():
        event_stream = await stream_mod.stream(
            {"provider": "claude", "id": "claude-haiku-4-5"},
            {"messages": [{"role": "user", "content": "Read a file"}]},
            {},
        )
        events = []
        async for event in event_stream:
            events.append(event)
        return events

    events = asyncio.run(run_stream())
    done_event = next(event for event in events if event["type"] == "done")
    tool_call = next(item for item in done_event["message"]["content"] if item["type"] == "toolCall")

    assert tool_call["id"] == "call_1"
    assert tool_call["name"] == "read_file"
    assert tool_call["arguments"] == {"path": "/tmp/example.txt"}


def test_claude_stream_error_event_includes_usage(monkeypatch):
    stream_obj = _FakeSyncStream([_fake_text_chunk("partial")], error=RuntimeError("boom"))
    client = _FakeClient(stream_obj, request_meta={"messages": [{"role": "user", "content": "Say hi"}], "tools": None, "response_format": None})
    prepared = _prepared_call(provider="claude", client=client, model_id="claude-haiku-4-5")
    monkeypatch.setattr(stream_mod, "_prepare_call", lambda model, context, options: prepared)

    async def run_stream():
        event_stream = await stream_mod.stream(
            {"provider": "claude", "id": "claude-haiku-4-5"},
            {"messages": [{"role": "user", "content": "Say hi"}]},
            {},
        )
        events = []
        async for event in event_stream:
            events.append(event)
        return events

    events = asyncio.run(run_stream())
    error_event = next(event for event in events if event["type"] == "error")
    usage = error_event["error"]["usage"]

    assert usage["total_tokens"] is not None
    assert error_event["error"]["content"][0] == {"type": "text", "text": "partial"}
    assert error_event["error"]["errorMessage"] == "boom"
