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


def test_stream_done_event_includes_provider_usage(monkeypatch):
    def fake_call_llm(model, context, options):
        del model, context, options
        return _fake_response(
            "hello",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    monkeypatch.setattr(stream_mod, "_call_llm", fake_call_llm)

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
    def fake_call_llm(model, context, options):
        del model, context, options
        return _fake_response("hello without provider usage")

    monkeypatch.setattr(stream_mod, "_call_llm", fake_call_llm)

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
