import json

import pytest

from POP.prompt_function import PromptFunction
from POP.stream import stream as stream_response


@pytest.mark.integration
def test_live_claude_basic(live_enabled):
    if not live_enabled("claude"):
        pytest.skip("Live Claude tests disabled or ANTHROPIC_API_KEY missing.")
    pf = PromptFunction(prompt="Say hello in one short sentence.", client="claude")
    result = pf.execute()
    assert isinstance(result, str)
    assert result.strip() != ""


@pytest.mark.integration
def test_live_claude_tool_call_optional(live_enabled, tools):
    if not live_enabled("claude"):
        pytest.skip("Live Claude tests disabled or ANTHROPIC_API_KEY missing.")
    pf = PromptFunction(prompt="Read a txt file at /tmp/example.txt", client="claude")
    result = pf.execute(tools=tools)
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        pytest.skip("Model did not return tool-call arguments.")
    assert "path" in data


@pytest.mark.integration
def test_live_claude_stream_basic(live_enabled):
    if not live_enabled("claude"):
        pytest.skip("Live Claude tests disabled or ANTHROPIC_API_KEY missing.")

    async def run_stream():
        event_stream = await stream_response(
            {"provider": "claude", "id": "claude-haiku-4-5"},
            {"messages": [{"role": "user", "content": "Say hello in one short sentence."}]},
            {},
        )
        events = []
        async for event in event_stream:
            events.append(event)
        return events

    import asyncio

    events = asyncio.run(run_stream())
    done_event = next(event for event in events if event["type"] == "done")
    text_items = [item for item in done_event["message"]["content"] if item["type"] == "text"]
    assert text_items
    assert text_items[0]["text"].strip() != ""
