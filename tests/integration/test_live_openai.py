import json
import pytest

from POP.prompt_function import PromptFunction


@pytest.mark.integration
def test_live_openai_basic(live_enabled):
    if not live_enabled("openai"):
        pytest.skip("Live OpenAI tests disabled or OPENAI_API_KEY missing.")
    pf = PromptFunction(prompt="Say hello in one short sentence.", client="openai")
    result = pf.execute()
    assert isinstance(result, str)
    assert result.strip() != ""


@pytest.mark.integration
def test_live_openai_tool_call_optional(live_enabled, tools):
    if not live_enabled("openai"):
        pytest.skip("Live OpenAI tests disabled or OPENAI_API_KEY missing.")
    pf = PromptFunction(prompt="Read a txt file at /tmp/example.txt", client="openai")
    result = pf.execute(tools=tools)
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        pytest.skip("Model did not return tool-call arguments.")
    assert "path" in data
