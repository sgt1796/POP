from __future__ import annotations

from types import SimpleNamespace

from POP.prompt_function import PromptFunction
from POP.providers.gemini_client import GeminiClient


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.payloads = []

    def create(self, **payload):
        self.payloads.append(payload)
        message = SimpleNamespace(content="ok", tool_calls=None)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeOpenAI:
    last_instance = None

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())
        _FakeOpenAI.last_instance = self


def _last_payload() -> dict:
    instance = _FakeOpenAI.last_instance
    assert instance is not None
    return instance.chat.completions.payloads[-1]


def test_prompt_function_omits_empty_tools_for_gemini(monkeypatch):
    monkeypatch.setattr("POP.providers.gemini_client.OpenAI", _FakeOpenAI)
    client = GeminiClient(model="gemini-2.5-pro")
    pf = PromptFunction(prompt="Hello <<<name>>>", client=client)

    result = pf.execute(name="World", tools=[])

    assert result == "ok"
    payload = _last_payload()
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_gemini_chat_completion_omits_tool_config_without_declarations(monkeypatch):
    monkeypatch.setattr("POP.providers.gemini_client.OpenAI", _FakeOpenAI)
    client = GeminiClient(model="gemini-2.5-pro")

    client.chat_completion(
        messages=[{"role": "user", "content": "Say hello."}],
        tools=[],
        tool_choice="required",
    )

    payload = _last_payload()
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_gemini_chat_completion_preserves_real_tools(monkeypatch):
    monkeypatch.setattr("POP.providers.gemini_client.OpenAI", _FakeOpenAI)
    client = GeminiClient(model="gemini-2.5-pro")
    tool = {
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Return pong.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    client.chat_completion(
        messages=[{"role": "user", "content": "Call ping."}],
        tools=[tool],
        tool_choice="auto",
    )

    payload = _last_payload()
    assert payload["tools"] == [tool]
    assert payload["tool_choice"] == "auto"
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "ping"
