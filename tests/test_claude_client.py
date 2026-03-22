from __future__ import annotations

from types import SimpleNamespace

import pytest

from POP.providers.claude_client import CLAUDE_OPENAI_BASE_URL, ClaudeClient


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


def test_claude_client_configures_openai_compat_endpoint(monkeypatch):
    monkeypatch.setattr("POP.providers.claude_client.OpenAI", _FakeOpenAI)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    ClaudeClient()

    instance = _FakeOpenAI.last_instance
    assert instance is not None
    assert instance.kwargs["api_key"] == "test-key"
    assert instance.kwargs["base_url"] == CLAUDE_OPENAI_BASE_URL


def test_claude_chat_completion_hoists_system_and_developer_messages(monkeypatch):
    monkeypatch.setattr("POP.providers.claude_client.OpenAI", _FakeOpenAI)
    client = ClaudeClient(model="claude-haiku-4-5")

    client.chat_completion(
        messages=[
            {"role": "system", "content": "System A"},
            {"role": "developer", "content": "Developer B"},
            {"role": "user", "content": "Hello"},
        ],
    )

    payload = _last_payload()
    assert payload["messages"][0] == {"role": "system", "content": "System A\nDeveloper B"}
    assert payload["messages"][1] == {"role": "user", "content": "Hello"}


def test_claude_chat_completion_warns_and_degrades_response_format_and_strict(monkeypatch):
    monkeypatch.setattr("POP.providers.claude_client.OpenAI", _FakeOpenAI)
    client = ClaudeClient(model="claude-haiku-4-5")
    tool = {
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Return pong.",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        },
    }

    with pytest.warns(UserWarning) as recorded:
        client.chat_completion(
            messages=[{"role": "user", "content": "Call ping."}],
            tools=[tool],
            response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
        )

    warning_text = " ".join(str(item.message) for item in recorded)
    assert "response_format" in warning_text
    assert "strict" in warning_text
    payload = _last_payload()
    assert "response_format" not in payload
    assert payload["tools"][0]["function"]["name"] == "ping"
    assert "strict" not in payload["tools"][0]["function"]


def test_claude_chat_completion_preserves_tools_and_tool_choice(monkeypatch):
    monkeypatch.setattr("POP.providers.claude_client.OpenAI", _FakeOpenAI)
    client = ClaudeClient(model="claude-haiku-4-5")
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup something.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    client.chat_completion(
        messages=[{"role": "user", "content": "Use the tool."}],
        tools=[tool],
        tool_choice="auto",
    )

    payload = _last_payload()
    assert payload["tools"] == [tool]
    assert payload["tool_choice"] == "auto"


def test_claude_chat_completion_clamps_temperature_and_normalizes_n(monkeypatch):
    monkeypatch.setattr("POP.providers.claude_client.OpenAI", _FakeOpenAI)
    client = ClaudeClient(model="claude-haiku-4-5")

    with pytest.warns(UserWarning) as recorded:
        client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=2.5,
            n=3,
        )

    warning_text = " ".join(str(item.message) for item in recorded)
    assert "temperature" in warning_text
    assert "`n=1`" in warning_text
    payload = _last_payload()
    assert payload["temperature"] == 1.0
    assert payload["n"] == 1
