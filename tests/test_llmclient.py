# tests/test_llmclient.py
import types
import pytest

import LLMClient
from LLMClient import OpenAIClient, DeepseekClient, GeminiClient, DoubaoClient


def test_openai_client_builds_request_payload(monkeypatch):
    # Stub the underlying OpenAI OpenAI client so no real API calls happen.
    captured = {}

    class DummyChat:
        def completions_create(self, **kwargs):
            captured["payload"] = kwargs
            # Minimal OpenAI-like response structure
            class Msg:
                def __init__(self):
                    self.content = "ok"
                    self.tool_calls = None

            class Choice:
                def __init__(self):
                    self.message = Msg()

            class Resp:
                def __init__(self):
                    self.choices = [Choice()]

            return Resp()

        # For compatibility with .chat.completions.create(...)
        @property
        def completions(self):
            return types.SimpleNamespace(create=self.completions_create)

    class DummyOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.chat = DummyChat()

    # Patch the OpenAI class inside LLMClient module
    monkeypatch.setattr(LLMClient, "OpenAI", DummyOpenAI)

    client = OpenAIClient()
    resp = client.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-test",
        temperature=0.3,
        response_format={"type": "object"},
        tools=[{"name": "f", "parameters": {}}],
    )

    assert resp.choices[0].message.content == "ok"
    payload = captured["payload"]
    assert payload["model"] == "gpt-test"
    assert payload["messages"][0]["role"] == "user"
    assert payload["temperature"] == 0.3
    # response_format converted to json_schema wrapper
    assert "response_format" in payload
    assert payload["response_format"]["type"] == "json_schema"
    assert "tools" in payload
    assert payload["tool_choice"] == "auto"


def test_deepseek_client_uses_openai_compatible_stub(monkeypatch):
    captured = {}

    class DummyChat:
        def completions_create(self, **kwargs):
            captured["payload"] = kwargs

            class Msg:
                def __init__(self):
                    self.content = "deepseek ok"
                    self.tool_calls = None

            class Choice:
                def __init__(self):
                    self.message = Msg()

            class Resp:
                def __init__(self):
                    self.choices = [Choice()]

            return Resp()

        @property
        def completions(self):
            return types.SimpleNamespace(create=self.completions_create)

    class DummyOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.chat = DummyChat()

    monkeypatch.setattr(LLMClient, "OpenAI", DummyOpenAI)

    client = DeepseekClient()
    resp = client.chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        model="deepseek-chat",
        temperature=0.0
    )

    assert resp.choices[0].message.content == "deepseek ok"
    payload = captured["payload"]
    assert payload["model"] == "deepseek-chat"
    assert payload["messages"][0]["content"] == "hello"


def test_doubao_client_builds_payload(monkeypatch):
    captured = {}

    class DummyChat:
        def completions_create(self, **kwargs):
            captured["payload"] = kwargs

            class Msg:
                def __init__(self):
                    self.content = "doubao ok"
                    self.tool_calls = None

            class Choice:
                def __init__(self):
                    self.message = Msg()

            class Resp:
                def __init__(self):
                    self.choices = [Choice()]

            return Resp()

        @property
        def completions(self):
            return types.SimpleNamespace(create=self.completions_create)

    class DummyOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.chat = DummyChat()

    monkeypatch.setattr(LLMClient, "OpenAI", DummyOpenAI)

    client = DoubaoClient()
    resp = client.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="doubao-model",
        temperature=0.2,
        max_tokens=128,
    )

    assert resp.choices[0].message.content == "doubao ok"
    payload = captured["payload"]
    assert payload["model"] == "doubao-model"
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 128
    assert payload["messages"][0]["role"] == "user"
    assert isinstance(payload["messages"][0]["content"], list) or isinstance(
        payload["messages"][0]["content"], str
    )
