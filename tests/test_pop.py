# tests/test_pop.py
import types
import json
import builtins
import pytest
import POP
from POP import PromptFunction, get_text_snapshot


class DummyMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class DummyChoice:
    def __init__(self, message):
        self.message = message


class DummyToolFn:
    def __init__(self, arguments):
        self.arguments = arguments


class DummyToolCall:
    def __init__(self, arguments):
        self.function = DummyToolFn(arguments)


class DummyResponse:
    def __init__(self, content, tool_calls=None):
        self.choices = [DummyChoice(DummyMessage(content, tool_calls or []))]


def test_get_place_holder_from_prompt():
    pf = PromptFunction(
        prompt="Hello <<<name>>> and <<<what>>>!",
        sys_prompt="",
        client="openai"
    )
    # Internal list of placeholders should match the prompt
    assert sorted(pf.placeholders) == ["name", "what"]


def test_prepare_prompt_replaces_placeholders_and_appends_args():
    pf = PromptFunction(
        prompt="Say hi to <<<name>>>.",
        sys_prompt="",
        client="openai"
    )
    # Access the private method directly – fine in tests.
    result = pf._prepare_prompt("extra line", name="World")
    # Placeholder replaced
    assert "Say hi to World." in result
    # Extra arg appended
    assert "extra line" in result


def test_prepare_prompt_uses_sys_prompt_when_prompt_missing():
    pf = PromptFunction(
        prompt="",
        sys_prompt="System base with <<<topic>>>",
        client="openai"
    )
    # Here _get_place_holder looked at sys_prompt, but _prepare_prompt
    # will synthesize a "User instruction:" block.
    out = pf._prepare_prompt(
        "do something",
        topic="biology",
        ADD_BEFORE="before",
        ADD_AFTER="after"
    )
    # It should use "User instruction:" flow, not raises
    print(out)
    assert "User instruction:" in out
    assert "do something" in out
    assert "topic: biology" in out
    assert out.startswith("before")
    assert out.rstrip().endswith("after")


def test_execute_calls_client_and_returns_content(monkeypatch):
    # Capture what execute sends into the LLM client
    received = {}

    def fake_chat_completion(self, messages, model, temperature, **kwargs):
        # record important stuff
        received["messages"] = messages
        received["model"] = model
        received["temperature"] = temperature
        received["kwargs"] = kwargs
        # Return dummy OpenAI-like response
        return DummyResponse("dummy reply")

    # Patch OpenAIClient.chat_completion so no real API is called.
    monkeypatch.setattr(POP.OpenAIClient, "chat_completion", fake_chat_completion)

    pf = PromptFunction(prompt="Hi <<<who>>>", client="openai")
    reply = pf.execute(who="world", sys="extra sys", fmt={"type": "object"}, temp=0.11)

    assert reply == "dummy reply"
    # The client should have been called with formatted system + user messages
    assert received["model"] == pf.default_model_name
    assert pytest.approx(received["temperature"], rel=1e-6) == 0.11
    assert len(received["messages"]) == 2
    assert received["messages"][0]["role"] == "system"
    assert "Base system prompt" in received["messages"][0]["content"]
    assert "extra sys" in received["messages"][0]["content"]
    assert received["messages"][1]["role"] == "user"
    assert "Hi world" in received["messages"][1]["content"]


def test_execute_prefers_tool_call_arguments(monkeypatch):
    arguments = json.dumps({"x": 1, "y": 2})

    def fake_chat_completion(self, messages, model, temperature, **kwargs):
        # Return a response that has a function call
        tool_calls = [DummyToolCall(arguments)]
        return DummyResponse("plain text that should be ignored", tool_calls=tool_calls)

    monkeypatch.setattr(POP.OpenAIClient, "chat_completion", fake_chat_completion)

    pf = PromptFunction(prompt="Tool test", client="openai")
    out = pf.execute()
    # Should return the tool-call arguments, not the message content
    assert out == arguments


def test_set_temperature_overrides_default_in_execute(monkeypatch):
    recorded = {}

    def fake_chat_completion(self, messages, model, temperature, **kwargs):
        recorded["temperature"] = temperature
        return DummyResponse("ok")

    monkeypatch.setattr(POP.OpenAIClient, "chat_completion", fake_chat_completion)

    pf = PromptFunction(prompt="temp test", client="openai")
    pf.set_temperature(0.3)
    pf.execute()
    assert pytest.approx(recorded["temperature"], rel=1e-6) == 0.3


def test_get_text_snapshot_builds_correct_url_and_headers(monkeypatch):
    calls = {}

    class DummyResp:
        def __init__(self):
            self.text = "snapshot"
            self.status_code = 200

        def raise_for_status(self):
            pass

    def fake_get(url, headers=None):
        calls["url"] = url
        calls["headers"] = headers or {}
        return DummyResp()

    monkeypatch.setattr(POP.requests, "get", fake_get)

    out = get_text_snapshot(
        "https://example.com/page",
        use_api_key=False,
        return_format="markdown",
        timeout=10,
        target_selector=[".article"],
        wait_for_selector=[".loaded"],
        exclude_selector=[".ads"],
        remove_image=True,
        links_at_end=True,
        images_at_end=True,
        json_response=True,
        image_caption=True,
        cookie="session=abc"
    )

    assert out == "snapshot"
    assert calls["url"] == "https://r.jina.ai/https://example.com/page"

    h = calls["headers"]
    # since use_api_key=False, Authorization should not be set
    assert "Authorization" not in h
    assert h["X-Return-Format"] == "markdown"
    assert h["X-Timeout"] == 10
    assert h["X-Target-Selector"] == ".article"
    assert h["X-Wait-For-Selector"] == ".loaded"
    assert h["X-Remove-Selector"] == ".ads"
    assert h["X-Retain-Images"] == "none"
    assert h["X-With-Links-Summary"] == "true"
    assert h["X-With-Images-Summary"] == "true"
    assert h["Accept"] == "application/json"
    assert h["X-With-Generated-Alt"] == "true"
    assert h["X-Set-Cookie"] == "session=abc"
