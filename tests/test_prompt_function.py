from __future__ import annotations

import json
from types import SimpleNamespace

from POP.prompt_function import PromptFunction
from POP.providers.llm_client import LLMClient


class DummyClient(LLMClient):
    def __init__(self, response):
        self.response = response
        self.model_name = "dummy-model"
        self.last_call = None

    def chat_completion(self, messages, model, temperature=0.0, **kwargs):
        self.last_call = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            **kwargs,
        }
        return self.response


def test_prepare_prompt_replacements(fake_response):
    client = DummyClient(fake_response("ok"))
    pf = PromptFunction(prompt="Hello <<<name>>>", client=client)
    result = pf._prepare_prompt("Extra", name="World", ADD_BEFORE="Before", ADD_AFTER="After")
    assert result == "Before\nHello World\nExtra\nAfter"


def test_prepare_prompt_sys_fallback(fake_response):
    client = DummyClient(fake_response("ok"))
    pf = PromptFunction(sys_prompt="System instructions", prompt="", client=client)
    result = pf._prepare_prompt(foo="bar")
    assert result == "User instruction:\nfoo: bar"


def test_execute_passes_options_and_defaults(fake_response):
    client = DummyClient(fake_response("ok"))
    pf = PromptFunction(prompt="Hello <<<name>>>", client=client)
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    tools = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]
    images = ["http://example.com/img.png"]

    pf.execute(name="World", tools=tools, fmt=fmt, images=images)

    assert client.last_call is not None
    assert client.last_call["tools"] == tools
    assert client.last_call["tool_choice"] == "auto"
    assert client.last_call["response_format"] == fmt
    assert client.last_call["images"] == images


def test_execute_returns_tool_call_arguments(fake_response):
    tool_args = json.dumps({"description": "walk", "when": "9am"})
    client = DummyClient(fake_response(None, tool_arguments=tool_args))
    pf = PromptFunction(prompt="<<<input>>>", client=client)
    result = pf.execute(input="Remind me to walk at 9am.", tools=[{"type": "function", "function": {"name": "x"}}])
    assert result == tool_args


def test_improve_prompt_replaces_and_strips_output(monkeypatch, fake_response):
    client = DummyClient(fake_response("unused"))
    pf = PromptFunction(sys_prompt="Base prompt", prompt="", client=client)
    monkeypatch.setattr(pf, "execute", lambda *args, **kwargs: "# OUTPUT\nImproved prompt")
    improved = pf.improve_prompt(replace=True)
    assert improved == "Improved prompt"
    assert pf.sys_prompt == "Improved prompt"


def test_load_prompt_resolves_relative_path():
    content = PromptFunction.load_prompt("prompts/json_formatter_prompt.md")
    assert "Generate a JSON Schema" in content


def test_generate_schema_default_prompt_saves(tmp_path, monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"name": "test", "schema": {"type": "object"}})))]
    )
    client = DummyClient(response)
    pf = PromptFunction(prompt="Return the square of an integer.", client=client)
    monkeypatch.chdir(tmp_path)

    schema = pf.generate_schema(save=True)

    assert schema["name"] == "test"
    saved = tmp_path / "schemas" / "test.json"
    assert saved.exists()


def test_save_writes_prompt(tmp_path, fake_response):
    client = DummyClient(fake_response("ok"))
    pf = PromptFunction(prompt="Hello", client=client)
    target = tmp_path / "prompt.txt"
    pf.save(str(target))
    assert target.read_text(encoding="utf-8") == "Hello"


def test_execute_populates_usage_metadata(fake_response):
    client = DummyClient(fake_response("ok"))
    pf = PromptFunction(prompt="Hello <<<name>>>", client=client)

    result = pf.execute(name="World")

    assert result == "ok"
    assert pf.last_usage is not None
    assert pf.last_usage["provider"] == "dummy"
    assert pf.last_usage["model"] == "dummy-model"
    assert pf.last_usage["source"] in {"estimate", "hybrid", "provider"}
    assert len(pf.usage_history) == 1
    assert pf.usage_history[0] == pf.last_usage


def test_usage_totals_accumulate(fake_response):
    client = DummyClient(fake_response("ok"))
    pf = PromptFunction(prompt="Hi <<<name>>>", client=client)

    pf.execute(name="A")
    pf.execute(name="B")
    summary = pf.get_usage_summary()

    assert summary["calls"] == 2
    assert summary["input_tokens"] >= 0
    assert summary["output_tokens"] >= 0
    assert summary["total_tokens"] >= 0


def test_usage_history_is_bounded(fake_response):
    client = DummyClient(fake_response("ok"))
    pf = PromptFunction(prompt="Hi <<<name>>>", client=client)

    for idx in range(205):
        pf.execute(name=str(idx))

    assert pf.usage_history.maxlen == 200
    assert len(pf.usage_history) == 200
