import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from POP.env_api_keys import has_api_key


def _build_fake_response(content: str | None = None, tool_arguments: str | None = None) -> SimpleNamespace:
    tool_calls = None
    if tool_arguments is not None:
        function = SimpleNamespace(arguments=tool_arguments)
        tool_calls = [SimpleNamespace(function=function)]
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def fake_response():
    return _build_fake_response


@pytest.fixture(scope="session")
def tools():
    tools_path = Path(__file__).resolve().parents[1] / "other_doc" / "function_calls.py"
    spec = importlib.util.spec_from_file_location("function_calls", tools_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load tools module from {tools_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.tools


def _is_live_enabled(provider: str | None = None) -> bool:
    if provider:
        return has_api_key(provider)
    return True


@pytest.fixture(scope="session")
def live_enabled():
    return _is_live_enabled
