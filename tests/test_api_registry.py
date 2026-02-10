from POP.api_registry import (
    list_providers,
    list_default_model,
    list_models,
    get_client,
    get_model,
)
from POP.providers.local_client import LocalPyTorchClient


def test_list_providers_contains_defaults():
    providers = list_providers()
    assert "openai" in providers
    assert "local" in providers


def test_list_default_model_contains_mapping():
    defaults = list_default_model()
    assert defaults.get("openai") is not None
    assert defaults.get("local") == "local-llm"


def test_list_models_includes_known_openai_model():
    models = list_models()
    assert "openai" in models
    assert "gpt-4o" in models["openai"]


def test_get_client_local():
    client = get_client("local")
    assert isinstance(client, LocalPyTorchClient)


def test_get_model_uses_default_mapping():
    client = get_model("local-llm")
    assert isinstance(client, LocalPyTorchClient)
