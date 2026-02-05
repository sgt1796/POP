from pop.env_api_keys import has_api_key


def test_has_api_key_true_when_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert has_api_key("openai") is True


def test_has_api_key_false_when_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert has_api_key("openai") is False


def test_has_api_key_for_keyless_provider():
    assert has_api_key("local") is True
