# tests/test_embedder.py
import types
import numpy as np
import pytest

import Embedder
from Embedder import Embedder as EmbedderCls


def test_get_embedding_requires_list_input():
    e = EmbedderCls(model_name="dummy", use_api="openai")
    with pytest.raises(ValueError):
        e.get_embedding("not a list")  # type: ignore[arg-type]


def test_invalid_api_raises():
    with pytest.raises(ValueError):
        EmbedderCls(model_name="dummy", use_api="unknown-api")


def test_openai_embedding_happy_path(monkeypatch):
    # Prepare a dummy client with an embeddings.create method
    class DummyEmbeddings:
        def create(self, input, model):
            # input should be list of strings with newlines removed
            assert input == ["hello world", "second"]
            assert model == "text-embedding-3-small"
            # Build something that has .data[i].embedding
            item1 = types.SimpleNamespace(embedding=[0.1, 0.2])
            item2 = types.SimpleNamespace(embedding=[0.3, 0.4])
            return types.SimpleNamespace(data=[item1, item2])

    # Instantiate with use_api='openai', but then patch the client
    e = EmbedderCls(model_name=None, use_api="openai")
    e.client = types.SimpleNamespace(embeddings=DummyEmbeddings())

    arr = e.get_embedding(["hello\nworld", "second"])
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    # dtype "f" -> float32
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, np.array([[0.1, 0.2], [0.3, 0.4]], dtype="f"))


def test_jina_segmenter_returns_chunks(monkeypatch):
    # Only test _Jina_segmenter, not the whole _get_jina_embedding logic
    e = EmbedderCls(model_name=None, use_api="jina")

    class DummyResp:
        def __init__(self):
            self.status_code = 200

        def json(self):
            return {"chunks": ["a", "b", "c"]}

    calls = {}

    def fake_post(url, headers=None, json=None):
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        return DummyResp()

    monkeypatch.setattr(Embedder.HTTPRequests, "post", fake_post)

    chunks = e._Jina_segmenter("some text", max_token=100)
    assert chunks == ["a", "b", "c"]
    assert calls["url"].endswith("/segment.jina.ai/")
    assert "Authorization" in calls["headers"]
    assert calls["json"]["content"] == "some text"
    assert calls["json"]["max_chunk_length"] == 100
