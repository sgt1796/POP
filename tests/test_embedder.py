from types import SimpleNamespace
import builtins
import sys

import numpy as np
import pytest

import POP.embedder as emb_mod
from POP.embedder import Embedder


def test_invalid_use_api_raises():
    with pytest.raises(ValueError):
        Embedder(use_api="bad-api")


def test_openai_embedding_stub(monkeypatch):
    class DummyEmbeddings:
        def create(self, input, model):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in input])

    class DummyClient:
        def __init__(self, api_key=None):
            self.embeddings = DummyEmbeddings()

    monkeypatch.setattr(emb_mod, "openai", SimpleNamespace(Client=DummyClient))
    embedder = Embedder(use_api="openai", model_name="text-embedding-3-small")
    vecs = embedder.get_embedding(["hello", "world"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (2, 2)


def test_jina_embedding_stub(monkeypatch):
    class DummyResponse:
        status_code = 200

        def json(self):
            return {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ]
            }

    def dummy_post(url, headers=None, json=None):
        return DummyResponse()

    monkeypatch.setattr(emb_mod.HTTPRequests, "post", dummy_post)
    embedder = Embedder(use_api="jina", model_name="jina-embeddings-v3")
    vecs = embedder.get_embedding(["a", "b"])
    assert vecs.shape == (2, 3)


def test_get_embedding_requires_list(monkeypatch):
    class DummyEmbeddings:
        def create(self, input, model):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in input])

    class DummyClient:
        def __init__(self, api_key=None):
            self.embeddings = DummyEmbeddings()

    monkeypatch.setattr(emb_mod, "openai", SimpleNamespace(Client=DummyClient))
    embedder = Embedder(use_api="openai", model_name="text-embedding-3-small")
    with pytest.raises(ValueError):
        embedder.get_embedding("not-a-list")


def test_local_embedder_without_transformers_raises_helpful_error(monkeypatch):
    fake_torch = SimpleNamespace(float16="float16")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("transformers unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ImportError, match="optional dependency 'transformers'"):
        Embedder(model_name="local-model")
