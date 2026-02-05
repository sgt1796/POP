"""Provider and model registry.

This module offers helper functions to inspect and instantiate LLM
providers.  It borrows the concept of a central registry from the
``api‑registry.ts`` file in the pi‑ai project and exposes:

* :func:`list_providers` – return a list of provider identifiers.
* :func:`list_default_model` – return a mapping of provider identifiers
  to their default model names.
* :func:`_get_default_model` – return the default model for a provider.
* :func:`list_models` – return a mapping of provider identifiers to
  all available model names from ``providers.json``.
* :func:`get_model` – instantiate and return a client based on a model name.
* :func:`get_client` – instantiate and return a client class given
  its provider identifier.
"""

from __future__ import annotations

import json
from os import path
from typing import Dict, List, Optional

from .providers import DEFAULT_CLIENTS
from .models import DEFAULT_MODEL

_PROVIDERS_JSON = path.join(path.dirname(__file__), "providers", "providers.json")


def _load_provider_catalog() -> Dict[str, Dict[str, object]]:
    if not path.exists(_PROVIDERS_JSON):
        return {}
    with open(_PROVIDERS_JSON, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    return data


def _flatten_models(provider_data: Dict[str, object]) -> List[str]:
    models: List[str] = []
    seen: set[str] = set()
    for value in provider_data.values():
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            for item in value:
                if item not in seen:
                    seen.add(item)
                    models.append(item)
    return models


def list_providers() -> List[str]:
    """Return the list of registered provider identifiers."""
    return list(DEFAULT_CLIENTS.keys())


def list_default_model() -> Dict[str, str]:
    """Return a mapping of provider identifiers to default model names."""
    models: Dict[str, str] = {}
    for name, cls in DEFAULT_CLIENTS.items():
        model_name = DEFAULT_MODEL.get(cls.__name__)
        if model_name:
            models[name] = model_name
    return models


def _get_default_model(provider_name: str) -> Optional[str]:
    """Return the default model name for a provider."""
    cls = DEFAULT_CLIENTS.get(provider_name)
    if cls is None:
        return None
    return DEFAULT_MODEL.get(cls.__name__)


def list_models() -> Dict[str, List[str]]:
    """Return a mapping of provider identifiers to available model names."""
    providers = _load_provider_catalog()
    models: Dict[str, List[str]] = {}
    for provider_name, provider_data in providers.items():
        if isinstance(provider_data, dict):
            flattened = _flatten_models(provider_data)
            if flattened:
                models[provider_name] = flattened
    return models

def get_client(provider_name: str, model_name: Optional[str] = None) -> Optional[object]:
    """Instantiate and return an LLM client for the given provider.

    Parameters
    ----------
    provider_name : str
        The provider identifier (e.g. ``"openai"``).

    model_name : str
        The model name (available in provider.json).

    Returns
    -------
    object | None
        An instance of the provider's client class if recognised,
        otherwise ``None``.
    """
    if model_name is None:
        model_name = _get_default_model(provider_name)  
    cls = DEFAULT_CLIENTS.get(provider_name)

    if cls is not None:
        return cls(model=model_name)
    
    return None


__all__ = [
    "list_providers",
    "list_default_model",
    "list_models",
    "get_model",
    "get_client",
]
