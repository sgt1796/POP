"""Top‑level package for the restructured POP library.

This package exposes the main classes and helper functions for creating
prompt functions, embeddings and conversation contexts.  It also
re‑exports provider registry functions for convenience.

Example usage::

    from pop import PromptFunction, Context, list_providers

    ctx = Context(system="You are a helpful assistant")
    pf = PromptFunction(sys_prompt="Translate", prompt="<<<text>>>", client="openai")
    result = pf.execute(text="Hello")
    print(result)
"""

from .prompt_function import PromptFunction
from .embedder import Embedder
from .context import Context, MessageBlock
from .api_registry import (
    list_providers,
    list_default_model,
    list_models,
    get_default_model,
    get_model,
    get_client,
)

__all__ = [
    "PromptFunction",
    "Embedder",
    "Context",
    "MessageBlock",
    "list_providers",
    "list_default_model",
    "list_models",
    "get_default_model",
    "get_model",
    "get_client",
]
