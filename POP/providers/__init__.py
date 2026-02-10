"""Provider registry.

This subpackage contains one module per supported provider.  The
``CLIENTS`` mapping associates short provider names (e.g. ``"openai"``)
with the corresponding client class.  External consumers should use
``POP.api_registry.get_client()`` rather than importing classes
directly from this module.

Adding a new provider is as simple as creating a new module in this
directory that defines a class derived from :class:`POP.providers.llm_client.LLMClient`
and then registering it here.
"""

from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .deepseek_client import DeepseekClient
from .local_client import LocalPyTorchClient
from .doubao_client import DoubaoClient
from .ollama_client import OllamaClient


# Map short provider names to their client classes.  New providers
# should be inserted here.
DEFAULT_CLIENTS = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "deepseek": DeepseekClient,
    "local": LocalPyTorchClient,
    "doubao": DoubaoClient,
    "ollama": OllamaClient,
}

__all__ = ["DEFAULT_CLIENTS"]
