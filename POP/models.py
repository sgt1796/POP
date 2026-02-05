"""Model names and default values.

This module centralises the default model names used for each
provider.  It mirrors the ``default_model`` dictionary from the
original POP implementation.  Other modules import this to obtain a
provider's default model.
"""

# Map provider class names to their default model identifiers
DEFAULT_MODEL = {
    "OpenAIClient": "gpt-5-nano",
    "GeminiClient": "gemini-2.5-flash",
    "DeepseekClient": "deepseek-chat",
    "DoubaoClient": "doubao-seed-1-6-flash-250715",
    "OllamaClient": "mistral:7b",
    # LocalPyTorchClient is a stub; choose an arbitrary default
    "LocalPyTorchClient": "local-llm",
}

__all__ = ["DEFAULT_MODEL"]