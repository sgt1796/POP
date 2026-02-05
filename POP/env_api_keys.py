"""Environment key helpers.

Providers may require API keys to function.  This module defines
utility functions to inspect whether the necessary keys are present in
the environment.  Consumers can call :func:`has_api_key` to check
whether a provider is ready for use.
"""

import os
from typing import Optional
from dotenv import load_dotenv

if not load_dotenv():
    print("No .env file found or could not be loaded.")

# Mapping from provider identifier to the environment variable used
REQUIRED_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "doubao": "DOUBAO_API_KEY",
    # local and ollama do not require API keys by default
}



def has_api_key(provider: str) -> bool:
    """Return True if the required API key for *provider* is set."""
    env_var = REQUIRED_KEYS.get(provider)
    return bool(os.getenv(env_var)) if env_var else True


__all__ = ["has_api_key"]