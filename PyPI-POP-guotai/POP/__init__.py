"""
POP - Prompt Oriented Programming
Reusable, mutable, prompt functions for LLMs.

Author: Guotai Shen
"""

from .POP import PromptFunction
from .Embedder import Embedder

# Versioning
__version__ = "0.3.2"
__author__ = "Guotai Shen"
__license__ = "MIT"

# Expose key functionalities
__all__ = ["PromptFunction", "Embedder"]
