"""
Validation helpers.

This module contains light-weight validation routines for inputs,
outputs and schemas.  The original pi‑ai project uses zod for type
checking; here we use simple runtime checks and raise ValueError on
failure.
"""

from typing import Any
import json

def validate_not_empty(value: Any, message: str = "Value must not be empty") -> None:
    """Raise a ValueError if the provided value is empty or falsy."""
    if not value:
        raise ValueError(message)

def validate_json(value: str, message: str = "Invalid JSON") -> Any:
    """Parse a string as JSON, raising a ValueError if it fails."""
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"{message}: {e}")
