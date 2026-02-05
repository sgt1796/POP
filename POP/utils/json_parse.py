"""
JSON parsing utilities.

In the pi‑ai project this module contains helpers for streaming JSON
parsing and error handling.  Here we provide simple functions to
parse JSON strings into Python objects and to safely extract values.
"""

import json
from typing import Any, Dict, Optional

def parse_json(json_str: str) -> Any:
    """Parse a JSON string into a Python object.

    Raises a ValueError if the input is not valid JSON.
    """
    return json.loads(json_str)

def get_value(data: Dict[str, Any], key: str, default: Optional[Any] = None) -> Any:
    """Safely get a value from a dict, returning a default if the key is missing."""
    return data.get(key, default)
