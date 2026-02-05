"""
Unicode sanitisation helpers.

Language models occasionally emit characters outside the basic
multilingual plane.  To ensure consistent downstream processing,
this module provides helper functions to normalise and strip
unsupported Unicode characters.
"""

import unicodedata

def sanitize(text: str) -> str:
    """Normalise and strip diacritics from a string.

    Uses NFKD normalisation and discards combining characters.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join([c for c in normalized if not unicodedata.combining(c)])
