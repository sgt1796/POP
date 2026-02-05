"""
Overflow helpers.

This stub module mirrors pi‑ai's overflow handlers for managing
large outputs.  In this rewrite we provide a basic function to
check if a message exceeds a threshold and to truncate it.
"""

from typing import List

def truncate_messages(messages: List[str], max_length: int = 4096) -> List[str]:
    """Truncate a list of messages so that their concatenated length does not exceed max_length.

    Parameters
    ----------
    messages:
        A list of message strings.
    max_length:
        The maximum total length allowed.

    Returns
    -------
    List[str]
        A list of messages truncated to the allowed length.
    """
    total = 0
    result: List[str] = []
    for msg in messages:
        if total + len(msg) > max_length:
            break
        result.append(msg)
        total += len(msg)
    return result
