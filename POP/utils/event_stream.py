"""
Utility helpers for streaming events.

This module provides a simple event streaming interface.  In POP's
rewritten architecture we mirror the ``event-stream`` helper from
the original pi-ai project but do not implement a full server-sent
events protocol.  Instead, we define a ``EventStream`` class and
a helper function ``to_event_stream`` that can wrap a generator.
"""

from typing import Iterable, Iterator, Any, Dict

class EventStream:
    """Wrap a generator to provide an iterator of event dictionaries.

    An event dictionary contains at least a ``type`` key (e.g.
    ``start``, ``text_start``, ``text_delta``, ``text_end``, ``done``)
    and a ``data`` key.  Consumers of this class can send these events
    to clients over websockets or Server‑Sent Events.
    """
    def __init__(self, generator: Iterable):
        self._iterator = iter(generator)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self

    def __next__(self) -> Dict[str, Any]:
        return next(self._iterator)

def to_event_stream(generator: Iterable) -> EventStream:
    """Convert a plain generator into an EventStream.

    Parameters
    ----------
    generator:
        A generator which yields dictionaries representing events.

    Returns
    -------
    EventStream
        An iterator wrapper around the generator.
    """
    return EventStream(generator)
