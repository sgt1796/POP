"""Asynchronous event stream implementation.

The :class:`EventStream` class provides a simple interface for pushing
events from producer coroutines and consuming them via async iteration.
Instances of this class behave similarly to the `EventStream` used in
the TypeScript implementation of `pi‑agent`.  In particular they
support:

* Calling :meth:`push` to append an event to the stream.
* Calling :meth:`end` to close the stream and optionally attach a
  result value.
* Using ``async for event in stream`` to consume events in the order
  they were produced.
* Awaiting :meth:`result` to retrieve the final result after the
  stream has ended.

Internally the implementation uses an :class:`asyncio.Queue` to buffer
events between producers and consumers.  A sentinel value of ``None``
marks the end of the stream.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Generic, Optional, TypeVar

# Generic type variables for event payloads and the final result
TEvent = TypeVar("TEvent")
TResult = TypeVar("TResult")


class EventStream(Generic[TEvent, TResult]):
    """A simple asynchronous event queue.

    Parameters
    ----------
    done_predicate : callable, optional
        A function that receives an event and returns ``True`` when the
        event marks the end of the stream.  When the predicate
        evaluates to ``True`` the event is passed through and the
        stream is automatically closed.  If omitted no event will
        automatically close the stream and you must call
        :meth:`end` yourself.

    result_selector : callable, optional
        A function that receives an event and returns the result value
        for the stream when the event marks the end of the stream.
        Ignored when ``done_predicate`` is not provided.

    Notes
    -----
    Consumers can iterate over events with ``async for``.  When the
    stream is closed the iteration ends.  Use :meth:`result` to
    retrieve the value supplied to :meth:`end` or computed by the
    ``result_selector``.
    """

    def __init__(
        self,
        done_predicate: Optional[callable[[TEvent], bool]] = None,
        result_selector: Optional[callable[[TEvent], TResult]] = None,
    ) -> None:
        self._queue: asyncio.Queue[Optional[TEvent]] = asyncio.Queue()
        self._result: Optional[TResult] = None
        self._done: asyncio.Event = asyncio.Event()
        self._closed: bool = False
        self._done_predicate = done_predicate
        self._result_selector = result_selector

    async def __aiter__(self) -> AsyncIterator[TEvent]:
        """Iterate over events until the stream is closed."""
        while True:
            event = await self._queue.get()
            if event is None:
                self._queue.task_done()
                break
            self._queue.task_done()
            yield event
            # If the producer specified a done predicate and it matches,
            # automatically end the stream and compute the result.
            if self._done_predicate and self._done_predicate(event):
                if self._result_selector:
                    try:
                        self._result = self._result_selector(event)
                    except Exception:
                        # swallow exceptions from result selector
                        self._result = None
                self.end()

    def push(self, event: TEvent) -> None:
        """Append an event to the stream.

        Parameters
        ----------
        event : TEvent
            The event to add.  Must not be ``None``.
        """
        if self._closed:
            raise RuntimeError("Cannot push to a closed EventStream")
        self._queue.put_nowait(event)

    def end(self, result: Optional[TResult] = None) -> None:
        """Close the stream and optionally set the final result.

        Parameters
        ----------
        result : TResult, optional
            A value to return from :meth:`result`.  Overrides any
            value computed by a result selector.
        """
        if self._closed:
            return
        self._closed = True
        if result is not None:
            self._result = result
        # Put sentinel into queue to stop iteration
        self._queue.put_nowait(None)
        # Signal to waiters that the stream is done
        self._done.set()

    async def result(self) -> Optional[TResult]:
        """Wait for the stream to close and return the final result.

        Returns
        -------
        Optional[TResult]
            The value supplied to :meth:`end` or computed by the
            result selector.  Returns ``None`` if no result was set.
        """
        await self._done.wait()
        return self._result