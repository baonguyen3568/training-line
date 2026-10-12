"""In-process event bus for graph runs: the executor thread publishes, WebSocket clients replay + follow.

One process, one user (P009 §11). Events are kept in memory per graph run so a client that connects late (or
reconnects) first receives the backlog, then live events. Thread-safe publish; async consumption.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

Event = dict[str, Any]
_END = {"type": "_end"}


class EventBus:
    def __init__(self, max_events_per_run: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, list[Event]] = {}
        self._closed: set[str] = set()
        self._waiters: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Event]]]] = {}
        self._max = max_events_per_run

    def open(self, run_id: str) -> None:
        with self._lock:
            self._events.setdefault(run_id, [])

    def known(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._events

    def publish(self, run_id: str, event: Event) -> None:
        """Called from any thread (executor)."""
        with self._lock:
            events = self._events.setdefault(run_id, [])
            if len(events) < self._max:
                events.append(event)
            waiters = list(self._waiters.get(run_id, ()))
        for loop, queue in waiters:
            loop.call_soon_threadsafe(queue.put_nowait, event)

    def close(self, run_id: str) -> None:
        """No more events for ``run_id``; followers stop after the backlog."""
        with self._lock:
            self._closed.add(run_id)
            waiters = list(self._waiters.get(run_id, ()))
        for loop, queue in waiters:
            loop.call_soon_threadsafe(queue.put_nowait, _END)

    def backlog(self, run_id: str) -> list[Event]:
        with self._lock:
            return list(self._events.get(run_id, ()))

    async def follow(self, run_id: str) -> AsyncIterator[Event]:
        """Backlog first, then live events until the run is closed."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with self._lock:
            backlog = list(self._events.get(run_id, ()))
            closed = run_id in self._closed
            if not closed:
                self._waiters.setdefault(run_id, []).append((loop, queue))
        try:
            for event in backlog:
                yield event
            if closed:
                return
            while True:
                event = await queue.get()
                if event is _END:
                    return
                yield event
        finally:
            with self._lock:
                waiters = self._waiters.get(run_id, [])
                if (loop, queue) in waiters:
                    waiters.remove((loop, queue))
