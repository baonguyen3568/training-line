"""In-process event bus for graph runs: the executor thread publishes, WebSocket clients replay + follow.

One process, one user (P009 §11). A client that connects late or reconnects first receives the backlog, then live
events. State events (``node_status``, ``run_status``) are always kept so a reconnecting client never gets stuck on
"running"; when the backlog is full the OLDEST other events (metrics, logs) are dropped and a ``backlog_truncated``
event is replayed first. Only the last ``keep_closed_runs`` finished runs stay in memory.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from typing import Any

Event = dict[str, Any]
STATE_EVENTS = frozenset({"node_status", "run_status"})
_END: Event = {"type": "_end"}


class _RunLog:
    def __init__(self, max_other: int) -> None:
        self.state: list[tuple[int, Event]] = []
        self.other: deque[tuple[int, Event]] = deque(maxlen=max_other)
        self.dropped = 0
        self.closed = False
        self.waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Event]]] = []

    def add(self, seq: int, event: Event) -> None:
        if event.get("type") in STATE_EVENTS:
            self.state.append((seq, event))
            return
        if self.other.maxlen is not None and len(self.other) == self.other.maxlen:
            self.dropped += 1
        self.other.append((seq, event))

    def backlog(self) -> list[Event]:
        merged = [e for _, e in sorted([*self.state, *self.other], key=lambda item: item[0])]
        if self.dropped:
            merged.insert(0, {"type": "backlog_truncated", "dropped": self.dropped})
        return merged


class EventBus:
    def __init__(self, max_other_events_per_run: int = 10_000, keep_closed_runs: int = 20) -> None:
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, _RunLog] = OrderedDict()
        self._seq = itertools.count()
        self._max_other = max_other_events_per_run
        self._keep_closed = keep_closed_runs

    def open(self, run_id: str) -> None:
        with self._lock:
            self._runs.setdefault(run_id, _RunLog(self._max_other))

    def known(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._runs

    def publish(self, run_id: str, event: Event) -> None:
        """Called from any thread (executor)."""
        with self._lock:
            log = self._runs.setdefault(run_id, _RunLog(self._max_other))
            log.add(next(self._seq), event)
            waiters = list(log.waiters)
        for loop, queue in waiters:
            loop.call_soon_threadsafe(queue.put_nowait, event)

    def close(self, run_id: str) -> None:
        """No more events for ``run_id``; followers stop after the backlog. Old closed runs are forgotten."""
        with self._lock:
            log = self._runs.get(run_id)
            if log is None:
                return
            log.closed = True
            waiters = list(log.waiters)
            closed = [rid for rid, rl in self._runs.items() if rl.closed and not rl.waiters]
            for rid in closed[: max(0, len(closed) - self._keep_closed)]:
                del self._runs[rid]
        for loop, queue in waiters:
            loop.call_soon_threadsafe(queue.put_nowait, _END)

    def backlog(self, run_id: str) -> list[Event]:
        with self._lock:
            log = self._runs.get(run_id)
            return log.backlog() if log else []

    async def follow(self, run_id: str) -> AsyncIterator[Event]:
        """Backlog first, then live events until the run is closed."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with self._lock:
            log = self._runs.get(run_id)
            if log is None:
                return
            backlog = log.backlog()
            closed = log.closed
            if not closed:
                log.waiters.append((loop, queue))
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
                if (loop, queue) in log.waiters:
                    log.waiters.remove((loop, queue))
