"""Reliability utilities: concurrency gating and graceful drain."""
from __future__ import annotations
import asyncio, signal
from contextlib import asynccontextmanager

class ConcurrencyGate:
    """Caps concurrent streams and supports drain on SIGTERM."""
    def __init__(self, max_concurrent: int, drain_seconds: int):
        self._sem = asyncio.Semaphore(max_concurrent)
        self._draining = asyncio.Event()
        self._drain_seconds = drain_seconds

    def start_drain(self) -> None:
        self._draining.set()

    def install_sigterm_handler(self) -> None:
        loop = asyncio.get_event_loop()
        def _handler(*_):
            self.start_drain()
            loop.call_later(self._drain_seconds, lambda: None)
        try:
            signal.signal(signal.SIGTERM, _handler)
        except Exception:
            # Some environments (Windows) may not allow setting this.
            pass

    @asynccontextmanager
    async def slot(self):
        if self._draining.is_set():
            raise RuntimeError("Draining")
        async with self._sem:
            yield
