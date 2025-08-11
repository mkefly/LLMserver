import asyncio, signal
from contextlib import asynccontextmanager

class ConcurrencyGate:
    """Simple gate: cap concurrent streams, support drain on SIGTERM."""
    def __init__(self, max_concurrent: int, drain_seconds: int):
        self._sem = asyncio.Semaphore(max_concurrent)
        self._draining = asyncio.Event()
        self._drain_seconds = drain_seconds

    def install_sigterm_handler(self):
        loop = asyncio.get_event_loop()
        def _handler(*_):
            self._draining.set()
            # allow in-flight to complete; readiness probe should fail outside this lib
            loop.call_later(self._drain_seconds, lambda: None)
        signal.signal(signal.SIGTERM, _handler)

    # Convenience for tests and manual drains
    def start_drain(self):
        self._draining.set()

    @asynccontextmanager
    async def slot(self):
        if self._draining.is_set():
            raise RuntimeError("Draining")
        async with self._sem:
            yield
