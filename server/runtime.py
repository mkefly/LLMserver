"""MLServer runtime that hosts pluggable LLM adapters with streaming.

This file is intentionally small and stable; new frameworks plug in via
the registry and CapLLM.
"""
from __future__ import annotations
import asyncio, contextlib
from contextlib import asynccontextmanager
from typing import AsyncIterator
try:
    from mlserver import MLModel  # prod
except Exception:  # local tests without mlserver
    class MLModel: pass  # type: ignore[misc]

from .config import Params
from .observability import REQS, TOKENS, LAT, tracer, init_prom, init_tracing
from .io_utils import decode_input, pack_text, pack_chunk
from .reliability import ConcurrencyGate
from .registry import adapter as get_adapter, resolver as get_resolver, memory as get_memory  # noqa: F401

# Ensure resolvers and adapters register themselves on import
from .models import resolvers as _resolvers  # noqa: F401
from .adapters import __init__ as _adapters  # noqa: F401
from .memory import __init__ as _memory      # noqa: F401

class LLMUnifiedRuntime(MLModel):
    async def load(self) -> bool:
        init_tracing("mlserver")
        init_prom("PROM_PORT")
        raw = (self._settings.parameters or {}).dict() if getattr(self, "_settings", None) and self._settings.parameters else {}
        self.p = Params(**raw)

        # Resolve initial model
        self._resolver = get_resolver(self.p.resolver)()
        resolved = await self._resolver.resolve_initial(self.p.model_ref)

        # Instantiate adapter
        if self.p.adapter == "llama_index":
            self._adapter = get_adapter("llama_index")(resolved_uri=resolved, top_k=self.p.top_k)
        elif self.p.adapter == "langchain":
            # For code-factory: set resolver=static_uri and pass the factory path via model_ref
            self._adapter = get_adapter("langchain")(factory_path=self.p.model_ref)
        else:
            # Third-party adapter registered elsewhere
            self._adapter = get_adapter(self.p.adapter)(resolved_uri=resolved, top_k=self.p.top_k)

        await self._adapter.load()

        # Hot reload
        self._watch_task = None
        if self.p.hot_reload and hasattr(self._resolver, "watch"):
            async def _on_change(new_uri: str): await self._adapter.reload_from_uri(new_uri)
            self._watch_task = asyncio.create_task(self._resolver.watch(self.p.model_ref, self.p.hot_reload_interval_s, _on_change))

        # Reliability
        self._gate = ConcurrencyGate(self.p.max_concurrent_streams, self.p.drain_seconds)
        self._gate.install_sigterm_handler()
        return True

    async def finalize(self):
        if getattr(self, "_watch_task", None):
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task

    @asynccontextmanager
    async def _timeout(self):
        try:
            async with asyncio.timeout(self.p.timeout_s):
                yield
        except TimeoutError:
            raise

    async def predict(self, request):
        text, sid, params = await decode_input(request)
        REQS.labels(model=self.name, method="predict").inc()
        with tracer.start_as_current_span("predict", attributes={"model": self.name, "adapter": self.p.adapter, "sid": sid or ""}):
            with LAT.labels(self.name, "predict").time():
                async with self._timeout():
                    if self.p.engine_type == "chat":
                        out = await self._adapter.chat(text, session_id=sid, params=params)
                    elif self.p.engine_type == "query":
                        out = await self._adapter.query(text, session_id=sid, params=params)
                    else:
                        out = await self._adapter.retrieve(text, session_id=sid, params=params)
        return pack_text("output_text", out, self.name)

    async def predict_stream(self, request) -> AsyncIterator:
        async with self._gate.slot():
            text, sid, params = await decode_input(request)
            REQS.labels(model=self.name, method="predict_stream").inc()
            with tracer.start_as_current_span("predict_stream", attributes={"model": self.name, "adapter": self.p.adapter, "sid": sid or ""}):
                try:
                    async with self._timeout():
                        if self.p.engine_type == "chat":
                            agen = self._adapter.stream_chat(text, session_id=sid, params=params)
                        elif self.p.engine_type == "query":
                            agen = self._adapter.stream_query(text, session_id=sid, params=params)
                        else:
                            yield pack_chunk(self.name, await self._adapter.retrieve(text, session_id=sid, params=params), self.p.stream_format)
                            return
                        async for tok in agen:
                            TOKENS.labels(model=self.name).inc()
                            yield pack_chunk(self.name, tok, self.p.stream_format)
                except (TimeoutError, asyncio.CancelledError):
                    yield pack_chunk(self.name, "[stream-ended]", self.p.stream_format)
