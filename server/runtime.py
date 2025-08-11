"""MLServer runtime that hosts pluggable LLM adapters with streaming.

This file is intentionally small and stable; new frameworks plug in via
the registry and CapLLM.
"""
from __future__ import annotations
import asyncio, contextlib
from contextlib import asynccontextmanager
from typing import AsyncIterator
from .capabilities import CapLLM
from mlserver import MLModel
from .config import Params
from .observability import REQS, TOKENS, LAT, tracer, init_prom, init_tracing
from .io_utils import decode_input, pack_text, pack_chunk
from .reliability import ConcurrencyGate
from .registry import adapter as get_adapter, resolver as get_resolver, memory as get_memory  # noqa: F401

import inspect
import logging

log = logging.getLogger(__name__)

def _build_adapter(adapter_name: str, resolved_uri: str, p: Params) -> CapLLM:
    """
    Create an adapter instance from the registered factory, passing only
    the kwargs it accepts. This makes adding new adapters zero-touch.
    """
    factory = get_adapter(adapter_name)

    # Common kwargs we *may* provide to any adapter
    candidate_kwargs = {
        "resolved_uri": resolved_uri,      # for UC / MLflow-backed adapters
        "model_ref": p.model_ref,          # for code-factory style adapters (e.g., LangChain)
        "top_k": getattr(p, "top_k", None),
        "engine_type": getattr(p, "engine_type", None),
    }

    # Optional pass-through bag (if you add Params.adapter_kwargs: dict = Field(default_factory=dict))
    adapter_kwargs = getattr(p, "adapter_kwargs", None) or {}
    candidate_kwargs.update(adapter_kwargs)

    # Filter by the factory's accepted parameters
    sig = inspect.signature(factory)
    accepted = {k: v for k, v in candidate_kwargs.items()
                if k in sig.parameters and v is not None}

    # Friendly diagnostics for mismatches
    missing_required = [
        name for name, param in sig.parameters.items()
        if param.default is inspect._empty and name not in accepted and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY
        )
    ]
    if missing_required:
        log.warning(
            "Adapter '%s' factory requires missing args %s. "
            "Provided keys: %s. Consider supplying via Params.adapter_kwargs.",
            adapter_name, missing_required, list(candidate_kwargs.keys()),
        )

    try:
        adapter = factory(**accepted)
        log.debug("Constructed adapter '%s' with args: %s", adapter_name, accepted)
        return adapter
    except TypeError as e:
        raise ValueError(
            f"Could not construct adapter '{adapter_name}'. "
            f"Factory parameters: {list(sig.parameters.keys())}. "
            f"Provided kwargs: {accepted}. Original error: {e}"
        ) from e


class LLMUnifiedRuntime(MLModel):
    def __init__(self, *args, **kwargs):
        # makes mypy happy if it inspects attributes created in load()
        super().__init__(*args, **kwargs) if hasattr(super(), "__init__") else None
        self._adapter: CapLLM | None = None

    async def load(self) -> bool:
        init_tracing("mlserver")
        init_prom("PROM_PORT")
        raw = (self._settings.parameters or {}).dict() if getattr(self, "_settings", None) and self._settings.parameters else {}
        self.p = Params(**raw)

        # Resolve initial model
        self._resolver = get_resolver(self.p.resolver)()
        resolved = await self._resolver.resolve_initial(self.p.model_ref)

        # Instantiate adapter
        self._adapter = _build_adapter(self.p.adapter, resolved, self.p)
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
        assert self._adapter is not None  # for type checkers
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
