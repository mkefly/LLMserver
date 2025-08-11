import asyncio, contextlib
from contextlib import asynccontextmanager
from typing import AsyncIterator
from mlserver import MLModel
from .config import Params
from .observability import REQS, TOKENS, LAT, ACTIVE, tracer, init_prom, init_tracing
from .io_utils import decode_input, pack_text, pack_chunk
from .reloader import uc_watch
from .reliability import ConcurrencyGate
from .flavors.llama_index import LlamaIndexStrategy
from .flavors.langchain import LangChainStrategy

class LLMUnifiedRuntime(MLModel):
    async def load(self) -> bool:
        init_tracing("mlserver")
        init_prom("PROM_PORT")

        raw = (self._settings.parameters or {}).dict() if self._settings and self._settings.parameters else {}
        self.p = Params(**raw)

        # reliability
        self._gate = ConcurrencyGate(self.p.max_concurrent_streams, self.p.drain_seconds)
        self._gate.install_sigterm_handler()

        # strategy
        if self.p.framework == "llamaindex":
            self.strategy = LlamaIndexStrategy(self.p.engine_type, self.p.top_k)
            await self.strategy.load_from_stage(self.p.uc_model_uri)
        else:
            self.strategy = LangChainStrategy(self.p.langchain_factory, self.p.use_history)
            await self.strategy.load()

        # UC hot-reload
        self._watch_task = None
        if self.p.hot_reload and self.p.framework == "llamaindex":
            async def _on_new(resolved_uri: str):
                await self.strategy.reload_from_version(resolved_uri)
            self._watch_task = asyncio.create_task(uc_watch(self.p.uc_model_uri, self.p.hot_reload_interval_s, _on_new))

        return True

    async def finalize(self):
        if self._watch_task:
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
        text, sid, call = await decode_input(request)
        REQS.labels(model=self.name, method="predict").inc()
        with tracer.start_as_current_span("predict", attributes={"model": self.name, "framework": self.p.framework, "sid": sid or ""}):
            with LAT.labels(self.name, "predict").time():
                async with self._timeout():
                    out = await self.strategy.predict_text(text, sid, call)
        return pack_text("output_text", out, self.name)

    async def predict_stream(self, request) -> AsyncIterator:
        async with self._gate.slot():
            text, sid, call = await decode_input(request)
            REQS.labels(model=self.name, method="predict_stream").inc()
            ACTIVE.labels(model=self.name).inc()
            try:
                with tracer.start_as_current_span("predict_stream", attributes={"model": self.name, "framework": self.p.framework, "sid": sid or ""}):
                    async with self._timeout():
                        async for t in self.strategy.stream_tokens(text, sid, call):
                            TOKENS.labels(model=self.name).inc()
                            yield pack_chunk(self.name, t, self.p.stream_format)
            except (TimeoutError, asyncio.CancelledError):
                yield pack_chunk(self.name, "[stream-ended]", self.p.stream_format)
            finally:
                pass
