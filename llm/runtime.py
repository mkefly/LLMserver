import asyncio, os
from typing import AsyncIterator, Dict, Any, Optional
from fastapi import Depends
from starlette.responses import Response, StreamingResponse
from mlserver import MLModel
from mlserver.handlers import custom_handler
from mlserver.settings import ModelSettings
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.codecs import StringCodec
from .registry import ADAPTERS
from .metrics import REQS, TOKS, LAT
from .capabilities import CapLLM
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def init_tracing(service_name: str):
    res = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=res)
    exporter = OTLPSpanExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT","http://otel-collector:4317"), insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

class ConcurrencyManager:
    def __init__(self, limits: Dict[str, int], timeout_s: int):
        self._sems = {k: asyncio.Semaphore(v) for k, v in limits.items()}
        self._timeout = timeout_s
    async def run(self, mode: str, coro):
        async with self._sems.get(mode, asyncio.Semaphore(1000)):
            return await asyncio.wait_for(coro, timeout=self._timeout)

class LLMUnifiedRuntime(MLModel):
    def __init__(self, settings: ModelSettings):
        super().__init__(settings)
        p = dict(settings.parameters or {})
        self._framework = p.get("framework")
        self._model_uri = p.get("model_uri")
        self._params = p
        self._adapter: Optional[CapLLM] = None
        self._conc = ConcurrencyManager(p.get("per_mode_limits", {}), p.get("timeout_seconds", 60))
        if os.getenv("PROM_PORT"):
            from prometheus_client import start_http_server
            start_http_server(int(os.getenv("PROM_PORT")))
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            init_tracing(os.getenv("OTEL_SERVICE_NAME", "llm-server"))

    async def load(self) -> bool:
        from .adapters import __init__ as _
        self._adapter = ADAPTERS.create(self._framework, model_uri=self._model_uri, **self._params)
        await self._adapter.load()
        self._register_mode_handlers()
        return True

    def _register_mode_handlers(self):
        for mode in self._adapter.supported_modes():  # type: ignore
            setattr(self.__class__, mode, self._make_handler(mode))
            setattr(self.__class__, f"stream_{mode}", self._make_stream_handler(mode))

    def _make_handler(self, mode: str):
        @custom_handler(rest_path=f"/{mode}", rest_method="POST")
        async def handler(self: "LLMUnifiedRuntime", *, body: str = Depends(lambda req: req.body())) -> Response:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"llm.{mode}"):
                res = await self._predict(body.decode(), {"mode": mode}, mode)
                return Response(content=str(getattr(res, "outputs", res)), media_type="text/plain")
        return handler

    def _make_stream_handler(self, mode: str):
        @custom_handler(rest_path=f"/stream/{mode}", rest_method="POST")
        async def handler(self: "LLMUnifiedRuntime", *, body: str = Depends(lambda req: req.body())):
            tracer = trace.get_tracer(__name__)
            async def gen():
                with tracer.start_as_current_span(f"llm.stream_{mode}"):
                    async for r in self._predict_stream(body.decode(), {"mode": mode}, mode):
                        yield f"data: {getattr(r, 'outputs', r)}\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return handler

    async def _predict(self, prompt: str, params: Dict[str, Any], endpoint: str) -> InferenceResponse:
        mode = params.get("mode")
        model = getattr(self._settings, "name", "model")
        fw = self._framework
        REQS.labels(model=model, mode=mode, framework=fw, endpoint=endpoint).inc()
        with LAT.labels(model=model, mode=mode, framework=fw, endpoint=endpoint).time():
            out = await self._conc.run(mode, self._adapter.run(prompt, params))  # type: ignore
            TOKS.labels(model=model, mode=mode, framework=fw, endpoint=endpoint).inc(len(out.split()))
            return await self.encode(InferenceResponse, out, default_codec=StringCodec)

    async def _predict_stream(self, prompt: str, params: Dict[str, Any], endpoint: str):
        mode = params.get("mode")
        model = getattr(self._settings, "name", "model")
        fw = self._framework
        REQS.labels(model=model, mode=mode, framework=fw, endpoint=endpoint).inc()
        with LAT.labels(model=model, mode=mode, framework=fw, endpoint=endpoint).time():
            async for token in self._adapter.stream(prompt, params):  # type: ignore
                TOKS.labels(model=model, mode=mode, framework=fw, endpoint=endpoint).inc(len(token.split()))
                yield await self.encode(InferenceResponse, token, default_codec=StringCodec)
