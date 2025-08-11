"""Gateway sidecar: HTTP SSE that bridges to MLServer gRPC streaming."""
from __future__ import annotations
import os, json, time, asyncio, grpc
from typing import AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import start_http_server, Counter, Histogram
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.grpc import GrpcAioInstrumentorClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# gRPC stubs are provided by mlserver_grpc_protos package
from mlserver_grpc_protos import inference_pb2, inference_pb2_grpc

GRPC_HOST = os.getenv("GATEWAY_GRPC_HOST", "mlserver")
GRPC_PORT = int(os.getenv("GATEWAY_GRPC_PORT", "8081"))
PROM_PORT = int(os.getenv("PROM_PORT", "9091"))
HB_SEC = int(os.getenv("HEARTBEAT_SEC", "10"))

REQS = Counter("gateway_requests_total", "Requests", ["model"])
TOK  = Counter("gateway_tokens_total", "Tokens", ["model"])
LAT  = Histogram("gateway_latency_seconds", "Latency", ["model"])
start_http_server(PROM_PORT)

provider = TracerProvider(resource=Resource.create({"service.name": "gateway"}))
provider.add_span_processor(BatchSpanProcessor(
  OTLPSpanExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"), insecure=True)
))

trace.set_tracer_provider(provider)
GrpcAioInstrumentorClient().instrument()

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

def _req(model: str, payload: dict | str):
    data = json.dumps(payload).encode() if isinstance(payload, dict) else str(payload).encode()
    return inference_pb2.ModelInferRequest(
        model_name=model,
        inputs=[inference_pb2.ModelInferRequest_InferInputTensor(
            name="input", datatype="BYTES", shape=[1],
            contents=inference_pb2.InferTensorContents(bytes_contents=[data])
        )]
    )

@app.get("/v2/models/{model}/stream")
async def stream(model: str, request: Request, input: str, session_id: str | None = None, params: str | None = None):
    REQS.labels(model).inc()
    t0 = time.monotonic()
    try:
        call_params = json.loads(params) if params else {}
    except Exception as e:
        raise HTTPException(400, f"Invalid params JSON: {e}")
    payload = {"input": input}
    if session_id: payload["session_id"] = session_id
    if call_params: payload["params"] = call_params

    async def gen() -> AsyncGenerator[str, None]:
        async with grpc.aio.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}") as ch:
            stub = inference_pb2_grpc.GRPCInferenceServiceStub(ch)
            stream = stub.ModelStreamInfer(_req(model, payload))
            last = time.monotonic()
            while True:
                try:
                    resp = await asyncio.wait_for(stream.read(), timeout=1.0)
                except asyncio.TimeoutError:
                    if time.monotonic() - last > HB_SEC:
                        yield ": heartbeat\n\n"
                        last = time.monotonic()
                    if await request.is_disconnected(): break
                    continue
                if resp is None: break
                out = resp.outputs[0].contents.bytes_contents[0].decode("utf-8")
                TOK.labels(model).inc()
                yield f"data: {out}\n\n"
                last = time.monotonic()
                if await request.is_disconnected(): break

    try:
        return StreamingResponse(gen(), media_type="text/event-stream")
    finally:
        LAT.labels(model).observe(time.monotonic() - t0)
