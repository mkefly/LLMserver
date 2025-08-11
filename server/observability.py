"""Prometheus metrics and OpenTelemetry tracing init."""
from __future__ import annotations
import os
from prometheus_client import Counter, Histogram, start_http_server
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

REQS = Counter("llm_requests_total", "Requests", ["model", "method"])
TOKENS = Counter("llm_tokens_total", "Tokens", ["model"])
LAT = Histogram("llm_latency_seconds", "Latency", ["model", "method"])

def init_prom(env: str = "PROM_PORT") -> None:
    port = int(os.getenv(env, "0") or "0")
    if port:
        start_http_server(port)

def init_tracing(service: str = "mlserver") -> None:
    res = Resource.create({"service.name": service})
    provider = TracerProvider(resource=res)
    exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"),
        insecure=True,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)
