import os
from server.observability import init_prom, init_tracing, tracer

def test_init_prometheus_enables_server(monkeypatch):
    monkeypatch.setenv("PROM_PORT", "0")  # no-op
    init_prom()
    monkeypatch.setenv("PROM_PORT", "9100")  # starts server
    init_prom()

def test_init_tracing_sets_provider():
    init_tracing("mlserver-test")
    s = tracer.start_span("unit-span")
    s.end()
