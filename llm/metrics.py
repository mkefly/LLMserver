from prometheus_client import Counter, Histogram

REQS = Counter("llm_requests_total", "Total LLM requests", ["model", "mode", "framework", "endpoint"])
TOKS = Counter("llm_tokens_total", "Total tokens generated", ["model", "mode", "framework", "endpoint"])
LAT = Histogram("llm_latency_seconds", "LLM request latency", ["model", "mode", "framework", "endpoint"])
