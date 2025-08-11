# LLM Runtime + Gateway - Test Suite
Drop-in server runtime, gateway, and a pytest suite with coverage configured.

## Run tests
```bash
pip install -r model/requirements.txt || true  # optional if you have deps already
pip install pytest pytest-asyncio pytest-cov fastapi uvicorn prometheus_client opentelemetry-sdk opentelemetry-exporter-otlp
pytest
```
