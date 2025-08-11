import pytest, asyncio, os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

@pytest.fixture
def app_client_timeout_heartbeat(monkeypatch):
    # Make heartbeats frequent and timeout low
    monkeypatch.setenv("HEARTBEAT_SEC", "1")
    with patch("gateway.app.grpc") as grpc_mod:
        class FakeStream:
            def __init__(self):
                self.count = 0
            async def read(self):
                await asyncio.sleep(0)  # yield control
                self.count += 1
                # Force the wait_for timeout to trigger by sleeping >1s twice
                if self.count <= 2:
                    await asyncio.sleep(1.1)
                return None
        stub = MagicMock()
        stub.ModelStreamInfer.return_value = FakeStream()
        grpc_mod.aio.insecure_channel.return_value.__aenter__.return_value = MagicMock()
        grpc_mod.inference_pb2 = MagicMock()
        grpc_mod.inference_pb2_grpc = MagicMock(GRPCInferenceServiceStub=MagicMock(return_value=stub))
        from gateway.app import app
        yield TestClient(app)

def test_gateway_bad_params():
    from gateway.app import app
    c = TestClient(app)
    r = c.get("/v2/models/m/stream", params={"input":"hi", "params":"{bad"})
    assert r.status_code == 400

def test_gateway_heartbeat(app_client_timeout_heartbeat):
    r = app_client_timeout_heartbeat.get("/v2/models/m/stream", params={"input":"x"}, timeout=5)
    assert r.status_code == 200
    assert "data:" in r.text or ": heartbeat" in r.text
