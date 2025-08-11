import json
import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

@pytest.fixture
def client(monkeypatch):
    # Make heartbeat frequent so tests are fast if needed
    monkeypatch.setenv("HEARTBEAT_SEC", "1")
    with patch("gateway.app.grpc") as grpc_mod:
        # Fake stream: three responses with tokens "A","B","C"
        class FakeResp:
            def __init__(self, s): 
                self.outputs = [MagicMock(contents=MagicMock(bytes_contents=[s.encode()]))]
        class FakeStream:
            def __init__(self): self._it = iter([FakeResp("A"), FakeResp("B"), FakeResp("C"), None])
            async def read(self):
                try: return next(self._it)
                except StopIteration: return None
        stub = MagicMock()
        stub.ModelStreamInfer.return_value = FakeStream()
        grpc_mod.aio.insecure_channel.return_value.__aenter__.return_value = MagicMock()
        grpc_mod.inference_pb2 = MagicMock()
        grpc_mod.inference_pb2_grpc = MagicMock(GRPCInferenceServiceStub=MagicMock(return_value=stub))
        from gateway.app import app   # import now
        yield TestClient(app)

def test_sse_stream(client):
    r = client.get("/v2/models/m/stream", params={"input":"hi"})
    assert r.status_code == 200
    body = r.text.replace("data: ", "").replace("\n\n", "")
    # token order preserved
    assert body == "ABC"
