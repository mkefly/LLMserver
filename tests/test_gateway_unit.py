import asyncio, pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    with patch("gateway.app.grpc") as grpc_mod:
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
        from gateway.app import app
        yield TestClient(app)

def test_stream_sends_tokens(client):
    r = client.get("/v2/models/m/stream", params={"input":"hi"})
    assert r.status_code == 200
    body = r.text.replace("data: ", "").replace("\n\n", "")
    assert body == "ABC"

def test_bad_params(client):
    r = client.get("/v2/models/m/stream", params={"input":"hi","params":"{bad"})
    assert r.status_code == 400
