import pytest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock, MagicMock
from server.runtime import LLMUnifiedRuntime

class FakeReq(SimpleNamespace): pass

@pytest.mark.asyncio
async def test_runtime_predict(monkeypatch):
    rt = LLMUnifiedRuntime()
    rt._settings = SimpleNamespace(parameters=SimpleNamespace(dict=lambda: {
        "adapter":"langchain","resolver":"static_uri","model_ref":"pkg:factory"
    }))
    fake = MagicMock()
    fake.load = AsyncMock()
    fake.chat = AsyncMock(return_value="OK")
    with patch("server.runtime.get_resolver") as R,          patch("server.runtime.get_adapter") as A,          patch("server.runtime.decode_input") as dec:
        R.return_value.return_value.resolve_initial = AsyncMock(return_value="pkg:factory")
        A.return_value.return_value = fake
        dec.return_value = ("hello", "sid", {})
        await rt.load()
        resp = await rt.predict(FakeReq())
    assert resp.outputs[0].data == ["OK"]

@pytest.mark.asyncio
async def test_runtime_predict_stream_timeout(monkeypatch):
    rt = LLMUnifiedRuntime()
    rt._settings = SimpleNamespace(parameters=SimpleNamespace(dict=lambda: {
        "adapter":"langchain","resolver":"static_uri","model_ref":"pkg:factory","timeout_s":1
    }))
    fake = MagicMock()
    async def slow():
        import asyncio
        await asyncio.sleep(2)
        yield "never"
    fake.load = AsyncMock()
    fake.stream_chat = slow
    with patch("server.runtime.get_resolver") as R,          patch("server.runtime.get_adapter") as A,          patch("server.runtime.decode_input") as dec:
        R.return_value.return_value.resolve_initial = AsyncMock(return_value="pkg:factory")
        A.return_value.return_value = fake
        dec.return_value = ("hi", None, {})
        await rt.load()
        outs = []
        async for ch in rt.predict_stream(FakeReq()):
            outs.append(ch.outputs[0].data[0])
        assert outs[-1] == "[stream-ended]"
