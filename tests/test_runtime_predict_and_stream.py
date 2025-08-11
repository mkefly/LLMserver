import pytest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock, MagicMock
from server.runtime import LLMUnifiedRuntime

class FakeReq(SimpleNamespace):
    pass

@pytest.mark.asyncio
async def test_runtime_predict(monkeypatch):
    rt = LLMUnifiedRuntime()
    rt._settings = SimpleNamespace(parameters=SimpleNamespace(dict=lambda: {
        "framework": "langchain", "langchain_factory": "x:y"
    }))
    fake_strategy = MagicMock()
    fake_strategy.predict_text = AsyncMock(return_value="OK")
    with patch("server.runtime.LangChainStrategy") as LC,          patch("server.runtime.decode_input") as dec:
        LC.return_value.load = AsyncMock()
        LC.return_value = fake_strategy
        dec.return_value = ("hello", "sid1", {"t":1})
        await rt.load()
        resp = await rt.predict(FakeReq())
    assert resp.outputs[0].data == ["OK"]

@pytest.mark.asyncio
async def test_runtime_predict_stream(monkeypatch):
    rt = LLMUnifiedRuntime()
    rt._settings = SimpleNamespace(parameters=SimpleNamespace(dict=lambda: {
        "framework": "langchain", "langchain_factory": "x:y"
    }))
    fake_strategy = MagicMock()
    async def gen():
        for t in ["A","B","C"]: yield t
    fake_strategy.stream_tokens = gen
    with patch("server.runtime.LangChainStrategy") as LC,          patch("server.runtime.decode_input") as dec:
        LC.return_value.load = AsyncMock()
        LC.return_value = fake_strategy
        dec.return_value = ("hi", None, {})
        await rt.load()
        chunks = []
        async for c in rt.predict_stream(FakeReq()):
            chunks.append(c.outputs[0].data[0])
        assert "".join(chunks) == "ABC"
