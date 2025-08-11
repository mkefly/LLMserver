import pytest
from server.adapters.langchain import LangChainAdapter

class FakeChain:
    async def ainvoke(self, payload): return {"text": "OK-"+payload["input"]}
    async def astream(self, payload):
        for t in ["x","y","z"]: yield t

@pytest.mark.asyncio
async def test_langchain_chat_and_stream(monkeypatch):
    def factory(): return FakeChain()
    monkeypatch.setattr("server.adapters.langchain._import", lambda _: factory)
    a = LangChainAdapter("pkg:factory")
    await a.load()
    out = await a.chat("hi", session_id=None, params={})
    assert out == "OK-hi"
    toks = []
    async for t in a.stream_chat("foo", session_id=None, params={}):
        toks.append(t)
    assert "".join(toks) == "xyz"
