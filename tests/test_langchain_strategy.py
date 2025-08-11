import pytest
from server.flavors.langchain import LangChainStrategy

class FakeChain:
    async def ainvoke(self, payload): return {"text": "OK-"+payload["input"]}
    async def astream(self, payload):
        for t in ["x","y","z"]:
            yield t

class FakeEvents:
    async def astream_events(self, payload, version="v1"):
        for t in ["a","b"]:
            yield {"event":"on_llm_stream","data":{"chunk":t}}

@pytest.mark.asyncio
async def test_langchain_predict_and_stream(monkeypatch):
    def factory(): return FakeChain()
    monkeypatch.setattr("server.flavors.langchain._import", lambda _: factory)
    s = LangChainStrategy("pkg.mod:build", True)
    await s.load()
    out = await s.predict_text("hi", None, {})
    assert out == "OK-hi"
    toks = []
    async for t in s.stream_tokens("foo", None, {}):
        toks.append(t)
    assert "".join(toks) == "xyz"

@pytest.mark.asyncio
async def test_langchain_stream_events(monkeypatch):
    def factory(): return FakeEvents()
    monkeypatch.setattr("server.flavors.langchain._import", lambda _: factory)
    s = LangChainStrategy("pkg.mod:build", False)
    await s.load()
    toks = []
    async for t in s.stream_tokens("bar", None, {}):
        toks.append(t)
    assert "".join(toks) == "ab"
