import pytest
from unittest.mock import MagicMock, patch
from server.adapters.llama_index import LlamaIndexAdapter

@pytest.mark.asyncio
async def test_llama_chat_and_stream():
    fake_chat = MagicMock()
    async def achat(msg): return MagicMock(message=MagicMock(content=f"echo:{msg}"))
    async def astream_chat(msg):
        for t in ["a","b","c"]: yield MagicMock(delta=t)
    fake_chat.achat.side_effect = achat
    fake_chat.astream_chat.side_effect = astream_chat

    fake_index = MagicMock()
    fake_index.as_chat_engine.return_value = fake_chat

    with patch("server.adapters.llama_index.mlflow") as M:
        M.llama_index.load_model.return_value = fake_index
        a = LlamaIndexAdapter(resolved_uri="models:/c.s.m/1", top_k=4)
        await a.load()
        out = await a.chat("hi", session_id=None, params={})
        assert out == "echo:hi"
        toks = []
        async for t in a.stream_chat("x", session_id=None, params={}):
            toks.append(t)
        assert "".join(toks) == "abc"
