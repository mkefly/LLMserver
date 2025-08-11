import pytest
from unittest.mock import MagicMock, patch
from server.flavors.llama_index import LlamaIndexStrategy

@pytest.mark.asyncio
async def test_llama_predict_and_stream():
    # mock mlflow.llama_index.load_model and engines
    fake_chat = MagicMock()
    async def achat(msg): return MagicMock(message=MagicMock(content=f"echo:{msg}"))
    async def astream_chat(msg):
        for t in ["a", "b", "c"]:
            yield MagicMock(delta=t)
    fake_chat.achat.side_effect = achat
    fake_chat.astream_chat.side_effect = astream_chat

    fake_index = MagicMock()
    fake_index.as_chat_engine.return_value = fake_chat

    with patch("server.flavors.llama_index.mlflow") as M:
        M.llama_index.load_model.return_value = fake_index
        s = LlamaIndexStrategy(engine_type="chat", top_k=4)
        await s.load_from_stage("models:/cat.sch.idx/Production")
        out = await s.predict_text("hi", None, {})
        assert out == "echo:hi"
        toks = []
        async for t in s.stream_tokens("hello", None, {}):
            toks.append(t)
        assert "".join(toks) == "abc"
