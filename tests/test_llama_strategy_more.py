import pytest
from unittest.mock import MagicMock, patch
from server.flavors.llama_index import LlamaIndexStrategy

@pytest.mark.asyncio
async def test_llama_query_and_retriever():
    fake_q = MagicMock()
    async def aquery(q): return MagicMock(response=f"R:{q}")
    async def astream_query(q):
        for t in ["1","2"]: yield MagicMock(delta=t)
    fake_q.aquery.side_effect = aquery
    fake_q.astream_query.side_effect = astream_query

    fake_r = MagicMock()
    async def aretrieve(q):
        m = MagicMock(); m.get_content.return_value = "C"
        return [m, m]
    fake_r.aretrieve.side_effect = aretrieve

    fake_index = MagicMock()
    fake_index.as_query_engine.return_value = fake_q
    fake_index.as_retriever.return_value = fake_r

    with patch("server.flavors.llama_index.mlflow") as M:
        M.llama_index.load_model.return_value = fake_index
        s = LlamaIndexStrategy(engine_type="query", top_k=3)
        await s.load_from_stage("models:/a.b.c/Prod")
        # streaming query
        toks = []
        async for t in s.stream_tokens("Q", None, {}):
            toks.append(t)
        assert "".join(toks) == "12"
        # non-stream retriever
        s.engine_type = "retriever"
        out = await s.predict_text("X", None, {})
        assert out == "C\nC"
