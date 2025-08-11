from typing import AsyncIterator, Dict, Optional
import mlflow

class LlamaIndexStrategy:
    """Native LlamaIndex (not pyfunc). UC hot-reload supported."""
    def __init__(self, engine_type: str, top_k: int):
        self.engine_type = engine_type
        self.top_k = top_k
        self._index = None

    async def load_from_stage(self, uc_stage_uri: str):
        self._index = mlflow.llama_index.load_model(uc_stage_uri)

    async def reload_from_version(self, resolved_uri: str):
        self._index = mlflow.llama_index.load_model(resolved_uri)

    # helpers
    def _chat(self, **kw): return self._index.as_chat_engine(**kw)
    def _query(self, **kw): return self._index.as_query_engine(**kw)
    def _retriever(self, **kw): return self._index.as_retriever(**kw)

    async def predict_text(self, text: str, session_id: Optional[str], params: Dict) -> str:
        kw = {"similarity_top_k": self.top_k, **params}
        if self.engine_type == "chat":
            r = await self._chat(**kw).achat(text)
            return getattr(r, "message", r).content if hasattr(r, "message") else str(r)
        if self.engine_type == "query":
            r = await self._query(**kw).aquery(text)
            return str(getattr(r, "response", r))
        nodes = await self._retriever(**kw).aretrieve(text)
        return "\n".join(n.get_content() for n in nodes)

    async def stream_tokens(self, text: str, session_id: Optional[str], params: Dict) -> AsyncIterator[str]:
        kw = {"similarity_top_k": self.top_k, **params}
        if self.engine_type == "chat":
            async for ch in self._chat(**kw).astream_chat(text):
                yield getattr(ch, "delta", None) or getattr(ch, "text", None) or str(ch)
            return
        if self.engine_type == "query":
            q = self._query(**kw)
            try:
                async for ch in q.astream_query(text):
                    yield getattr(ch, "delta", None) or getattr(ch, "text", None) or str(ch)
            except AttributeError:
                r = await q.aquery(text); yield str(getattr(r, "response", r))
            return
        out = await self.predict_text(text, session_id, params)
        yield out
