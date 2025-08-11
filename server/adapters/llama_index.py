from __future__ import annotations
from typing import AsyncIterator, Dict, Optional
import mlflow
from ..capabilities import CapLLM

class LlamaIndexAdapter(CapLLM):
    """Adapter for native LlamaIndex saved in MLflow (llama_index flavor)."""
    def __init__(self, resolved_uri: str, top_k: int):
        self._uri = resolved_uri
        self._top_k = top_k
        self._index = None

    async def load(self) -> None:
        self._index = mlflow.llama_index.load_model(self._uri)

    async def reload_from_uri(self, resolved_uri: str) -> None:
        self._uri = resolved_uri
        await self.load()

    def _chat(self, **kw): return self._index.as_chat_engine(**kw)
    def _query(self, **kw): return self._index.as_query_engine(**kw)
    def _retriever(self, **kw): return self._index.as_retriever(**kw)

    async def chat(self, text: str, *, session_id: Optional[str], params: Dict) -> str:
        r = await self._chat(similarity_top_k=self._top_k, **params).achat(text)
        return getattr(r, "message", r).content if hasattr(r, "message") else str(r)

    async def query(self, text: str, *, session_id: Optional[str], params: Dict) -> str:
        r = await self._query(similarity_top_k=self._top_k, **params).aquery(text)
        return str(getattr(r, "response", r))

    async def retrieve(self, text: str, *, session_id: Optional[str], params: Dict) -> str:
        nodes = await self._retriever(similarity_top_k=self._top_k, **params).aretrieve(text)
        return "\n".join(n.get_content() for n in nodes)

    async def stream_chat(self, text: str, *, session_id: Optional[str], params: Dict) -> AsyncIterator[str]:
        async for ch in self._chat(similarity_top_k=self._top_k, **params).astream_chat(text):
            yield getattr(ch, "delta", None) or getattr(ch, "text", None) or str(ch)

    async def stream_query(self, text: str, *, session_id: Optional[str], params: Dict) -> AsyncIterator[str]:
        q = self._query(similarity_top_k=self._top_k, **params)
        try:
            async for ch in q.astream_query(text):
                yield getattr(ch, "delta", None) or getattr(ch, "text", None) or str(ch)
        except AttributeError:
            yield await self.query(text, session_id=session_id, params=params)
