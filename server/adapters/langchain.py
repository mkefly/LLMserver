from __future__ import annotations
from typing import AsyncIterator, Dict, Optional, Any
import importlib
from ..capabilities import CapLLM

def _import(path: str):
    if ":" in path: m, a = path.split(":", 1)
    else: p = path.split("."); m, a = ".".join(p[:-1]), p[-1]
    return getattr(importlib.import_module(m), a)

def _tok(x: Any) -> Optional[str]:
    if x is None: return None
    if isinstance(x, str): return x
    if isinstance(x, dict):
        for k in ("output","text","delta","content"):
            v = x.get(k);  if isinstance(v, str): return v
    try: return str(x)
    except: return None

class LangChainAdapter(CapLLM):
    """Adapter for LangChain Runnable/Chain via factory path."""
    def __init__(self, factory_path: str):
        self._factory = factory_path
        self._chain = None

    async def load(self) -> None:
        self._chain = _import(self._factory)()

    async def reload_from_uri(self, resolved_uri: str) -> None:
        return  # typically code-based NOP

    async def chat(self, text: str, *, session_id: Optional[str], params: Dict) -> str:
        payload = {"input": text}
        chain = self._chain
        if hasattr(chain, "ainvoke"): res = await chain.ainvoke(payload)
        else:
            import asyncio; loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, lambda: chain.invoke(payload))
        if isinstance(res, dict):
            for k in ("output","text","content"):
                if isinstance(res.get(k), str): return res[k]
        return str(res)

    async def query(self, text: str, *, session_id: Optional[str], params: Dict) -> str:
        return await self.chat(text, session_id=session_id, params=params)

    async def retrieve(self, text: str, *, session_id: Optional[str], params: Dict) -> str:
        return await self.chat(text, session_id=session_id, params=params)

    async def stream_chat(self, text: str, *, session_id: Optional[str], params: Dict) -> AsyncIterator[str]:
        chain = self._chain
        if hasattr(chain, "astream"):
            async for ch in chain.astream({"input": text}):
                t = _tok(ch);  if t: yield t
            return
        if hasattr(chain, "astream_events"):
            async for ev in chain.astream_events({"input": text}, version="v1"):
                if ev.get("event") == "on_llm_stream":
                    t = str(ev.get("data", {}).get("chunk", "") or "")
                    if t: yield t
            return
        yield await self.chat(text, session_id=session_id, params=params)

    async def stream_query(self, text: str, *, session_id: Optional[str], params: Dict) -> AsyncIterator[str]:
        async for t in self.stream_chat(text, session_id=session_id, params=params):
            yield t
