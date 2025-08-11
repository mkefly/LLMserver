from typing import AsyncIterator, Dict, Optional, Any
import importlib

def _import(path: str):
    if ":" in path: m, a = path.split(":", 1)
    else: p = path.split("."); m, a = ".".join(p[:-1]), p[-1]
    return getattr(importlib.import_module(m), a)

def _tok(x: Any) -> Optional[str]:
    if x is None: return None
    if isinstance(x, str): return x
    if isinstance(x, dict):
        for k in ("output","text","delta","content"):
            v = x.get(k)
            if isinstance(v, str): return v
    try: return str(x)
    except: return None

class LangChainStrategy:
    def __init__(self, factory_path: str, use_history: bool):
        self.factory_path = factory_path
        self.use_history = use_history
        self.chain = None

    async def load(self):
        build = _import(self.factory_path)
        c = build()
        if self.use_history:
            try:
                from langchain_core.runnables.history import RunnableWithMessageHistory
                c = RunnableWithMessageHistory(c, lambda sid: [], input_messages_key="input")
            except Exception:
                pass
        self.chain = c

    async def reload_from_version(self, resolved_uri: str):
        return  # NOP for code-based LC; keep signature parity

    async def predict_text(self, text: str, session_id: Optional[str], params: Dict) -> str:
        payload = {"input": text}
        if hasattr(self.chain, "ainvoke"):
            res = await self.chain.ainvoke(payload)
        else:
            import asyncio; loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, lambda: self.chain.invoke(payload))
        if isinstance(res, dict):
            for k in ("output","text","content"):
                if isinstance(res.get(k), str): return res[k]
        return str(res)

    async def stream_tokens(self, text: str, session_id: Optional[str], params: Dict) -> AsyncIterator[str]:
        payload = {"input": text}
        if hasattr(self.chain, "astream"):
            async for ch in self.chain.astream(payload):
                t = _tok(ch);  if t: yield t
            return
        if hasattr(self.chain, "astream_events"):
            async for ev in self.chain.astream_events(payload, version="v1"):
                if ev.get("event") == "on_llm_stream":
                    t = str(ev.get("data", {}).get("chunk", "") or "")
                    if t: yield t
            return
        yield await self.predict_text(text, session_id, params)
