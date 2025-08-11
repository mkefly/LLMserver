from __future__ import annotations
from ..registry import register_adapter
from .llama_index import LlamaIndexAdapter
from .langchain import LangChainAdapter

register_adapter("llama_index", lambda resolved_uri, top_k, **_: LlamaIndexAdapter(resolved_uri, top_k))
register_adapter("langchain",   lambda factory_path, **_: LangChainAdapter(factory_path))
