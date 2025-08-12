from __future__ import annotations
from typing import Dict
from ..registry import ADAPTERS
from .base_mlflow_adapter import BaseMLflowAdapter
from .common import (
    extract_mode,
    pick,
    call_maybe_async,
    stringify,
)

@ADAPTERS.register("llamaindex")
def llamaindex_factory(model_uri: str, **_: dict) -> BaseMLflowAdapter:
    return LlamaIndexAdapter(model_uri)

class LlamaIndexAdapter(BaseMLflowAdapter):
    """
    MLflow-backed LlamaIndex adapter.

    Modes:
      - chat
      - query
      - retrieve (special handling: joins node contents)

    METHOD_MAP:
      - Each mode lists (run_methods, stream_methods)
      - retrieve mode has no streaming.
    """
    FRAMEWORK_NAME = "llama_index"
    SUPPORTED_MODES = ("chat", "query", "retrieve")
    METHOD_MAP = {
        "chat":     (("achat", "chat"), ("astream_chat",)),
        "query":    (("aquery", "query"), ("astream_query",)),
        "retrieve": (("aretrieve", "retrieve"), None),
    }

    async def run(self, prompt: str, params: Dict) -> str:
        """
        Runs the selected mode. For 'retrieve', joins node contents into one string.
        """
        mode = extract_mode(params, default=self.SUPPORTED_MODES[0], allowed=self.SUPPORTED_MODES)
        run_methods, _ = self.METHOD_MAP[mode]
        fn = pick(self._obj, run_methods)
        if not fn:
            raise TypeError(f"{self.FRAMEWORK_NAME} object must support one of: {run_methods}")

        result = await call_maybe_async(fn, prompt, **params)

        # Special handling for retrieval results
        if mode == "retrieve" and isinstance(result, list):
            return "\n".join(
                str(getattr(node, "get_content", lambda: str(node))())
                for node in result
            )
        return stringify(result)
