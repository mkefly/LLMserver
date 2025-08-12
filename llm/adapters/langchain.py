from __future__ import annotations
from typing import Dict
from ..registry import ADAPTERS
from .base_mlflow_adapter import BaseMLflowAdapter
from .common import (
    aiter_maybe_async,
    token_of,
    pick,
    extract_mode,
    call_maybe_async,
    stringify,
)

@ADAPTERS.register("langchain")
def langchain_factory(model_uri: str, **_: dict) -> BaseMLflowAdapter:
    return LangChainAdapter(model_uri)

class LangChainAdapter(BaseMLflowAdapter):
    """
    MLflow-backed LangChain adapter.

    Modes:
      - chat
      - query

    METHOD_MAP tells the base class which underlying methods to try for each mode:
      - run methods:    ("ainvoke", "invoke")
      - stream methods: ("astream",)  (optional; falls back to run() if missing)
    """
    FRAMEWORK_NAME = "langchain"
    SUPPORTED_MODES = ("chat", "query")
    METHOD_MAP = {
        "chat":  (("ainvoke", "invoke"), ("astream",)),
        "query": (("ainvoke", "invoke"), ("astream",)),
    }

    async def run(self, prompt: str, params: Dict) -> str:
        """
        Wraps the prompt as {'input': prompt} for LangChain run calls.
        Validates 'mode' and selects the first available run method.
        """
        mode = extract_mode(params, default=self.SUPPORTED_MODES[0], allowed=self.SUPPORTED_MODES)
        run_methods, _ = self.METHOD_MAP[mode]
        fn = pick(self._obj, run_methods)
        if not fn:
            raise TypeError(f"{self.FRAMEWORK_NAME} object must support one of: {run_methods}")
        res = await call_maybe_async(fn, {"input": prompt})
        return stringify(res)

    def _stream_args(self, prompt: str, params: Dict):
        """
        LangChain expects {"input": prompt} as the first argument.
        Ignores any other params here.
        """
        return ({"input": prompt},), {}
