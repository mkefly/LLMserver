from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple
import mlflow
from ..capabilities import CapLLM
from .common import pick, call_maybe_async, aiter_maybe_async, extract_mode, stringify, token_of

class BaseMLflowAdapter(CapLLM):
    """
    A reusable base adapter for any MLflow-saved LLM (LangChain, LlamaIndex, CrewAI, etc.).

    This class removes repetitive code:
    - It knows how to load a model from MLflow.
    - It runs or streams text prompts based on the mode.
    - It picks the right method automatically (async or sync).

    Subclasses only need to set:
        FRAMEWORK_NAME (str)
            The MLflow flavor name, e.g. "llama_index", "langchain", "crewai".
            Used to call `mlflow.<framework>.load_model(...)`.

        SUPPORTED_MODES (tuple of str)
            Which high-level modes this adapter understands, e.g.:
            ("chat", "query", "retrieve")

        METHOD_MAP (dict)
            Maps each mode to:
                (run_method_names, stream_method_names_or_None)

            - run_method_names: names to try for running a single prompt.
            - stream_method_names_or_None: names to try for streaming output,
              or None if streaming is not supported for that mode.

            Example:
                METHOD_MAP = {
                    "chat":     (("achat", "chat"), ("astream_chat",)),
                    "query":    (("aquery", "query"), ("astream_query",)),
                    "retrieve": (("aretrieve", "retrieve"), None),
                }

    How it works:
    -------------
    When you call `await adapter.run("Hello", {"mode": "query"})`:
        1. We check that "query" is in SUPPORTED_MODES.
        2. We look up METHOD_MAP["query"] → (("aquery", "query"), ("astream_query",))
        3. We try each name in ("aquery", "query") until we find a callable on the model.
        4. We call it, handling both sync and async functions.

    When you call:
        async for tok in adapter.stream("Hello", {"mode": "chat"}):
            print(tok)

        - We look up ("astream_chat",) and call it if found.
        - If not found, we fall back to calling run() once.

    Example:
    --------
    class LlamaIndexAdapter(BaseMLflowAdapter):
        FRAMEWORK_NAME = "llama_index"
        SUPPORTED_MODES = ("chat", "query", "retrieve")
        METHOD_MAP = {
            "chat":     (("achat", "chat"), ("astream_chat",)),
            "query":    (("aquery", "query"), ("astream_query",)),
            "retrieve": (("aretrieve", "retrieve"), None),
        }

    adapter = LlamaIndexAdapter("models:/my-llamaindex/1")
    await adapter.load()

    # Single response
    res = await adapter.run("What is AI?", {"mode": "query"})
    print(res)

    # Streaming
    async for token in adapter.stream("Hello!", {"mode": "chat"}):
        print(token)
    """

    FRAMEWORK_NAME: str = ""
    SUPPORTED_MODES: Tuple[str, ...] = ()
    METHOD_MAP: Dict[str, Tuple[Sequence[str], Optional[Sequence[str]]]] = {}

    def __init__(self, uri: str):
        self._uri = uri
        self._obj: Any = None
        self._close_fn: Optional[Any] = None

    def supported_modes(self) -> List[str]:
        return list(self.SUPPORTED_MODES)

    async def load(self) -> None:
        """
        Load the model from MLflow.<framework>.load_model(uri).
        """
        if not self.FRAMEWORK_NAME:
            raise ValueError("FRAMEWORK_NAME must be defined in subclass.")
        framework_mod = getattr(mlflow, self.FRAMEWORK_NAME, None)
        if not framework_mod or not hasattr(framework_mod, "load_model"):
            raise RuntimeError(f"mlflow.{self.FRAMEWORK_NAME}.load_model is not available.")
        self._obj = framework_mod.load_model(self._uri)
        self._close_fn = getattr(self._obj, "close", None)

    async def reload_from_uri(self, uri: str) -> None:
        self._uri = uri
        await self.load()

    async def run(self, prompt: str, params: Dict) -> str:
        """
        Run the model for a given mode.
        """
        mode = extract_mode(params, default=self.SUPPORTED_MODES[0], allowed=self.SUPPORTED_MODES)
        run_methods, _ = self.METHOD_MAP.get(mode, ((), None))
        fn = pick(self._obj, run_methods)
        if not fn:
            raise TypeError(f"{self.FRAMEWORK_NAME} object must support {run_methods}")
        res = await call_maybe_async(fn, prompt, **params)
        return stringify(res)

    async def stream(self, prompt: str, params: Dict) -> AsyncIterator[str]:
        """
        Stream tokens for the given mode if supported, else fall back to run().
        Subclasses may override `_stream_args()` to change how prompt/params
        are passed to the streaming method.
        """
        mode = extract_mode(params, default=self.SUPPORTED_MODES[0], allowed=self.SUPPORTED_MODES)
        _, stream_methods = self.METHOD_MAP[mode]

        if not stream_methods:
            yield await self.run(prompt, params)
            return

        stream_fn = pick(self._obj, stream_methods)
        if not stream_fn:
            yield await self.run(prompt, params)
            return

        args, kwargs = self._stream_args(prompt, params)
        async for chunk in aiter_maybe_async(stream_fn(*args, **kwargs)):
            if (tok := token_of(chunk)):
                yield tok

    def _stream_args(self, prompt: str, params: Dict):
        """
        Default: pass prompt + params as a single dict positional arg.
        Most frameworks accept {"prompt": ..., ...} as the first arg.
        """
        return ({"prompt": prompt, **params},), {}
        
    async def close(self) -> None:
        if self._close_fn:
            await call_maybe_async(self._close_fn)
