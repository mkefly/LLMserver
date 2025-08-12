from __future__ import annotations
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Sequence, Union
import asyncio

async def call_maybe_async(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Call a function that may be sync or async.
    Runs sync functions in a thread executor to avoid blocking the event loop.
    """
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

async def aiter_maybe_async(genlike) -> AsyncIterator[Any]:
    """
    Yield items from an async iterator or sync iterable, transparently.
    Converts sync iterables into async generators for consistent streaming.
    """
    if hasattr(genlike, "__anext__"):  # async generator
        async for x in genlike:
            yield x
        return
    for x in list(genlike):  # sync iterable
        yield x

def pick(obj: Any, names: Sequence[str]) -> Optional[Callable[..., Any]]:
    """
    Return the first callable attribute found on an object from a list of names.
    Example:
        pick(chain, ("ainvoke", "invoke"))
    """
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            return fn
    return None

def stringify(res: Any) -> str:
    """
    Convert common result types to a string.
    Checks common keys in dicts ('output', 'text', etc.).
    Fallback is str(res).
    """
    if res is None:
        return ""
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        for k in ("output", "text", "content", "message", "response"):
            v = res.get(k)
            if isinstance(v, str):
                return v
    return str(res)

def token_of(x: Any) -> Optional[str]:
    """
    Extract a string token from common chunk/delta formats.
    Used in streaming responses.
    """
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("delta", "text", "token", "content", "chunk"):
            v = x.get(k)
            if isinstance(v, str):
                return v
    return str(x)

def extract_mode(params: Dict, *, default: str, allowed: Iterable[str]) -> str:
    """
    Read and validate 'mode' from params.
    Raises ValueError if mode not in allowed.
    """
    mode = (params or {}).get("mode", default)
    if mode not in allowed:
        raise ValueError(f"Unsupported mode '{mode}'. Allowed: {', '.join(allowed)}")
    return mode
