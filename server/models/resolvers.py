"""Model resolvers: resolve a model reference, optionally watch for changes."""
from __future__ import annotations
import asyncio
from typing import Awaitable, Callable, Optional, Protocol, runtime_checkable
from mlflow.tracking import MlflowClient

@runtime_checkable
class ModelResolver(Protocol):
    async def resolve_initial(self, model_ref: str) -> str: ...
    async def watch(self, model_ref: str, interval_s: int, on_change: Callable[[str], Awaitable[None]]) -> None: ...

class UCStageResolver:
    """Resolves models:/CAT.SCH.NAME/Stage to concrete models:/.../<version>."""
    def __init__(self):
        self._client = MlflowClient()

    async def resolve_initial(self, model_ref: str) -> str:
        name, stage = _parse_uc(model_ref)
        v = self._client.get_latest_versions(name, stages=[stage])[0].version
        return f"models:/{name}/{v}"

    async def watch(self, model_ref: str, interval_s: int, on_change: Callable[[str], Awaitable[None]]) -> None:
        name, stage = _parse_uc(model_ref)
        current: Optional[str] = None
        while True:
            try:
                v = self._client.get_latest_versions(name, stages=[stage])[0].version
                resolved = f"models:/{name}/{v}"
                if resolved != current:
                    await on_change(resolved)
                    current = resolved
            except Exception:
                pass
            await asyncio.sleep(interval_s)

class StaticURIResolver:
    """Returns the given URI as-is and does not watch for changes."""
    async def resolve_initial(self, model_ref: str) -> str:
        return model_ref
    async def watch(self, model_ref: str, interval_s: int, on_change: Callable[[str], Awaitable[None]]) -> None:
        return

def _parse_uc(uri: str) -> tuple[str, str]:
    assert uri.startswith("models:/")
    name, stage = uri[len("models:/"):].rsplit("/", 1)
    return name, stage
