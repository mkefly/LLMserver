from __future__ import annotations
import asyncio
from typing import Awaitable, Callable, Optional, Protocol, runtime_checkable
from mlflow.tracking import MlflowClient

@runtime_checkable
class ModelResolver(Protocol):
    async def resolve_initial(self, model_ref: str) -> str: ...
    async def watch(self, model_ref: str, interval_s: int,
                    on_change: Callable[[str], Awaitable[None]]) -> None: ...

class UCAliasResolver:
    """
    Resolves models:/CAT.SCH.NAME@alias to concrete models:/.../<version>.
    Works with MLflow/Unity Catalog model **version aliases**.
    """
    def __init__(self):
        self._client = MlflowClient()

    async def resolve_initial(self, model_ref: str) -> str:
        name, alias = _parse_uc_alias(model_ref)
        mv = self._client.get_model_version_by_alias(name, alias)
        return f"models:/{name}/{mv.version}"

    async def watch(self, model_ref: str, interval_s: int,
                    on_change: Callable[[str], Awaitable[None]]) -> None:
        name, alias = _parse_uc_alias(model_ref)
        current: Optional[str] = None
        while True:
            try:
                mv = self._client.get_model_version_by_alias(name, alias)
                resolved = f"models:/{name}/{mv.version}"
                if resolved != current:
                    await on_change(resolved)
                    current = resolved
            except Exception:
                # swallow & retry
                pass
            await asyncio.sleep(interval_s)

def _parse_uc_alias(uri: str) -> tuple[str, str]:
    """
    Accepts: models:/catalog.schema.name@alias
    Returns: (catalog.schema.name, alias)
    """
    if not uri.startswith("models:/") or "@" not in uri:
        raise ValueError("Expected UC alias ref like 'models:/catalog.schema.name@alias'")
    name, alias = uri[len("models:/"):].split("@", 1)
    if not name or not alias:
        raise ValueError("Invalid UC alias ref; missing name or alias")
    return name, alias

class StaticURIResolver:
    """Returns the given URI as-is and does not watch for changes."""
    async def resolve_initial(self, model_ref: str) -> str:
        return model_ref
    async def watch(self, model_ref: str, interval_s: int, on_change: Callable[[str], Awaitable[None]]) -> None:
        return
