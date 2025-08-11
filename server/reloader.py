import asyncio, logging
from typing import Awaitable, Callable, Optional
from mlflow.tracking import MlflowClient
log = logging.getLogger(__name__)

def _parse_uc(uri: str) -> tuple[str, str]:
    assert uri.startswith("models:/")
    name, stage = uri[len("models:/"):].rsplit("/", 1)
    return name, stage

async def uc_watch(uc_stage_uri: str, interval_s: int, on_new_version: Callable[[str], Awaitable[None]]):
    client = MlflowClient()
    name, stage = _parse_uc(uc_stage_uri)
    current: Optional[str] = None
    while True:
        try:
            versions = client.get_latest_versions(name, stages=[stage])
            if versions:
                resolved = f"models:/{name}/{versions[0].version}"
                if resolved != current:
                    log.info("Hot-reload: %s -> %s", uc_stage_uri, resolved)
                    await on_new_version(resolved)
                    current = resolved
        except Exception as e:
            log.warning("UC watch failed: %s", e)
        await asyncio.sleep(interval_s)
