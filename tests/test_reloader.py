import asyncio
import pytest
from unittest.mock import MagicMock, patch
from server.reloader import uc_watch

class V:  # fake version object
    def __init__(self, version): self.version = version

@pytest.mark.asyncio
async def test_uc_watch_triggers_on_change():
    calls = []
    async def on_new(uri: str): calls.append(uri)

    with patch("server.reloader.MlflowClient") as C:
        inst = C.return_value
        # simulate: v1 then v1 (no change) then v2
        inst.get_latest_versions.side_effect = [
            [V("1")], [V("1")], [V("2")],
        ]
        task = asyncio.create_task(uc_watch("models:/cat.sch.name/Production", 0.01, on_new))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # Should have fired twice (first resolve, then v2)
    assert calls[0].endswith("/1")
    assert calls[1].endswith("/2")
