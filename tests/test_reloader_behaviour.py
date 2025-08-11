import asyncio
import pytest
from unittest.mock import patch, MagicMock
from server.reloader import uc_watch

class V: 
    def __init__(self, version): self.version = version

@pytest.mark.asyncio
async def test_uc_watch_handles_exception_and_continues():
    calls = []
    async def on_new(uri): calls.append(uri)
    with patch("server.reloader.MlflowClient") as C:
        inst = C.return_value
        inst.get_latest_versions.side_effect = [
            Exception("network"),
            [V("7")],
            [V("7")],
            [V("8")],
        ]
        task = asyncio.create_task(uc_watch("models:/c.s.m/Production", 0.01, on_new))
        await asyncio.sleep(0.06)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert calls[0].endswith("/7")
    assert calls[1].endswith("/8")
