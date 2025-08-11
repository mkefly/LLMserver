import asyncio, pytest
from unittest.mock import patch
from server.models.resolvers import UCAliasResolver

class V: 
    def __init__(self, version): self.version = version

@pytest.mark.asyncio
async def test_uc_alias_watch_triggers():
    calls = []
    async def on_change(uri: str): calls.append(uri)

    with patch("server.models.resolvers.MlflowClient") as C:
        inst = C.return_value
        inst.get_model_version_by_alias.side_effect = [V("1"), V("1"), V("2")]
        r = UCAliasResolver()
        task = asyncio.create_task(r.watch("models:/c.s.m@champion", 0.01, on_change))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert calls[0].endswith("/1")
    assert calls[1].endswith("/2")
