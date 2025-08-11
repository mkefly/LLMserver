import asyncio, pytest
from server.reliability import ConcurrencyGate

@pytest.mark.asyncio
async def test_gate_allows_then_drains():
    g = ConcurrencyGate(max_concurrent=2, drain_seconds=1)
    started = 0
    async def work():
        nonlocal started
        async with g.slot():
            started += 1
            await asyncio.sleep(0.02)
    tasks = [asyncio.create_task(work()) for _ in range(3)]
    await asyncio.sleep(0.005)
    g.start_drain()
    with pytest.raises(RuntimeError):
        async with g.slot():
            pass
    await asyncio.gather(*tasks)
    assert started >= 2
