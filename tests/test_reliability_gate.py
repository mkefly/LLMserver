import asyncio
import pytest
from server.reliability import ConcurrencyGate

@pytest.mark.asyncio
async def test_concurrency_gate_allows_and_drains():
    g = ConcurrencyGate(max_concurrent=2, drain_seconds=1)

    started = 0
    async def work():
        nonlocal started
        async with g.slot():
            started += 1
            await asyncio.sleep(0.05)

    tasks = [asyncio.create_task(work()) for _ in range(3)]
    await asyncio.sleep(0.01)
    # simulate SIGTERM path by calling start_drain()
    g.start_drain()

    # new slots after drain should fail fast
    with pytest.raises(RuntimeError):
        async with g.slot():
            pass

    await asyncio.gather(*tasks)
    assert started in (2, 3)
