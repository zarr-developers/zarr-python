import asyncio
from time import time

import pytest

import zarr
from zarr.storage import MemoryStore
from zarr.testing.store import LatencyStore


@pytest.mark.parametrize("num_members", [4, 8, 16])
def test_collect_members(num_members: int) -> None:
    local_store = MemoryStore(mode="a")
    local_latency_store = LatencyStore(local_store, get_latency=0.1, set_latency=0.0)

    root_group_raw = zarr.open_group(store=local_store)
    root_group_latency = zarr.open_group(store=local_latency_store)
    for i in range(num_members):
        root_group_raw.create_group(f"group_{i}")

    async def get_members_async() -> None:
        members2 = await root_group_latency._async_group._members2(max_depth=0, current_depth=0)
        print(members2)

    start = time()
    asyncio.run(get_members_async())
    elapsed = time() - start
    print(elapsed)
