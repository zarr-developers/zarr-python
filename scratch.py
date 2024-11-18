import asyncio
from time import time
from typing import Literal

import pytest

import zarr
from zarr.core.group import iter_members, members_recursive, iter_members_deep
from zarr.storage import MemoryStore
from zarr.testing.store import LatencyStore


@pytest.mark.parametrize("num_members", [10, 100, 1000])
@pytest.mark.parametrize("method", ["default", "fast_members"])
def test_collect_members(num_members: int, method: Literal["fast_members", "default", "fast_members_2"]) -> None:
    local_store = MemoryStore(mode="a")
    local_latency_store = LatencyStore(local_store, get_latency=0.1, set_latency=0.0)

    root_group_raw = zarr.open_group(store=local_store)
    root_group_latency = zarr.open_group(store=local_latency_store)

    for i in range(num_members):
        subgroup = root_group_raw.create_group(f"group_outer_{i}")

    if method == "fast_members":
        async def amain() -> None:
            res = [x async for x in iter_members(root_group_latency._async_group)]

        start = time()
        asyncio.run(amain())
        elapsed = time() - start
        print(elapsed)
    else:
        start = time()
        root_group_latency.members(max_depth=None)
        elapsed = time() - start
        print(elapsed)
