import asyncio
from time import time
from typing import Literal

import pytest

import zarr
from zarr.core.group import members_v3
from zarr.storage import MemoryStore
from zarr.testing.store import LatencyStore


@pytest.mark.parametrize("num_members", [4, 8, 16])
@pytest.mark.parametrize("method", ["default", "fast_members"])
def test_collect_members(num_members: int, method: Literal["fast_members", "default"]) -> None:
    local_store = MemoryStore(mode="a")
    local_latency_store = LatencyStore(local_store, get_latency=0.1, set_latency=0.0)

    root_group_raw = zarr.open_group(store=local_store)
    root_group_latency = zarr.open_group(store=local_latency_store)

    for i in range(num_members):
        subgroup = root_group_raw.create_group(f"group_outer_{i}")
        for j in range(num_members):
            subgroup.create_group(f"group_inner_{j}")

    if method == "fast_members":

        async def amain() -> None:
            res = await members_v3(local_latency_store, path="")
            print(res)

        start = time()
        asyncio.run(amain())
        elapsed = time() - start
        print(elapsed)
    else:
        start = time()
        root_group_latency.members(max_depth=None)
        elapsed = time() - start
        print(elapsed)
