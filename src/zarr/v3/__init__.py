from __future__ import annotations

from typing import Union

import zarr.v3.codecs  # noqa: F401
from zarr.v3.array import Array  # noqa: F401
from zarr.v3.array_v2 import ArrayV2  # noqa: F401
from zarr.v3.group import Group  # noqa: F401
from zarr.v3.metadata import RuntimeConfiguration, runtime_configuration  # noqa: F401
from zarr.v3.store import (  # noqa: F401
    LocalStore,
    RemoteStore,
    Store,
    StoreLike,
    StorePath,
    make_store_path,
)
from zarr.v3.sync import sync as _sync


async def open_auto_async(
    store: StoreLike,
    runtime_configuration_: RuntimeConfiguration = RuntimeConfiguration(),
) -> Union[Array, ArrayV2, Group]:
    store_path = make_store_path(store)
    try:
        return await Array.open(store_path, runtime_configuration=runtime_configuration_)
    except KeyError:
        return await Group.open(store_path, runtime_configuration=runtime_configuration_)
    


def open_auto(
    store: StoreLike,
    runtime_configuration_: RuntimeConfiguration = RuntimeConfiguration(),
) -> Union[Array, ArrayV2, Group]:
    return _sync(
        open_auto_async(store, runtime_configuration_),
        runtime_configuration_.asyncio_loop,
    )
