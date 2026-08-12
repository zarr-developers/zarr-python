"""The original engine spec must survive `with_config` / `update_attributes`.

`with_config` and `update_attributes` build a fresh array wrapper; each must
re-thread the array's engine spec so a custom engine is not silently dropped in
favour of the default one. Each path stores the ORIGINAL spec (name/instance),
so a named engine is re-resolved against the new config and an instance is
carried through unchanged -- these tests use instances and assert identity,
which is exactly what regresses when the spec is dropped (the copy would fall
back to a freshly-resolved default engine instead).
"""

from __future__ import annotations

from typing import Any

import zarr
from zarr.core.array import AsyncArray
from zarr.core.engine import DefaultArrayEngine, DefaultAsyncArrayEngine
from zarr.storage import MemoryStore


def _async_array_with_custom_engine() -> tuple[AsyncArray[Any], DefaultAsyncArrayEngine]:
    z = zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="int8")
    aa = z.async_array
    custom = DefaultAsyncArrayEngine(
        store_path=aa.store_path, metadata=aa.metadata, config=aa.config
    )
    built = AsyncArray(
        metadata=aa.metadata, store_path=aa.store_path, config=aa.config, engine=custom
    )
    return built, custom


def test_asyncarray_with_config_preserves_engine() -> None:
    built, custom = _async_array_with_custom_engine()
    assert built.engine is custom
    copy = built.with_config({"order": "F"})
    # the custom engine spec is re-resolved (an instance is returned unchanged),
    # so the copy keeps it instead of falling back to a default engine
    assert copy.engine is custom


def test_array_with_config_preserves_engine() -> None:
    z = zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="int8")
    aa = z.async_array
    custom = DefaultArrayEngine(
        DefaultAsyncArrayEngine(store_path=aa.store_path, metadata=aa.metadata, config=aa.config)
    )
    arr = zarr.Array(aa, engine_spec=custom)
    assert arr.engine is custom
    copy = arr.with_config({"order": "F"})
    assert copy.engine is custom


def test_array_update_attributes_preserves_engine() -> None:
    z = zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="int8")
    aa = z.async_array
    custom = DefaultArrayEngine(
        DefaultAsyncArrayEngine(store_path=aa.store_path, metadata=aa.metadata, config=aa.config)
    )
    arr = zarr.Array(aa, engine_spec=custom)
    assert arr.engine is custom
    updated = arr.update_attributes({"foo": "bar"})
    assert updated.engine is custom
