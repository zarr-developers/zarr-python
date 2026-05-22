from __future__ import annotations

import math
from typing import TYPE_CHECKING

from zarr.core._json import buffer_to_json, get_json, json_to_buffer, set_json
from zarr.core.buffer import cpu, default_buffer_prototype
from zarr.storage import MemoryStore
from zarr.storage._common import StorePath

if TYPE_CHECKING:
    from zarr.core.common import JSON


def test_json_to_buffer_round_trips() -> None:
    """`buffer_to_json` inverts `json_to_buffer` for an arbitrary JSON value."""
    obj: JSON = {"zarr_format": 3, "node_type": "group", "attributes": {"a": [1, 2, 3]}}
    buffer = json_to_buffer(obj)
    assert buffer_to_json(buffer) == obj


def test_json_to_buffer_uses_given_prototype() -> None:
    """`json_to_buffer` constructs the buffer from the supplied prototype."""
    prototype = default_buffer_prototype()
    buffer = json_to_buffer({"x": 1}, prototype=prototype)
    assert isinstance(buffer, prototype.buffer)


def test_json_to_buffer_allows_nan() -> None:
    """`json_to_buffer` permits NaN (writes it as `NaN`), matching zarr's policy."""
    buffer = json_to_buffer({"fill_value": math.nan})
    decoded = buffer_to_json(buffer)
    assert isinstance(decoded, dict)
    assert math.isnan(decoded["fill_value"])


async def test_get_json_reads_existing_key() -> None:
    """`get_json` returns the parsed document stored at an existing key."""
    store = MemoryStore()
    obj: JSON = {"zarr_format": 3, "node_type": "array"}
    await set_json(store, "zarr.json", obj)
    assert await get_json(store, "zarr.json") == obj


async def test_get_json_returns_none_for_missing_key() -> None:
    """`get_json` returns None (rather than raising) when the key is absent."""
    store = MemoryStore()
    assert await get_json(store, "does-not-exist") is None


async def test_set_json_then_get_json_round_trips() -> None:
    """`set_json` followed by `get_json` returns the original value."""
    store = MemoryStore()
    obj: JSON = {"a": 1, "b": [2, 3], "c": {"d": None}}
    await set_json(store, "doc.json", obj)
    assert await get_json(store, "doc.json") == obj


async def test_storepath_get_json_reads_existing_key() -> None:
    """`StorePath.get_json` reads and parses the document at its own path."""
    store = MemoryStore()
    obj: JSON = {"zarr_format": 2}
    await set_json(store, "group/.zgroup", obj)
    sp = StorePath(store, "group/.zgroup")
    assert await sp.get_json() == obj


async def test_storepath_get_json_returns_none_for_missing() -> None:
    """`StorePath.get_json` returns None when its path is absent."""
    store = MemoryStore()
    sp = StorePath(store, "missing")
    assert await sp.get_json() is None


def test_buffer_to_json_on_cpu_buffer() -> None:
    """`buffer_to_json` works on a plain CPU buffer built from raw bytes."""
    buffer = cpu.Buffer.from_bytes(b'{"hello": "world"}')
    assert buffer_to_json(buffer) == {"hello": "world"}
