from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Self

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.buffer.cpu import Buffer as CpuBuffer

from zarr_storage.legacy import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)


class DictStore(Store):
    """A third-party-style store exercising inherited implementations."""

    supports_writes = True
    supports_deletes = True
    supports_listing = True

    def __init__(self, *, read_only: bool = False) -> None:
        super().__init__(read_only=read_only)
        self.data: dict[str, bytes] = {}
        self.reads = 0

    def __eq__(self, other: object) -> bool:
        return self is other

    def with_read_only(self, read_only: bool = False) -> Self:
        result = type(self)(read_only=read_only)
        result.data = self.data
        return result

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        self.reads += 1
        value = self.data.get(key)
        if value is None:
            return None
        match byte_range:
            case RangeByteRequest(start, end):
                value = value[start:end]
            case OffsetByteRequest(offset):
                value = value[offset:]
            case SuffixByteRequest(suffix):
                value = value[-suffix:]
        return prototype.buffer.from_bytes(value)

    async def get_partial_values(
        self, prototype: BufferPrototype, key_ranges: Iterable[tuple[str, ByteRequest | None]]
    ) -> list[Buffer | None]:
        return [await self.get(key, prototype, request) for key, request in key_ranges]

    async def exists(self, key: str) -> bool:
        return key in self.data

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        self.data[key] = value.to_bytes()

    async def delete(self, key: str) -> None:
        self._check_writable()
        self.data.pop(key, None)

    async def list(self) -> AsyncIterator[str]:
        for key in list(self.data):
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for key in list(self.data):
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/") + "/" if prefix else ""
        for child in sorted(
            {key[len(prefix) :].split("/")[0] for key in self.data if key.startswith(prefix)}
        ):
            yield child


async def test_inherited_store_operations() -> None:
    store = await DictStore.open()
    assert store._is_open
    await store.set_if_not_exists("a/one", CpuBuffer.from_bytes(b"0123456789"))
    await store.set_if_not_exists("a/one", CpuBuffer.from_bytes(b"replacement"))
    await store.set("a/two", CpuBuffer.from_bytes(b"abc"))
    await store.set("b/three", CpuBuffer.from_bytes(b"xyz"))
    assert await store.getsize("a/one") == 10
    assert await store.getsize_prefix("a/") == 13
    assert not await store.is_empty("a/")
    await store.delete_dir("a")
    assert store.data == {"b/three": b"xyz"}
    await store.clear()
    assert await store.is_empty("")
    store.close()
    assert not store._is_open


async def test_coalesced_ranges_use_extracted_request_types() -> None:
    store = DictStore()
    store.data["key"] = b"0123456789"
    requests = [RangeByteRequest(5, 8), RangeByteRequest(0, 3), SuffixByteRequest(2)]
    results = [
        item
        async for batch in store.get_ranges("key", requests, prototype=default_buffer_prototype())
        for item in batch
    ]
    assert {index: value.to_bytes() for index, value in results if value is not None} == {
        0: b"567",
        1: b"012",
        2: b"89",
    }
    assert store.reads == 2


async def test_getsize_missing_key() -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        await DictStore().getsize("missing")


async def test_read_only_write() -> None:
    with pytest.raises(ValueError, match="read-only"):
        await DictStore(read_only=True).set("key", CpuBuffer.from_bytes(b"data"))


@pytest.mark.parametrize(
    "byte_request", [RangeByteRequest(1, 3), OffsetByteRequest(2), SuffixByteRequest(4)]
)
def test_request_pickle(byte_request: ByteRequest) -> None:
    restored = pickle.loads(pickle.dumps(byte_request))
    assert type(restored) is type(byte_request)
    assert restored == byte_request
