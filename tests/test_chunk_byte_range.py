from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numcodecs
import numpy as np
import pytest

import zarr
from zarr.abc.store import RangeByteRequest, SuffixByteRequest
from zarr.codecs import BytesCodec, ShardingCodec, TransposeCodec, ZstdCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.errors import ZarrUserWarning
from zarr.storage import LocalStore, MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.abc.store import ByteRequest
    from zarr.codecs.sharding import IndexLocation
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.types import AnyArray


def make_array(
    tmp_path: Path,
    store_kind: str,
    layout: str,
    location: IndexLocation = "end",
    compressed: bool = False,
) -> AnyArray:
    store = MemoryStore() if store_kind == "memory" else LocalStore(tmp_path)
    if layout == "sharded":
        inner: list[BytesCodec | ZstdCodec] = (
            [BytesCodec(), ZstdCodec()] if compressed else [BytesCodec()]
        )
        return zarr.create(
            store=store,
            path="nested/array",
            shape=(5, 5),
            chunks=(4, 4),
            dtype="int32",
            zarr_format=3,
            codecs=[ShardingCodec(chunk_shape=(2, 2), codecs=inner, index_location=location)],
        )
    if layout == "v2":
        compressors = numcodecs.Zstd() if compressed else None
    else:
        compressors = [ZstdCodec()] if compressed else None
    return zarr.create_array(
        store=store,
        name="nested/array",
        shape=(5, 5),
        chunks=(2, 2),
        dtype="int32",
        zarr_format=2 if layout == "v2" else 3,
        compressors=compressors,
    )


@pytest.mark.parametrize("store_kind", ["memory", "local"])
@pytest.mark.parametrize("layout", ["v2", "v3", "sharded"])
@pytest.mark.parametrize("location", ["start", "end"])
@pytest.mark.parametrize("compressed", [False, True])
def test_chunk_byte_range_decodes_actual_payload(
    tmp_path: Path,
    store_kind: str,
    layout: str,
    location: IndexLocation,
    compressed: bool,
) -> None:
    array = make_array(tmp_path, store_kind, layout, location, compressed)
    data = np.arange(25, dtype="int32").reshape(5, 5) + 1
    array[:] = data
    # The last logical chunk intersects the array but is not a complete 2x2 block.
    for coords in ((0, 0), (0, 1), (2, 2)):
        result = array.chunk_byte_range(coords)
        assert result is not None
        key, offset, length = result
        assert key.startswith("nested/array/")
        encoded = sync(
            array.store_path.store.get(
                key, default_buffer_prototype(), RangeByteRequest(offset, offset + length)
            )
        )
        assert encoded is not None
        raw = numcodecs.Zstd().decode(encoded.to_bytes()) if compressed else encoded.to_bytes()
        decoded = np.frombuffer(raw, dtype="<i4").reshape(2, 2)
        i, j = (coord * 2 for coord in coords)
        expected = data[i : i + 2, j : j + 2]
        np.testing.assert_array_equal(decoded[: expected.shape[0], : expected.shape[1]], expected)


@pytest.mark.parametrize("layout", ["v2", "v3", "sharded"])
@pytest.mark.parametrize("store_kind", ["memory", "local"])
async def test_missing_chunks_and_async_api(tmp_path: Path, layout: str, store_kind: str) -> None:
    array = make_array(tmp_path, store_kind, layout)
    assert await array.async_array.chunk_byte_range((0, 0)) is None
    array[:2, :2] = 7
    assert await array.async_array.chunk_byte_range((0, 0)) is not None
    # In a sharded array this shares an existing object with the written chunk.
    assert await array.async_array.chunk_byte_range((0, 1)) is None
    assert await array.async_array.chunk_byte_range((2, 2)) is None
    array[:2, :2] = 0
    assert await array.async_array.chunk_byte_range((0, 0)) is None


@pytest.mark.parametrize(
    ("coords", "exception"),
    [
        ((-1, 0), IndexError),
        ((3, 0), IndexError),
        ((0, 3), IndexError),
        ((0,), ValueError),
        ((0, 0, 0), ValueError),
        ((0.0, 0), TypeError),
        ((True, 0), TypeError),
        ((np.bool_(False), 0), TypeError),
    ],
)
def test_invalid_coordinates(tmp_path: Path, coords: Any, exception: type[Exception]) -> None:
    array = make_array(tmp_path, "memory", "sharded")
    with pytest.raises(exception):
        array.chunk_byte_range(coords)


def test_numpy_integer_coordinates(tmp_path: Path) -> None:
    array = make_array(tmp_path, "memory", "sharded")
    array[:2, :2] = 5
    assert array.chunk_byte_range((np.int64(0), np.int32(0))) == array.chunk_byte_range((0, 0))


@pytest.mark.parametrize("location", ["start", "end"])
async def test_sharded_inspection_reads_only_index(tmp_path: Path, location: IndexLocation) -> None:
    class RecordingStore(MemoryStore):
        reads: list[ByteRequest | None]

        async def get(
            self,
            key: str,
            prototype: BufferPrototype | None = None,
            byte_range: ByteRequest | None = None,
        ) -> Buffer | None:
            self.reads.append(byte_range)
            return await super().get(key, prototype, byte_range)

    store = RecordingStore()
    store.reads = []
    array = zarr.create(
        store=store,
        shape=(4, 4),
        chunks=(4, 4),
        dtype="int32",
        zarr_format=3,
        codecs=[ShardingCodec(chunk_shape=(2, 2), codecs=[BytesCodec()], index_location=location)],
    )
    array[:2, :2] = 3
    store.reads.clear()
    assert await array.async_array.chunk_byte_range((0, 0)) is not None
    expected = RangeByteRequest(0, 68) if location == "start" else SuffixByteRequest(68)
    assert store.reads == [expected]


def test_scalar_array(tmp_path: Path) -> None:
    array = zarr.create_array(
        store=LocalStore(tmp_path), shape=(), chunks=(), dtype="int32", compressors=None
    )
    assert array.chunk_byte_range(()) is None
    array[...] = 7
    result = array.chunk_byte_range(())
    assert result == ("c", 0, 4)


def test_zero_size_array(tmp_path: Path) -> None:
    array = zarr.create_array(store=LocalStore(tmp_path), shape=(0,), chunks=(2,), dtype="int32")
    with pytest.raises(IndexError):
        array.chunk_byte_range((0,))


def test_nested_sharding_rejected(tmp_path: Path) -> None:
    array = zarr.create(
        store=LocalStore(tmp_path),
        shape=(4, 4),
        chunks=(4, 4),
        dtype="int32",
        zarr_format=3,
        codecs=[ShardingCodec(chunk_shape=(2, 2), codecs=[ShardingCodec(chunk_shape=(1, 1))])],
    )
    with pytest.raises(NotImplementedError, match="nested"):
        array.chunk_byte_range((0, 0))


def test_outer_transform_rejected(tmp_path: Path) -> None:
    with pytest.warns(ZarrUserWarning, match="Combining"):
        array = zarr.create(
            store=LocalStore(tmp_path),
            shape=(4, 4),
            chunks=(4, 4),
            dtype="int32",
            zarr_format=3,
            codecs=[TransposeCodec(order=(1, 0)), ShardingCodec(chunk_shape=(2, 2))],
        )
    with pytest.raises(NotImplementedError, match="outer codecs"):
        array.chunk_byte_range((0, 0))


@pytest.mark.parametrize(("offset", "length"), [(2**64 - 1, 16), (2**64 - 10, 20)])
async def test_invalid_index_range(tmp_path: Path, offset: int, length: int) -> None:
    array = make_array(tmp_path, "memory", "sharded")
    array[:2, :2] = 3
    assert isinstance(array.metadata, ArrayV3Metadata)
    codec = array.metadata.codecs[0]
    assert isinstance(codec, ShardingCodec)
    path = array.store_path / "c/0/0"
    original = await path.get(default_buffer_prototype())
    assert original is not None
    index = await codec._decode_shard_index(original[-68:], (2, 2))
    index = index._replace(offsets_and_lengths=index.offsets_and_lengths.copy())
    index.offsets_and_lengths[0, 0] = (offset, length)
    encoded_index = await codec._encode_shard_index(index)
    await path.set(
        default_buffer_prototype().buffer.from_bytes(
            original[:-68].to_bytes() + encoded_index.to_bytes()
        )
    )
    with pytest.raises(ValueError, match="Invalid byte range"):
        await array.async_array.chunk_byte_range((0, 0))


def test_rectilinear_grid_rejected(tmp_path: Path) -> None:
    with zarr.config.set({"array.rectilinear_chunks": True}):
        array = zarr.create_array(
            store=LocalStore(tmp_path), shape=(5,), chunks=((2, 3),), dtype="int32"
        )
        with pytest.raises(NotImplementedError, match="regular chunk grid"):
            array.chunk_byte_range((0,))


async def test_size_lookup_errors_are_not_reported_as_missing(tmp_path: Path) -> None:
    class UnavailableStore(MemoryStore):
        async def getsize(self, key: str) -> int:
            raise PermissionError(key)

    array = zarr.create_array(store=UnavailableStore(), shape=(4,), chunks=(2,), dtype="int32")
    with pytest.raises(PermissionError):
        await array.async_array.chunk_byte_range((0,))
