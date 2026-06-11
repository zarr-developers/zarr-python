from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest

from zarr.abc.codec import ArrayArrayCodec
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.sharding import (
    MAX_UINT_64,
    ShardingCodec,
    _ShardIndex,
    _ShardingByteGetter,
    _ShardReader,
)
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.buffer.cpu import Buffer
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.storage._common import StorePath
from zarr.storage._memory import MemoryStore

# ============================================================================
# _ShardIndex tests
# ============================================================================


def test_shard_index_create_empty() -> None:
    """Test that create_empty creates an index filled with MAX_UINT_64."""
    chunks_per_shard = (2, 3)
    index = _ShardIndex.create_empty(chunks_per_shard)

    assert index.chunks_per_shard == chunks_per_shard
    assert index.offsets_and_lengths.shape == (2, 3, 2)
    assert index.offsets_and_lengths.dtype == np.dtype("<u8")
    assert np.all(index.offsets_and_lengths == MAX_UINT_64)


def test_shard_index_create_empty_1d() -> None:
    """Test create_empty with 1D chunks_per_shard."""
    chunks_per_shard = (4,)
    index = _ShardIndex.create_empty(chunks_per_shard)

    assert index.chunks_per_shard == chunks_per_shard
    assert index.offsets_and_lengths.shape == (4, 2)


def test_shard_index_is_all_empty_true() -> None:
    """Test is_all_empty returns True for a freshly created empty index."""
    index = _ShardIndex.create_empty((2, 2))
    assert index.is_all_empty() is True


def test_shard_index_is_all_empty_false() -> None:
    """Test is_all_empty returns False when at least one chunk is set."""
    index = _ShardIndex.create_empty((2, 2))
    index.set_chunk_slice((0, 0), slice(0, 100))
    assert index.is_all_empty() is False


def test_shard_index_get_chunk_slice_empty() -> None:
    """Test get_chunk_slice returns None for empty chunks."""
    index = _ShardIndex.create_empty((2, 2))
    assert index.get_chunk_slice((0, 0)) is None
    assert index.get_chunk_slice((1, 1)) is None


def test_shard_index_get_chunk_slice_set() -> None:
    """Test get_chunk_slice returns correct (start, end) tuple after setting."""
    index = _ShardIndex.create_empty((2, 2))
    index.set_chunk_slice((0, 1), slice(100, 200))

    result = index.get_chunk_slice((0, 1))
    assert result == (100, 200)


def test_shard_index_set_chunk_slice() -> None:
    """Test set_chunk_slice correctly sets offset and length."""
    index = _ShardIndex.create_empty((3, 3))

    # Set a chunk slice
    index.set_chunk_slice((1, 2), slice(50, 150))

    # Verify the underlying array
    assert index.offsets_and_lengths[1, 2, 0] == 50  # offset
    assert index.offsets_and_lengths[1, 2, 1] == 100  # length (150 - 50)


def test_shard_index_set_chunk_slice_none() -> None:
    """Test set_chunk_slice with None marks chunk as empty."""
    index = _ShardIndex.create_empty((2, 2))

    # First set a value
    index.set_chunk_slice((0, 0), slice(0, 100))
    assert index.get_chunk_slice((0, 0)) == (0, 100)

    # Then clear it
    index.set_chunk_slice((0, 0), None)
    assert index.get_chunk_slice((0, 0)) is None
    assert index.offsets_and_lengths[0, 0, 0] == MAX_UINT_64
    assert index.offsets_and_lengths[0, 0, 1] == MAX_UINT_64


def test_shard_index_get_full_chunk_map() -> None:
    """Test get_full_chunk_map returns correct boolean array."""
    index = _ShardIndex.create_empty((2, 3))

    # Set some chunks
    index.set_chunk_slice((0, 0), slice(0, 10))
    index.set_chunk_slice((1, 2), slice(10, 20))

    chunk_map = index.get_full_chunk_map()

    assert chunk_map.shape == (2, 3)
    assert chunk_map.dtype == np.bool_
    assert chunk_map[0, 0] is np.True_
    assert chunk_map[0, 1] is np.False_
    assert chunk_map[0, 2] is np.False_
    assert chunk_map[1, 0] is np.False_
    assert chunk_map[1, 1] is np.False_
    assert chunk_map[1, 2] is np.True_


def test_shard_index_localize_chunk() -> None:
    """Test _localize_chunk maps global coords to local shard coords via modulo."""
    index = _ShardIndex.create_empty((2, 3))

    # Within bounds - should return same coords
    assert index._localize_chunk((0, 0)) == (0, 0)
    assert index._localize_chunk((1, 2)) == (1, 2)

    # Out of bounds - should wrap via modulo
    assert index._localize_chunk((2, 0)) == (0, 0)  # 2 % 2 = 0
    assert index._localize_chunk((3, 5)) == (1, 2)  # 3 % 2 = 1, 5 % 3 = 2
    assert index._localize_chunk((4, 6)) == (0, 0)  # 4 % 2 = 0, 6 % 3 = 0


# ============================================================================
# _load_partial_shard_maybe tests
#
# These exercise the partial-shard read path against a real MemoryStore wrapped
# in a StorePath (the external-store branch in `_load_partial_shard_maybe`),
# plus one test against a real `_ShardingByteGetter` (the in-memory branch used
# by nested sharding).
# ============================================================================


async def _store_path_with_blob(key: str, blob: bytes) -> StorePath:
    """Build a `StorePath` over a fresh `MemoryStore` containing `blob` at `key`."""
    store = MemoryStore()
    await store.set(key, Buffer.from_bytes(blob))
    return StorePath(store, key)


async def test_load_partial_shard_maybe_index_load_fails() -> None:
    """Returns None when the shard key is absent (index load fails)."""
    codec = ShardingCodec(chunk_shape=(8,))
    byte_getter = StorePath(MemoryStore(), "missing")

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=(2,),
        all_chunk_coords={(0,)},
    )

    assert result is None


async def test_load_partial_shard_maybe_with_empty_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunks whose index entry is empty are silently skipped."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    # Index where chunk (1,) is empty; the others point into the stored blob.
    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((2,), slice(100, 200))
    index.set_chunk_slice((3,), slice(200, 300))

    async def mock_load_index(
        self: ShardingCodec, byte_getter: StorePath, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    byte_getter = await _store_path_with_blob("shard", b"x" * 300)

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords={(0,), (1,), (2,)},
    )

    assert result is not None
    assert (0,) in result
    assert (1,) not in result  # empty in index
    assert (2,) in result


async def test_load_partial_shard_maybe_all_chunks_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns an empty dict when all requested chunks are empty (no I/O issued)."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    # Fully-empty index — `get_chunk_slice` returns None for every coord.
    index = _ShardIndex.create_empty(chunks_per_shard)

    async def mock_load_index(
        self: ShardingCodec, byte_getter: StorePath, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    # Empty store is fine — we never reach the chunk-read path when all are empty.
    byte_getter = StorePath(MemoryStore(), "shard")

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords={(0,), (1,), (2,)},
    )

    assert result == {}


async def test_load_partial_shard_returns_chunk_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns the correct bytes for each requested chunk."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((1,), slice(100, 200))

    async def mock_load_index(
        self: ShardingCodec, byte_getter: StorePath, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    blob = b"A" * 100 + b"B" * 100
    byte_getter = await _store_path_with_blob("shard", blob)

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords={(0,), (1,)},
    )

    assert result is not None
    buf_0, buf_1 = result[(0,)], result[(1,)]
    assert buf_0 is not None
    assert buf_1 is not None
    assert buf_0.to_bytes() == b"A" * 100
    assert buf_1.to_bytes() == b"B" * 100


async def test_load_partial_shard_shard_disappears_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the shard key is missing when chunk reads run, returns None.

    This models a race: the index loaded successfully, but the shard was deleted
    before the chunk-byte fetches landed. `Store.get_ranges` surfaces this as a
    `BaseExceptionGroup` containing `FileNotFoundError`, which the codec catches
    and converts to None to match the index-missing branch's behavior.
    """
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))

    async def mock_load_index(
        self: ShardingCodec, byte_getter: StorePath, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    # Store has no value for "shard" — `get_ranges` will raise FileNotFoundError.
    byte_getter = StorePath(MemoryStore(), "shard")

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords={(0,)},
    )

    assert result is None


async def test_load_partial_shard_non_fnf_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-FileNotFoundError errors from get_ranges are re-raised, not swallowed.

    Our `BaseExceptionGroup.split(FileNotFoundError)` keeps the "shard gone"
    behavior for FNF only; anything else (e.g. an OSError from the underlying
    fetch) must bubble up.
    """
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))

    async def mock_load_index(
        self: ShardingCodec, byte_getter: StorePath, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    # Make the underlying store.get raise OSError. The default Store.get_ranges
    # impl routes through self.get; coalesced_get wraps the failure in a
    # BaseExceptionGroup, which our code re-raises (minus FNF leaves, of which
    # there are none here).
    async def boom(*args: object, **kwargs: object) -> Buffer | None:
        raise OSError("injected disk error")

    store = MemoryStore()
    monkeypatch.setattr(store, "get", boom)
    byte_getter = StorePath(store, "shard")

    with pytest.RaisesGroup(pytest.RaisesExc(OSError, match="injected disk error")):
        await codec._load_partial_shard_maybe(
            byte_getter=byte_getter,
            prototype=default_buffer_prototype(),
            chunks_per_shard=chunks_per_shard,
            all_chunk_coords={(0,)},
        )


async def test_load_partial_shard_nested_sharding_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested sharding: byte_getter is a `_ShardingByteGetter` over an in-memory dict."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((1,), slice(100, 200))

    async def mock_load_index(
        self: ShardingCodec, byte_getter: _ShardingByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    # The "store" for an inner shard is a dict keyed by outer-chunk coords; the
    # byte_getter reads ranges out of one entry of that dict.
    blob = b"A" * 100 + b"B" * 100
    shard_dict: dict[tuple[int, ...], Buffer | None] = {(0,): Buffer.from_bytes(blob)}
    byte_getter = _ShardingByteGetter(shard_dict, (0,))

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords={(0,), (1,)},
    )

    assert result is not None
    buf_0, buf_1 = result[(0,)], result[(1,)]
    assert buf_0 is not None
    assert buf_1 is not None
    assert buf_0.to_bytes() == b"A" * 100
    assert buf_1.to_bytes() == b"B" * 100


async def test_load_partial_shard_nested_sharding_missing_outer_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested sharding: outer chunk absent → `_ShardingByteGetter.get` returns None
    → chunks are silently skipped, yielding an empty shard_dict."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))

    async def mock_load_index(
        self: ShardingCodec, byte_getter: _ShardingByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    # Empty outer dict — _ShardingByteGetter.get(...) returns None for any range.
    shard_dict: dict[tuple[int, ...], Buffer | None] = {}
    byte_getter = _ShardingByteGetter(shard_dict, (0,))

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords={(0,)},
    )

    assert result == {}


# ============================================================================
# Supporting class tests (_ShardReader, _is_total_shard)
# ============================================================================


def test_shard_reader_create_empty() -> None:
    """Test _ShardReader.create_empty creates reader with empty index."""
    chunks_per_shard = (2, 3)
    reader = _ShardReader.create_empty(chunks_per_shard)

    assert reader.index.is_all_empty()
    assert len(reader.buf) == 0
    assert len(reader) == 2 * 3


def test_shard_reader_iteration() -> None:
    """Test _ShardReader iteration yields all chunk coordinates."""
    chunks_per_shard = (2, 2)
    reader = _ShardReader.create_empty(chunks_per_shard)

    coords = list(reader)

    assert len(coords) == 4
    assert (0, 0) in coords
    assert (0, 1) in coords
    assert (1, 0) in coords
    assert (1, 1) in coords


def test_shard_reader_getitem_raises_for_empty() -> None:
    """Test _ShardReader.__getitem__ raises KeyError for empty chunks."""
    chunks_per_shard = (2,)
    reader = _ShardReader.create_empty(chunks_per_shard)

    with pytest.raises(KeyError):
        _ = reader[(0,)]


def test_is_total_shard_full() -> None:
    """Test _is_total_shard returns True when all chunk coords are present."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (2, 2)
    all_chunk_coords: set[tuple[int, ...]] = {(0, 0), (0, 1), (1, 0), (1, 1)}

    assert codec._is_total_shard(all_chunk_coords, chunks_per_shard) is True


def test_is_total_shard_partial() -> None:
    """Test _is_total_shard returns False for partial chunk coords."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (2, 2)
    all_chunk_coords: set[tuple[int, ...]] = {(0, 0), (1, 1)}  # Missing (0, 1) and (1, 0)

    assert codec._is_total_shard(all_chunk_coords, chunks_per_shard) is False


def test_is_total_shard_empty() -> None:
    """Test _is_total_shard returns False for empty chunk coords."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (2, 2)
    all_chunk_coords: set[tuple[int, ...]] = set()

    assert codec._is_total_shard(all_chunk_coords, chunks_per_shard) is False


def test_is_total_shard_1d() -> None:
    """Test _is_total_shard works with 1D shards."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)
    all_chunk_coords: set[tuple[int, ...]] = {(0,), (1,), (2,), (3,)}

    assert codec._is_total_shard(all_chunk_coords, chunks_per_shard) is True

    # Partial
    partial_coords: set[tuple[int, ...]] = {(0,), (2,)}
    assert codec._is_total_shard(partial_coords, chunks_per_shard) is False


# ============================================================================
# _inner_codecs_fixed_size tests
# ============================================================================


def test_inner_codecs_fixed_size_no_compression() -> None:
    """Inner codecs without compression should be fixed-size."""
    codec = ShardingCodec(chunk_shape=(10,), codecs=[BytesCodec()])
    assert codec._inner_codecs_fixed_size is True


def test_inner_codecs_fixed_size_with_compression() -> None:
    """Inner codecs with compression should NOT be fixed-size."""
    codec = ShardingCodec(chunk_shape=(10,), codecs=[BytesCodec(), GzipCodec()])
    assert codec._inner_codecs_fixed_size is False


# ============================================================================
# inner-chain spec threading
# ============================================================================


@dataclass(frozen=True)
class _WidenToInt16(ArrayArrayCodec):
    """Test-only sync-capable AA codec that reports its output dtype as int16."""

    is_fixed_size = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": "_widen_to_int16"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_WidenToInt16":
        return cls()

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return replace(chunk_spec, dtype=get_data_type_from_native_dtype(np.dtype("int16")))

    def compute_encoded_size(self, input_byte_length: int, _spec: ArraySpec) -> int:
        return input_byte_length

    def _encode_sync(self, chunk_array: Any, chunk_spec: ArraySpec) -> Any:
        return chunk_array  # pragma: no cover

    def _decode_sync(self, chunk_array: Any, chunk_spec: ArraySpec) -> Any:
        return chunk_array  # pragma: no cover

    async def _encode_single(self, chunk_array: Any, chunk_spec: ArraySpec) -> Any:
        return chunk_array  # pragma: no cover

    async def _decode_single(self, chunk_array: Any, chunk_spec: ArraySpec) -> Any:
        return chunk_array  # pragma: no cover


def _int8_spec(shape: tuple[int, ...]) -> ArraySpec:
    zdtype = get_data_type_from_native_dtype(np.dtype("int8"))  # single-byte source
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=False),
        prototype=default_buffer_prototype(),
    )


def test_inner_chunk_transform_threads_spec() -> None:
    """The inner codec chain must be evolved with the spec threaded forward.

    A dtype-widening inner array->array codec means the BytesCodec serializer
    is evolved against the WIDENED dtype, not the single-byte source —
    otherwise it strips its `endian` to None and fails to decode multi-byte
    inner chunks. Same contract as the pipeline-level `evolve_codecs`
    regression test, applied to `_get_inner_chunk_transform`.
    """
    codec = ShardingCodec(chunk_shape=(4,), codecs=[_WidenToInt16(), BytesCodec(endian="little")])
    shard_spec = _int8_spec((8,))

    transform = codec._get_inner_chunk_transform(shard_spec)
    serializer = transform._ab_codec
    assert isinstance(serializer, BytesCodec)
    assert serializer.endian is not None, (
        "inner BytesCodec lost its `endian` — _get_inner_chunk_transform did not "
        "thread the dtype-widening codec's spec into the serializer"
    )


def test_evolve_from_array_spec_threads_spec() -> None:
    """`ShardingCodec.evolve_from_array_spec` must thread the spec through the
    inner chain, like `_get_inner_chunk_transform` does.

    This method runs EARLIER, on the real array-creation path (the outer
    pipeline evolves the sharding codec itself), so an unthreaded evolve here
    bakes an endian-stripped BytesCodec into the evolved instance's `codecs`
    before the transform builders ever run — and the later threaded evolve then
    raises instead of recovering. Calling `_get_inner_chunk_transform` on the
    EVOLVED instance pins the full real path.
    """
    codec = ShardingCodec(chunk_shape=(4,), codecs=[_WidenToInt16(), BytesCodec(endian="little")])
    # the array spec the OUTER pipeline evolves the sharding codec against
    array_spec = _int8_spec((8,))

    evolved = codec.evolve_from_array_spec(array_spec)
    inner_serializer = next(c for c in evolved.codecs if isinstance(c, BytesCodec))
    assert inner_serializer.endian is not None, (
        "evolve_from_array_spec evolved the inner BytesCodec against the "
        "un-widened spec, stripping its `endian`"
    )

    # and the evolved instance must still build a working inner transform
    transform = evolved._get_inner_chunk_transform(array_spec)
    serializer = transform._ab_codec
    assert isinstance(serializer, BytesCodec)
    assert serializer.endian is not None
