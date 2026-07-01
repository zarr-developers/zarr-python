from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
from hypothesis import given

import zarr
from zarr.codecs import BytesCodec, CastValue, GzipCodec, TransposeCodec
from zarr.core.array import _get_chunk_spec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.codec_pipeline import codecs_from_list
from zarr.core.indexing import BasicIndexer
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from zarr.abc.codec import Codec
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype


@pytest.mark.parametrize(
    ("write_slice", "read_slice", "expected_statuses"),
    [
        # Write all chunks, read all — all present
        (slice(None), slice(None), ("present", "present", "present")),
        # Write first chunk only, read all — first present, rest missing
        (slice(0, 2), slice(None), ("present", "missing", "missing")),
        # Write nothing, read all — all missing
        (None, slice(None), ("missing", "missing", "missing")),
    ],
)
async def test_read_returns_get_results(
    write_slice: slice | None,
    read_slice: slice,
    expected_statuses: tuple[str, ...],
) -> None:
    """
    Test that CodecPipeline.read returns a tuple of GetResult with correct statuses.
    """
    store = MemoryStore()
    arr = zarr.open_array(store, mode="w", shape=(6,), chunks=(2,), dtype="int64", fill_value=-1)

    if write_slice is not None:
        arr[write_slice] = 0

    async_arr = arr._async_array
    pipeline = async_arr.codec_pipeline
    metadata = async_arr.metadata

    prototype = default_buffer_prototype()
    config = async_arr.config
    indexer = BasicIndexer(
        read_slice,
        shape=metadata.shape,
        chunk_grid=async_arr._chunk_grid,
    )

    out_buffer = prototype.nd_buffer.empty(
        shape=indexer.shape,
        dtype=metadata.dtype.to_native_dtype(),
        order=config.order,
    )

    results = await pipeline.read(
        [
            (
                async_arr.store_path / metadata.encode_chunk_key(chunk_coords),
                _get_chunk_spec(metadata, async_arr._chunk_grid, chunk_coords, config, prototype),
                chunk_selection,
                out_selection,
                is_complete_chunk,
            )
            for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexer
        ],
        out_buffer,
        drop_axes=indexer.drop_axes,
    )

    assert len(results) == len(expected_statuses)
    for result, expected_status in zip(results, expected_statuses, strict=True):
        assert result["status"] == expected_status


try:
    import cast_value_rs  # noqa: F401

    _HAS_CAST_VALUE_RS = True
except ModuleNotFoundError:
    _HAS_CAST_VALUE_RS = False

requires_cast_value_rs = pytest.mark.skipif(
    not _HAS_CAST_VALUE_RS, reason="cast-value-rs not installed"
)


@requires_cast_value_rs
@pytest.mark.parametrize(
    ("source_dtype", "target_dtype"),
    [
        # Source is single-byte (no endianness); target is multi-byte (has endianness).
        # Without the fix, BytesCodec.evolve_from_array_spec sees the source dtype,
        # strips its `endian` to None, and then chokes when the chunk_spec dtype
        # gets transformed to the multi-byte target before bytes-decoding.
        ("int8", "int16"),
        ("uint8", "int32"),
        ("int8", "float32"),
        # Source is multi-byte; target is single-byte (the reverse direction also
        # exercises the spec-threading logic).
        ("int16", "int8"),
    ],
)
def test_codec_pipeline_threads_dtype_through_evolve(source_dtype: str, target_dtype: str) -> None:
    """Regression for #3937: each codec must be evolved against the spec it
    will see at runtime, not the original array spec. cast_value transforms
    the dtype between AA codecs and the array->bytes serializer."""
    arr = zarr.create_array(
        store={},
        shape=(4,),
        chunks=(4,),
        dtype=source_dtype,
        fill_value=0,
        filters=[CastValue(data_type=target_dtype)],
        serializer=BytesCodec(endian="little"),
        compressors=[],
        zarr_format=3,
        overwrite=True,
    )
    arr[:] = np.asarray([0, 1, 2, 3], dtype=source_dtype)
    np.testing.assert_array_equal(arr[:], np.asarray([0, 1, 2, 3], dtype=source_dtype))


# Property-based check of codecs_from_list ordering validation.
#
# Valid codec orderings are exactly: (ArrayArrayCodec)* (ArrayBytesCodec)
# (BytesBytesCodec)*. codecs_from_list walks adjacent pairs and must raise
# TypeError the moment a codec appears in a structurally invalid position --
# notably, a BytesBytesCodec immediately following an ArrayArrayCodec with no
# ArrayBytesCodec in between (which previously built an error message but never
# raised it, falling through to an unrelated ValueError instead).
_AA = "AA"  # ArrayArrayCodec   -> TransposeCodec
_AB = "AB"  # ArrayBytesCodec   -> BytesCodec
_BB = "BB"  # BytesBytesCodec   -> GzipCodec

_CODEC_FACTORY: dict[str, Callable[[], Codec]] = {
    _AA: lambda: TransposeCodec(order=(0, 1)),
    _AB: BytesCodec,
    _BB: GzipCodec,
}


def _expected_codec_order_outcome(labels: list[str]) -> str:
    """Independently predict codecs_from_list's outcome: 'TypeError',
    'ValueError' or 'ok', mirroring its left-to-right scan and the order in
    which it checks ordering violations (TypeError) vs. the ArrayBytes-count
    constraints (ValueError)."""
    prev = None
    seen_array_bytes = False
    for cur in labels:
        if cur == _AA:
            if prev in (_AB, _BB):
                return "TypeError"
        elif cur == _AB:
            if prev == _BB:
                return "TypeError"
            if seen_array_bytes:
                return "ValueError"  # two ArrayBytesCodecs
            seen_array_bytes = True
        else:  # _BB
            if prev == _AA:
                return "TypeError"
        prev = cur
    if not seen_array_bytes:
        return "ValueError"  # Required ArrayBytesCodec was not found
    return "ok"


@given(labels=st.lists(st.sampled_from([_AA, _AB, _BB]), min_size=1, max_size=5))
def test_codecs_from_list_outcome_matches_order_rules(labels: list[str]) -> None:
    codecs = [_CODEC_FACTORY[label]() for label in labels]
    expected = _expected_codec_order_outcome(labels)
    if expected == "TypeError":
        with pytest.raises(TypeError):
            codecs_from_list(codecs)
    elif expected == "ValueError":
        with pytest.raises(ValueError):
            codecs_from_list(codecs)
    else:
        # Valid ordering: must classify without raising.
        aa, _ab, bb = codecs_from_list(codecs)
        assert labels.count(_AA) == len(aa)
        assert labels.count(_BB) == len(bb)


async def test_read_uses_bulk_get_many() -> None:
    """The pipeline should fetch a whole multi-chunk read with a single
    ``Store.get_many`` call (spanning the entire request, independent of
    ``codec_pipeline.batch_size``), rather than one ``get`` per chunk."""
    store = MemoryStore()
    arr = zarr.create_array(store, shape=(20,), chunks=(5,), dtype="int64")  # 4 chunks
    arr[:] = np.arange(20)

    calls: dict[str, int] = {"get_many": 0, "requests": 0}
    orig_get_many = store.get_many

    # get_many is an async generator, so the spy is a sync function returning
    # the underlying async iterator; count at call time.
    def spy_get_many(
        requests: Sequence[tuple[str, ByteRequest | None] | str],
        *,
        prototype: BufferPrototype,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        requests = list(requests)
        calls["get_many"] += 1
        calls["requests"] += len(requests)
        return orig_get_many(requests, prototype=prototype)

    store.get_many = spy_get_many  # type: ignore[method-assign]

    result = arr[:]
    np.testing.assert_array_equal(result, np.arange(20))
    # one bulk call covering all four chunks
    assert calls["get_many"] == 1
    assert calls["requests"] == 4


async def test_read_bulk_handles_missing_chunks() -> None:
    """A bulk read where some chunks were never written must still fill those
    positions with the fill value (get_many reports missing keys as None)."""
    store = MemoryStore()
    arr = zarr.open_array(store, mode="w", shape=(20,), chunks=(5,), dtype="int64", fill_value=-1)
    arr[0:5] = 7  # write only the first chunk; the other three are missing

    result = arr[:]
    expected = np.full(20, -1, dtype="int64")
    expected[0:5] = 7
    np.testing.assert_array_equal(result, expected)
