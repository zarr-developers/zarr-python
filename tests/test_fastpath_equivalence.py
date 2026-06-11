"""Property tests: every fast path must equal the general path.

The codec pipelines contain fast paths that skip work whose result is known —
the complete-chunk merge view, the vectorized whole-shard bulk decode, the
scalar-broadcast write memoization, byte-range coalescing. Each is only safe if
it produces results identical to the general path it bypasses. These tests pin
that equivalence on randomized inputs, so a fast path that silently diverges
(the bug class behind the bulk-decode endianness fix) fails here instead of
corrupting data downstream.

Convention for new fast paths: a fast path is "skip work whose result is
known", never "a different algorithm" — and it ships with a property test in
this module asserting equality with the general path.
"""

from __future__ import annotations

from typing import Any

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

import zarr
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.sharding import ShardingCodec, ShardingCodecIndexLocation
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.buffer.cpu import Buffer as CPUBuffer
from zarr.core.buffer.cpu import NDBuffer as CPUNDBuffer
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.codec_pipeline import _merge_chunk_array
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.indexing import BasicIndexer
from zarr.storage import MemoryStore

_DTYPES = st.sampled_from(["uint8", "int16", "float32"])


def _spec(shape: tuple[int, ...], dtype: str, *, write_empty_chunks: bool = True) -> ArraySpec:
    zdtype = get_data_type_from_native_dtype(np.dtype(dtype))
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=write_empty_chunks),
        prototype=default_buffer_prototype(),
    )


# ---------------------------------------------------------------------------
# _merge_chunk_array: the complete-chunk early return (a view of
# value[out_selection]) must equal the general create/copy + setitem path.
# ---------------------------------------------------------------------------


@st.composite
def _merge_cases(draw: st.DrawFn) -> tuple[np.ndarray, tuple[int, ...], tuple[slice, ...]]:
    ndim = draw(st.integers(1, 3))
    chunk_shape = tuple(draw(st.integers(1, 5)) for _ in range(ndim))
    n_blocks = draw(st.integers(1, 3))
    dtype = draw(_DTYPES)
    # value spans n_blocks chunk-sized blocks along axis 0; out_selection picks one
    value_shape = (chunk_shape[0] * n_blocks, *chunk_shape[1:])
    value = draw(npst.arrays(dtype=np.dtype(dtype), shape=value_shape))
    block = draw(st.integers(0, n_blocks - 1))
    out_selection = (
        slice(block * chunk_shape[0], (block + 1) * chunk_shape[0]),
        *(slice(0, s) for s in chunk_shape[1:]),
    )
    return value, chunk_shape, out_selection


@settings(max_examples=200, deadline=None)
@given(case=_merge_cases(), with_existing=st.booleans())
def test_merge_complete_chunk_equals_general_path(
    case: tuple[np.ndarray, tuple[int, ...], tuple[slice, ...]], with_existing: bool
) -> None:
    """The is_complete_chunk fast path (return a view of value[out_selection])
    must produce exactly what the general merge path produces — and a complete
    write must be independent of any existing chunk content."""
    value_np, chunk_shape, out_selection = case
    spec = _spec(chunk_shape, str(value_np.dtype))
    value = CPUNDBuffer.from_numpy_array(value_np)
    chunk_selection = tuple(slice(0, s) for s in chunk_shape)

    existing = None
    if with_existing:
        existing = CPUNDBuffer.from_numpy_array(np.full(chunk_shape, 7, dtype=value_np.dtype))

    fast = _merge_chunk_array(existing, value, out_selection, spec, chunk_selection, True, ())
    general = _merge_chunk_array(existing, value, out_selection, spec, chunk_selection, False, ())
    np.testing.assert_array_equal(fast.as_numpy_array(), general.as_numpy_array())
    np.testing.assert_array_equal(fast.as_numpy_array(), value_np[out_selection])


# ---------------------------------------------------------------------------
# _decode_full_shard_bulk: the vectorized dense-shard decode must equal the
# general per-chunk decode (_decode_sync), across dtypes, endianness, write
# orders, and index locations. This is the bug class of the historical
# bulk-decode endianness fix.
# ---------------------------------------------------------------------------


@st.composite
def _shard_cases(draw: st.DrawFn) -> dict[str, Any]:
    ndim = draw(st.integers(1, 2))
    chunk_shape = tuple(draw(st.integers(1, 4)) for _ in range(ndim))
    grid = tuple(draw(st.integers(1, 3)) for _ in range(ndim))
    shard_shape = tuple(c * g for c, g in zip(chunk_shape, grid, strict=True))
    dtype = draw(_DTYPES)
    data = draw(npst.arrays(dtype=np.dtype(dtype), shape=shard_shape))
    return {
        "chunk_shape": chunk_shape,
        "shard_shape": shard_shape,
        "data": data,
        "endian": draw(st.sampled_from(["little", "big"])),
        "index_location": draw(st.sampled_from(list(ShardingCodecIndexLocation))),
        "subchunk_write_order": draw(
            st.sampled_from(["morton", "lexicographic", "colexicographic", "unordered"])
        ),
    }


@settings(max_examples=100, deadline=None)
@given(case=_shard_cases())
def test_bulk_shard_decode_equals_general_decode(case: dict[str, Any]) -> None:
    """For dense fixed-size uncompressed shards, the vectorized bulk decode must
    reproduce the general per-chunk decode exactly, whatever the endianness,
    subchunk write order, or index location."""
    codec = ShardingCodec(
        chunk_shape=case["chunk_shape"],
        codecs=[BytesCodec(endian=case["endian"])],
        index_location=case["index_location"],
        subchunk_write_order=case["subchunk_write_order"],
    )
    spec = _spec(case["shard_shape"], str(case["data"].dtype), write_empty_chunks=True)
    blob = codec._encode_sync(CPUNDBuffer.from_numpy_array(case["data"]), spec)
    assert blob is not None  # write_empty_chunks=True -> dense, never elided

    general = codec._decode_sync(blob, spec)
    indexer = BasicIndexer(
        tuple(slice(0, s) for s in case["shard_shape"]),
        shape=case["shard_shape"],
        chunk_grid=ChunkGrid.from_sizes(case["shard_shape"], case["chunk_shape"]),
    )
    bulk = codec._decode_full_shard_bulk(blob, spec, indexer)
    # the fast path must APPLY for this dense uncompressed configuration —
    # a vacuous None would silently stop testing the equivalence
    assert bulk is not None
    np.testing.assert_array_equal(bulk.as_numpy_array(), general.as_numpy_array())
    np.testing.assert_array_equal(general.as_numpy_array(), case["data"])


def test_merge_complete_chunk_returns_view_and_write_does_not_mutate_source() -> None:
    """The complete-chunk merge fast path returns a VIEW of the caller's value
    (no copy — that is the perf win), and a multi-chunk write through either
    pipeline leaves the user's source array untouched.

    Pins both halves of the aliasing contract: a future "defensive copy"
    refactor that silently reintroduces the per-chunk create/fill/copy breaks
    the first assertion, and an in-place-mutating codec that corrupts the
    user's array through the shared view breaks the second.
    """
    # the fast path must return a view aliasing `value`, not a copy
    value_np = np.arange(30, dtype="uint16")
    spec = _spec((10,), "uint16")
    value = CPUNDBuffer.from_numpy_array(value_np)
    merged = _merge_chunk_array(None, value, (slice(10, 20),), spec, (slice(0, 10),), True, ())
    assert np.shares_memory(merged.as_numpy_array(), value_np), (
        "complete-chunk merge no longer returns a view of the caller's value"
    )

    # end-to-end: the source array is byte-identical after a multi-chunk write
    for pipeline_path in (
        "zarr.core.codec_pipeline.FusedCodecPipeline",
        "zarr.core.codec_pipeline.BatchedCodecPipeline",
    ):
        with zarr.config.set({"codec_pipeline.path": pipeline_path}):
            arr = zarr.create_array(
                store=MemoryStore(),
                shape=(30,),
                chunks=(10,),
                dtype="uint16",
                compressors=None,
                fill_value=0,
            )
            source = np.arange(30, dtype="uint16")
            snapshot = source.copy()
            arr[:] = source
            np.testing.assert_array_equal(source, snapshot, err_msg=pipeline_path)
            np.testing.assert_array_equal(arr[:], snapshot, err_msg=pipeline_path)


# ---------------------------------------------------------------------------
# Scalar-broadcast write memoization: writing a scalar must produce the same
# STORED BYTES as writing the equivalent broadcast array.
# ---------------------------------------------------------------------------


@st.composite
def _scalar_cases(draw: st.DrawFn) -> dict[str, Any]:
    n_chunks = draw(st.integers(2, 6))
    chunk = draw(st.integers(2, 6))
    shape = n_chunks * chunk
    start = draw(st.integers(0, shape - 1))
    stop = draw(st.integers(start + 1, shape))
    return {
        "shape": shape,
        "chunk": chunk,
        "sel": slice(start, stop),
        "scalar": draw(st.integers(0, 255)),
        "write_empty_chunks": draw(st.booleans()),
        "sharded": draw(st.booleans()),
    }


@settings(max_examples=100, deadline=None)
@given(case=_scalar_cases())
def test_scalar_write_equals_broadcast_write(case: dict[str, Any]) -> None:
    """arr[sel] = scalar and arr[sel] = full(sel_shape, scalar) must leave the
    store byte-identical (pins the scalar-broadcast memoization in the sharded
    partial-write path, incl. its empty-chunk normalization)."""

    def build() -> tuple[MemoryStore, zarr.Array[Any]]:
        store = MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=(case["shape"],),
            chunks=(case["chunk"],),
            shards=(case["shape"],) if case["sharded"] else None,
            dtype="uint8",
            compressors=None,
            fill_value=0,
            config={"write_empty_chunks": case["write_empty_chunks"]},
        )
        return store, arr

    store_a, arr_a = build()
    arr_a[case["sel"]] = case["scalar"]

    store_b, arr_b = build()
    n = case["sel"].stop - case["sel"].start
    arr_b[case["sel"]] = np.full(n, case["scalar"], dtype="uint8")

    keys_a = {k: bytes(v.to_bytes()) for k, v in store_a._store_dict.items()}
    keys_b = {k: bytes(v.to_bytes()) for k, v in store_b._store_dict.items()}
    assert keys_a == keys_b


# ---------------------------------------------------------------------------
# get_ranges_sync coalescing: merged fetches must return exactly what
# individual per-range gets return, for any gap/coalesce limits.
# ---------------------------------------------------------------------------


@st.composite
def _range_cases(draw: st.DrawFn) -> dict[str, Any]:
    blob_len = draw(st.integers(1, 200))
    n = draw(st.integers(1, 8))
    ranges: list[RangeByteRequest | OffsetByteRequest | SuffixByteRequest | None] = []
    for _ in range(n):
        kind = draw(st.sampled_from(["range", "offset", "suffix", "none"]))
        if kind == "range":
            start = draw(st.integers(0, blob_len - 1))
            end = draw(st.integers(start + 1, blob_len))
            ranges.append(RangeByteRequest(start, end))
        elif kind == "offset":
            ranges.append(OffsetByteRequest(draw(st.integers(0, blob_len - 1))))
        elif kind == "suffix":
            ranges.append(SuffixByteRequest(draw(st.integers(1, blob_len))))
        else:
            ranges.append(None)
    return {
        "blob": draw(st.binary(min_size=blob_len, max_size=blob_len)),
        "ranges": ranges,
        "max_gap": draw(st.integers(0, 64)),
        "max_coalesced": draw(st.integers(1, 512)),
    }


@settings(max_examples=200, deadline=None)
@given(case=_range_cases())
def test_get_ranges_sync_equals_individual_gets(case: dict[str, Any]) -> None:
    """Coalesced byte-range reads must return exactly what one get_sync per
    range returns — for any gap/coalesce limits (the offset re-slicing math is
    where a coalescing bug would corrupt data)."""
    store = MemoryStore()
    store._is_open = True
    proto = default_buffer_prototype()
    store._store_dict["k"] = CPUBuffer.from_bytes(case["blob"])

    expected = [store.get_sync("k", prototype=proto, byte_range=r) for r in case["ranges"]]

    got: dict[int, bytes | None] = {}
    for idx, buf in store.get_ranges_sync(
        "k",
        case["ranges"],
        prototype=proto,
        max_gap_bytes=case["max_gap"],
        max_coalesced_bytes=case["max_coalesced"],
    ):
        got[idx] = None if buf is None else bytes(buf.to_bytes())

    for i, exp in enumerate(expected):
        exp_bytes = None if exp is None else bytes(exp.to_bytes())
        assert got.get(i) == exp_bytes, f"range {i} ({case['ranges'][i]!r}) mismatch"
