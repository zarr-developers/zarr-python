from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from zarr.core.config import config as zarr_config
from zarr.core.indexing import BasicIndexer
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from zarr.abc.codec import Codec


@pytest.fixture(autouse=True)
def _enable_rectilinear_chunks() -> Generator[None]:
    """Enable rectilinear chunks for all tests in this module."""
    with zarr_config.set({"array.rectilinear_chunks": True}):
        yield


pipeline_paths = [
    "zarr.core.codec_pipeline.BatchedCodecPipeline",
    "zarr.core.codec_pipeline.FusedCodecPipeline",
]


@pytest.fixture(params=pipeline_paths, ids=["batched", "sync"])
def pipeline_class(request: pytest.FixtureRequest) -> Generator[str]:
    """Temporarily set the codec pipeline class for the test."""
    path = request.param
    with zarr_config.set({"codec_pipeline.path": path}):
        yield path


# ---------------------------------------------------------------------------
# GetResult status tests (low-level pipeline API)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("write_slice", "read_slice", "expected_statuses"),
    [
        (slice(None), slice(None), ("present", "present", "present")),
        (slice(0, 2), slice(None), ("present", "missing", "missing")),
        (None, slice(None), ("missing", "missing", "missing")),
    ],
)
async def test_read_returns_get_results(
    pipeline_class: str,
    write_slice: slice | None,
    read_slice: slice,
    expected_statuses: tuple[str, ...],
) -> None:
    """CodecPipeline.read returns GetResult with correct statuses."""
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


# ---------------------------------------------------------------------------
# write_empty_chunks / read_missing_chunks config tests
# ---------------------------------------------------------------------------


async def test_write_empty_chunks_false_no_store(pipeline_class: str) -> None:
    """With write_empty_chunks=False, fill_value-only chunks should not be stored."""
    store: dict[str, Any] = {}
    arr = zarr.create_array(
        store=store,
        shape=(20,),
        dtype="float64",
        chunks=(10,),
        shards=None,
        compressors=None,
        fill_value=0.0,
        config={"write_empty_chunks": False},
    )
    arr[:] = 0.0  # all fill_value

    # Chunks should NOT be persisted
    assert "c/0" not in store
    assert "c/1" not in store

    # But reading should still return fill values
    np.testing.assert_array_equal(arr[:], np.zeros(20, dtype="float64"))


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


def test_evolve_threads_spec_preserving_serializer_endian(pipeline_class: str) -> None:
    """Regression for #3937, dependency-free variant.

    `evolve_from_array_spec` must thread the spec FORWARD through the codec chain:
    each codec is evolved against the spec produced by the previous one, not the
    original array spec. An array->array codec that widens the dtype from a
    single-byte type (no endianness) to a multi-byte type means the BytesCodec
    serializer must be evolved against the *widened* dtype — otherwise it sees
    the single-byte source, strips its `endian` to None, and later fails to
    decode the multi-byte data.

    The original regression test for this needs `cast_value_rs` (so it only runs
    in the optional-deps CI job). This variant uses a minimal dtype-widening AA
    codec stub, so it runs everywhere and on both pipelines via `pipeline_class`.
    """
    from dataclasses import dataclass

    from zarr.abc.codec import ArrayArrayCodec
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.dtype import get_data_type_from_native_dtype
    from zarr.registry import get_pipeline_class

    @dataclass(frozen=True)
    class _WidenToInt16(ArrayArrayCodec):
        """Test-only AA codec: reports the encoded dtype as int16 (no real encode)."""

        is_fixed_size = True

        def to_dict(self) -> dict[str, Any]:
            return {"name": "_widen_to_int16"}

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> _WidenToInt16:
            return cls()

        def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
            from dataclasses import replace

            return replace(chunk_spec, dtype=get_data_type_from_native_dtype(np.dtype("int16")))

        def compute_encoded_size(self, input_byte_length: int, _spec: ArraySpec) -> int:
            return input_byte_length

        async def _decode_single(self, chunk_array: Any, chunk_spec: ArraySpec) -> Any:
            return chunk_array  # pragma: no cover

        async def _encode_single(self, chunk_array: Any, chunk_spec: ArraySpec) -> Any:
            return chunk_array  # pragma: no cover

    zdtype = get_data_type_from_native_dtype(np.dtype("int8"))  # single-byte source
    spec = ArraySpec(
        shape=(4,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=False),
        prototype=default_buffer_prototype(),
    )

    from zarr.core.codec_pipeline import BatchedCodecPipeline, FusedCodecPipeline

    pipeline = get_pipeline_class().from_codecs((_WidenToInt16(), BytesCodec(endian="little")))
    evolved = pipeline.evolve_from_array_spec(spec)
    # Both concrete pipelines expose `array_bytes_codec`; narrow off the ABC.
    assert isinstance(evolved, (BatchedCodecPipeline, FusedCodecPipeline))
    serializer = evolved.array_bytes_codec

    # The serializer must keep its little-endian setting: it is evolved against
    # the widened (int16) dtype, not the single-byte source.
    assert isinstance(serializer, BytesCodec)
    assert serializer.endian is not None, (
        "BytesCodec serializer lost its `endian` — evolve_from_array_spec did not "
        "thread the dtype-widening AA codec's spec into the serializer"
    )


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
