"""Codec / shard / buffer invariants.

These tests enforce the contracts described in
``docs/superpowers/specs/2026-04-17-codec-pipeline-invariants.md``.
They exist to catch the class of bug where pipeline code reasons
case-by-case about how codecs, shards, IO, and buffers interact and
silently breaks a combination.

Each test is short and focused on one invariant. If any test here
fails, the corresponding section of the design doc points at what
contract was broken.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

import zarr
from zarr.abc.codec import BytesBytesCodec, Codec
from zarr.abc.store import SupportsSetRange
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.core.codec_pipeline import ChunkTransform, FusedCodecPipeline
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.storage import LocalStore, MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(
    shape: tuple[int, ...] = (10,),
    dtype: str = "float64",
    *,
    fill_value: object = 0.0,
    write_empty_chunks: bool = False,
) -> ArraySpec:
    zdtype = get_data_type_from_native_dtype(np.dtype(dtype))
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(fill_value),
        config=ArrayConfig(order="C", write_empty_chunks=write_empty_chunks),
        prototype=default_buffer_prototype(),
    )


# ---------------------------------------------------------------------------
# C1: Codecs only mutate `shape`
# ---------------------------------------------------------------------------

# Codecs that we expect to satisfy C1 unconditionally. Each is in a
# state where calling resolve_metadata is safe with the helper spec.
_C1_CODECS: list[Codec] = [
    BytesCodec(),
    Crc32cCodec(),
    GzipCodec(level=1),
    ZstdCodec(level=1),
    TransposeCodec(order=(0,)),
]


@pytest.mark.parametrize("codec", _C1_CODECS, ids=lambda c: type(c).__name__)
def test_C1_resolve_metadata_only_mutates_shape(codec: Codec) -> None:
    """C1: prototype, dtype, fill_value, config never change across the codec chain."""
    spec_in = _spec()
    spec_out = codec.resolve_metadata(spec_in)
    assert spec_out.prototype is spec_in.prototype, f"{type(codec).__name__} changed prototype"
    assert spec_out.dtype == spec_in.dtype, f"{type(codec).__name__} changed dtype"
    assert spec_out.fill_value == spec_in.fill_value, f"{type(codec).__name__} changed fill_value"
    assert spec_out.config == spec_in.config, f"{type(codec).__name__} changed config"


# ---------------------------------------------------------------------------
# C2: Each codec call receives the runtime chunk_spec
# ---------------------------------------------------------------------------


class _PrototypeRecordingCodec(BytesBytesCodec):  # type: ignore[misc,unused-ignore]
    """A no-op BB codec that records the prototype it was called with."""

    is_fixed_size = True
    seen_prototypes: list[object]

    def __init__(self) -> None:
        object.__setattr__(self, "seen_prototypes", [])

    def to_dict(self) -> dict[str, Any]:
        return {"name": "_prototype_recording", "configuration": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _PrototypeRecordingCodec:
        return cls()

    def compute_encoded_size(self, input_byte_length: int, _spec: ArraySpec) -> int:
        return input_byte_length

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        self.seen_prototypes.append(chunk_spec.prototype)
        return chunk_bytes

    def _encode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
        self.seen_prototypes.append(chunk_spec.prototype)
        return chunk_bytes

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return self._decode_sync(chunk_bytes, chunk_spec)

    async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
        return self._encode_sync(chunk_bytes, chunk_spec)


def test_C2_chunk_transform_uses_runtime_prototype() -> None:
    """C2: the prototype the codec sees comes from the runtime chunk_spec, not a cache."""
    from zarr.core.buffer import BufferPrototype

    recording = _PrototypeRecordingCodec()
    transform = ChunkTransform(codecs=(BytesCodec(), recording))

    proto_default = default_buffer_prototype()
    # A distinct BufferPrototype instance with the same buffer/nd_buffer
    # types — fails identity check but works at runtime.
    proto_other = BufferPrototype(buffer=proto_default.buffer, nd_buffer=proto_default.nd_buffer)
    assert proto_other is not proto_default

    spec_a = replace(_spec(), prototype=proto_default)
    spec_b = replace(_spec(), prototype=proto_other)

    arr = proto_default.nd_buffer.from_numpy_array(np.arange(10, dtype="float64"))
    transform.encode_chunk(arr, spec_a)
    transform.encode_chunk(arr, spec_b)

    assert recording.seen_prototypes[0] is proto_default
    assert recording.seen_prototypes[1] is proto_other, (
        "ChunkTransform did not pass the runtime prototype to the codec"
    )


# ---------------------------------------------------------------------------
# C3: pipeline never branches on codec type
# ---------------------------------------------------------------------------


def test_C3_pipeline_methods_do_not_isinstance_check_sharding_codec() -> None:
    """C3: Pipeline read/write methods must use supports_partial_*, not isinstance(ShardingCodec).

    Static check: scan the pipeline classes' read/write methods for
    `isinstance(..., ShardingCodec)`. Other helpers (e.g. metadata
    validation in `codecs_from_list`) may legitimately need the check.
    """
    import inspect
    import re

    from zarr.core.codec_pipeline import BatchedCodecPipeline, FusedCodecPipeline

    pattern = re.compile(r"isinstance\s*\([^)]*ShardingCodec[^)]*\)")

    for cls in (FusedCodecPipeline, BatchedCodecPipeline):
        for method_name in ("read", "write", "read_sync", "write_sync"):
            method = getattr(cls, method_name, None)
            if method is None:
                continue
            source = inspect.getsource(method)
            matches = pattern.findall(source)
            assert not matches, (
                f"{cls.__name__}.{method_name} contains isinstance check on "
                f"ShardingCodec; use supports_partial_encode/decode instead. "
                f"Matches: {matches}"
            )


# ---------------------------------------------------------------------------
# S1 + S2: shard layout is compact and skips empty chunks by default
# ---------------------------------------------------------------------------


def test_S2_empty_chunks_omitted_under_default_config() -> None:
    """S2: writing fill-value data must not produce store keys for those chunks."""
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(20,),
        chunks=(10,),
        shards=None,
        dtype="float64",
        compressors=None,
        fill_value=0.0,
    )
    # Write fill values to the second chunk; assert no key created for it.
    arr[10:20] = 0.0
    assert "c/1" not in store._store_dict


def test_S2_empty_shard_deleted_after_partial_writes_to_fill() -> None:
    """S2: a sharded array where all inner chunks become fill should drop the shard."""
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(16,),
        chunks=(4,),
        shards=(8,),
        dtype="float64",
        compressors=None,
        fill_value=0.0,
    )
    # Fill the first shard with non-fill data, then overwrite back to fill.
    arr[0:8] = np.arange(8, dtype="float64") + 1
    assert "c/0" in store._store_dict
    arr[0:8] = 0.0
    assert "c/0" not in store._store_dict, "shard should be deleted when fully empty"


# ---------------------------------------------------------------------------
# S3: byte-range fast path requires write_empty_chunks=True
# ---------------------------------------------------------------------------


def _is_sync_pipeline_default() -> bool:
    """Check whether FusedCodecPipeline is the active pipeline."""
    store = MemoryStore()
    arr = zarr.create_array(store=store, shape=(8,), chunks=(8,), dtype="uint8", fill_value=0)
    return isinstance(arr._async_array.codec_pipeline, FusedCodecPipeline)


def test_S3_byte_range_path_skipped_when_write_empty_chunks_false() -> None:
    """S3: under default config, partial shard writes do not call set_range_sync."""
    if not _is_sync_pipeline_default():
        pytest.skip("byte-range fast path is specific to FusedCodecPipeline")

    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        chunks=(10,),
        shards=(100,),
        dtype="float64",
        compressors=None,
        fill_value=0.0,
        # Default config: write_empty_chunks=False
    )
    arr[:] = np.arange(100, dtype="float64")
    with patch.object(type(store), "set_range_sync", wraps=store.set_range_sync) as mock:
        arr[5] = 999.0
    assert mock.call_count == 0, (
        "byte-range fast path was taken with write_empty_chunks=False; "
        "this would produce a dense shard layout incompatible with empty-chunk skipping"
    )


def test_S3_byte_range_path_used_when_write_empty_chunks_true() -> None:
    """S3: with write_empty_chunks=True, partial shard writes use set_range_sync."""
    if not _is_sync_pipeline_default():
        pytest.skip("byte-range fast path is specific to FusedCodecPipeline")

    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        chunks=(10,),
        shards=(100,),
        dtype="float64",
        compressors=None,
        fill_value=0.0,
        config={"write_empty_chunks": True},
    )
    arr[:] = np.arange(100, dtype="float64")
    with patch.object(type(store), "set_range_sync", wraps=store.set_range_sync) as mock:
        arr[5] = 999.0
    assert mock.call_count >= 1, "byte-range fast path was not taken with write_empty_chunks=True"


# ---------------------------------------------------------------------------
# B1: code that mutates buffers from store IO must copy first
# ---------------------------------------------------------------------------


def test_B1_partial_shard_write_handles_readonly_store_buffers(tmp_path: Path) -> None:
    """B1: LocalStore returns read-only buffers; mutating-paths must copy."""
    store = LocalStore(tmp_path / "data.zarr")
    arr = zarr.create_array(
        store=store,
        shape=(16,),
        chunks=(4,),
        shards=(8,),
        dtype="float64",
        compressors=None,
        fill_value=0.0,
        config={"write_empty_chunks": True},
    )
    arr[:] = np.arange(16, dtype="float64")
    # This triggers the byte-range path which decodes the shard index from
    # a (potentially read-only) store buffer and then mutates it. If the
    # decode result isn't copied, the next line raises
    # `ValueError: assignment destination is read-only`.
    arr[2] = 42.0
    assert arr[2] == 42.0


# ---------------------------------------------------------------------------
# Sanity: SupportsSetRange is correctly implemented
# ---------------------------------------------------------------------------


def test_supports_set_range_is_runtime_checkable() -> None:
    """Stores should report SupportsSetRange membership via isinstance."""
    assert isinstance(MemoryStore(), SupportsSetRange)
