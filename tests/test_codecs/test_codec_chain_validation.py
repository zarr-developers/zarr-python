"""Validation of codec chains in which an earlier array->array codec changes the
shape or rank of a chunk.

The ``reshape`` extension codec (zarr-extensions) is not implemented in
zarr-python, so a minimal test double is used. Its README explicitly allows
combining ``reshape`` with ``transpose`` to both reorder and reshape; the
``transpose`` order then refers to the *reshaped* rank, so validating it against
the array-level shape must not reject the chain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import pytest

import zarr
from zarr.abc.codec import ArrayArrayCodec
from zarr.codecs import BytesCodec, ShardingCodec, TransposeCodec
from zarr.core.dtype import Int32
from zarr.core.metadata.v3 import (
    ArrayV3Metadata,
    RectilinearChunkGridMetadata,
    RegularChunkGridMetadata,
)
from zarr.registry import _codec_registries, register_codec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.common import JSON


@dataclass(frozen=True)
class ReshapeCodec(ArrayArrayCodec):
    """Minimal stand-in for the zarr-extensions ``reshape`` codec.

    Reshapes every chunk to the explicit ``shape`` (which therefore only makes
    sense for a regular chunk grid whose chunks all have the same size).
    """

    shape: tuple[int, ...]
    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        config = cast("dict[str, Any]", data["configuration"])
        return cls(shape=tuple(config["shape"]))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "reshape", "configuration": {"shape": list(self.shape)}}

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if np.prod(chunk_spec.shape) != np.prod(self.shape):
            raise ValueError(f"cannot reshape a chunk of shape {chunk_spec.shape} to {self.shape}")
        return replace(chunk_spec, shape=self.shape)

    async def _decode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        return chunk_array.reshape(chunk_spec.shape)

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        return chunk_array.reshape(self.shape)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


@pytest.fixture(autouse=True)
def _register_reshape() -> Iterator[None]:
    previous = _codec_registries.get("reshape")
    register_codec("reshape", ReshapeCodec)
    try:
        yield
    finally:
        _codec_registries.pop("reshape", None)
        if previous is not None:
            _codec_registries["reshape"] = previous


SHAPE = (4, 6, 8)
CHUNKS = (2, 3, 4)
# chunk (2, 3, 4) -> (2, 3, 2, 2), then transpose with a rank-4 order
RESHAPE_THEN_TRANSPOSE = (ReshapeCodec(shape=(2, 3, 2, 2)), TransposeCodec(order=(0, 2, 1, 3)))


@pytest.mark.parametrize("shards", [None, SHAPE, (2, 6, 8)])
def test_rank_changing_chain_roundtrip(shards: tuple[int, ...] | None) -> None:
    """A reshape+transpose chain is accepted, both standalone and as the inner
    codecs of a sharding codec, and round-trips data byte-for-byte."""
    data = np.arange(np.prod(SHAPE), dtype="i4").reshape(SHAPE)
    a = zarr.create_array(
        {},
        shape=SHAPE,
        chunks=CHUNKS,
        shards=shards,
        dtype="i4",
        filters=RESHAPE_THEN_TRANSPOSE,
    )
    a[:] = data
    assert np.array_equal(a[:], data)

    # The persisted metadata must be re-loadable, i.e. the same validation
    # must pass when the codecs come from JSON rather than from instances.
    reloaded = zarr.open_array(a.store, mode="r")
    assert reloaded.metadata == a.metadata
    assert np.array_equal(reloaded[:], data)


def _metadata(codecs: tuple[Any, ...], chunk_shape: tuple[int, ...] = CHUNKS) -> ArrayV3Metadata:
    return ArrayV3Metadata(
        shape=SHAPE,
        data_type=Int32(),
        chunk_grid=RegularChunkGridMetadata(chunk_shape=chunk_shape),
        chunk_key_encoding={"name": "default"},
        fill_value=0,
        codecs=codecs,
        attributes=None,
        dimension_names=None,
    )


def test_transpose_validated_against_reshaped_rank() -> None:
    """After a rank-changing codec, transpose is validated against the new rank:
    an order of the *original* rank is now the invalid one."""
    with pytest.raises(ValueError, match="`order` tuple must have as many entries"):
        _metadata((ReshapeCodec(shape=(2, 3, 2, 2)), TransposeCodec(order=(2, 1, 0)), BytesCodec()))


def test_reshape_validated_against_chunk_shape() -> None:
    """The chunk spec, not the array shape, is threaded through resolve_metadata:
    a reshape whose size matches the array but not the chunk is rejected."""
    with pytest.raises(ValueError, match="cannot reshape a chunk of shape"):
        _metadata((ReshapeCodec(shape=(4, 6, 8)), BytesCodec()))


def test_sharding_inner_chain_is_validated() -> None:
    """``ShardingCodec.validate`` validates its inner chain against the inner
    chunk shape, threading the spec through rank-changing codecs."""
    grid = RegularChunkGridMetadata(chunk_shape=SHAPE)
    ok = ShardingCodec(chunk_shape=CHUNKS, codecs=RESHAPE_THEN_TRANSPOSE)
    ok.validate(shape=SHAPE, dtype=Int32(), chunk_grid=grid)

    bad = ShardingCodec(
        chunk_shape=CHUNKS,
        codecs=(ReshapeCodec(shape=(2, 3, 2, 2)), TransposeCodec(order=(2, 1, 0))),
    )
    with pytest.raises(ValueError, match="`order` tuple must have as many entries"):
        bad.validate(shape=SHAPE, dtype=Int32(), chunk_grid=grid)
    with pytest.raises(ValueError, match="`order` tuple must have as many entries"):
        _metadata((bad,), chunk_shape=SHAPE)


def _rectilinear_transpose_sharding_metadata(inner: tuple[int, int]) -> ArrayV3Metadata:
    """Rectilinear grid (chunks (4,5) and (6,5)), transposed, then sharded."""
    return ArrayV3Metadata(
        shape=(10, 5),
        data_type=Int32(),
        chunk_grid=RectilinearChunkGridMetadata(chunk_shapes=((4, 6), 5)),
        chunk_key_encoding={"name": "default"},
        fill_value=0,
        codecs=(TransposeCodec(order=(1, 0)), ShardingCodec(chunk_shape=inner)),
        attributes=None,
        dimension_names=None,
    )


def test_rectilinear_every_chunk_shape_validated() -> None:
    """Under a rectilinear grid, size-sensitive validation after a
    shape-changing codec must consider every distinct chunk shape, not a single
    representative: an inner shard size dividing the largest transposed chunk
    (5,6) but not the smaller (5,4) is rejected."""
    with zarr.config.set({"array.rectilinear_chunks": True}):
        with pytest.raises(ValueError, match="not\\s+divisible"):
            _rectilinear_transpose_sharding_metadata((5, 3))
        # an inner shape dividing both transposed chunk shapes is accepted
        _rectilinear_transpose_sharding_metadata((5, 2))
