"""Validation of codec chains in which an earlier array->array codec changes the
shape or rank of a chunk.

The ``reshape`` extension codec (zarr-extensions) is not implemented in
zarr-python, so a minimal test double is used. Its README explicitly allows
combining ``reshape`` with ``transpose`` to both reorder and reshape; the
``transpose`` order then refers to the *reshaped* rank, so validating it against
the array-level shape must not reject the chain.

Geometry is threaded through the chain as a whole chunk grid (see
`evolve_and_validate_codecs`). On a rectilinear grid, a codec that does not
declare its grid (`resolve_chunk_grid`) makes the rest of the chain
"chunk-local": only a representative chunk is validated when the metadata is
created, and other chunk shapes are checked when they are encoded or decoded.
The tests below pin both sides of that trade-off.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import pytest

import zarr
from zarr.abc.codec import ArrayArrayCodec, Codec
from zarr.codecs import BytesCodec, ShardingCodec, TransposeCodec
from zarr.codecs.numcodecs import Delta
from zarr.codecs.scale_offset import ScaleOffset
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import Int32
from zarr.core.metadata.v3 import (
    ArrayV3Metadata,
    ChunkGridMetadata,
    RectilinearChunkGridMetadata,
    RegularChunkGridMetadata,
    _declared_chunk_grid,
)
from zarr.registry import _codec_registries, register_codec

if TYPE_CHECKING:
    from collections.abc import Iterator

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


@dataclass(frozen=True)
class OpaqueCodec(ArrayArrayCodec):
    """A pass-through filter that overrides `resolve_metadata` but does not
    declare its chunk grid, as a typical third-party codec would. Metadata
    validation cannot tell that it preserves chunk shape."""

    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls()

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "opaque"}

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return chunk_spec

    async def _decode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        return chunk_array

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        return chunk_array

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


@pytest.fixture(autouse=True)
def _register_test_codecs() -> Iterator[None]:
    names = {"reshape": ReshapeCodec, "opaque": OpaqueCodec}
    previous = {name: _codec_registries.get(name) for name in names}
    for name, codec_cls in names.items():
        register_codec(name, codec_cls)
    try:
        yield
    finally:
        for name, registry in previous.items():
            _codec_registries.pop(name, None)
            if registry is not None:
                _codec_registries[name] = registry


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


@pytest.mark.parametrize(
    "chunks",
    [(2, 4), [[2, 2], 4], [[2, 2], [4]]],
    ids=["regular", "rectilinear-mixed", "rectilinear-explicit"],
)
def test_pipeline_evolves_against_representative_chunk(
    chunks: tuple[int, ...] | list[Any],
) -> None:
    """The codec pipeline is evolved against the same representative chunk shape
    that metadata validation used, for regular and rectilinear grids alike. A
    reshape whose size matches every (2, 4) chunk but not a placeholder chunk
    must be accepted at array creation, and the evolved pipeline must carry the
    codecs metadata validation produced."""
    data = np.arange(16, dtype="i4").reshape(4, 4)
    with zarr.config.set({"array.rectilinear_chunks": True}):
        a = zarr.create_array(
            {},
            shape=(4, 4),
            chunks=chunks,
            dtype="i4",
            filters=[ReshapeCodec(shape=(8,))],
            compressors=None,
        )
    pipeline = a._async_array.codec_pipeline
    assert isinstance(pipeline, Iterable)
    assert isinstance(a.metadata, ArrayV3Metadata)
    assert tuple(pipeline) == a.metadata.codecs
    a[:] = data
    assert np.array_equal(a[:], data)


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
    """Metadata construction validates a sharding codec's inner chain against
    the inner chunk spec (the accepted case is `test_rank_changing_chain_roundtrip`
    with `shards`). Direct `ShardingCodec.validate` only has geometry and dtype,
    not the fill value needed to resolve arbitrary inner codecs, so the inner
    chain is validated during evolution, which has the full spec."""
    bad = ShardingCodec(
        chunk_shape=CHUNKS,
        codecs=(ReshapeCodec(shape=(2, 3, 2, 2)), TransposeCodec(order=(2, 1, 0))),
    )
    with pytest.raises(ValueError, match="`order` tuple must have as many entries"):
        _metadata((bad,), chunk_shape=SHAPE)


def _rectilinear_sharding_metadata(
    filters: tuple[Any, ...], inner: tuple[int, int]
) -> ArrayV3Metadata:
    """Rectilinear grid (chunks (4,5) and (6,5)), `filters`, then sharded."""
    return ArrayV3Metadata(
        shape=(10, 5),
        data_type=Int32(),
        chunk_grid=RectilinearChunkGridMetadata(chunk_shapes=((4, 6), 5)),
        chunk_key_encoding={"name": "default"},
        fill_value=0,
        codecs=(*filters, ShardingCodec(chunk_shape=inner)),
        attributes=None,
        dimension_names=None,
    )


TRANSPOSE = (TransposeCodec(order=(1, 0)),)
DELTA_TRANSPOSE = (Delta(dtype="<i4"), TransposeCodec(order=(1, 0)))


@pytest.mark.parametrize("filters", [TRANSPOSE, DELTA_TRANSPOSE], ids=["transpose", "delta"])
@pytest.mark.parametrize("inner", [(5, 2), (5, 1), (1, 2)])
def test_rectilinear_shape_change_accepts_inner_dividing_every_chunk(
    filters: tuple[Any, ...], inner: tuple[int, int]
) -> None:
    """After a shape-changing codec on a rectilinear grid, an inner shard shape
    dividing every transposed chunk shape ((5,4) and (5,6)) is accepted."""
    with zarr.config.set({"array.rectilinear_chunks": True}):
        meta = _rectilinear_sharding_metadata(filters, inner)
        assert ArrayV3Metadata.from_dict(meta.to_dict()) == meta


@pytest.mark.parametrize("filters", [TRANSPOSE, DELTA_TRANSPOSE], ids=["transpose", "delta"])
def test_rectilinear_every_chunk_shape_validated(filters: tuple[Any, ...]) -> None:
    """When every codec before the sharding codec declares its chunk grid,
    validation covers every distinct chunk shape, not a single representative:
    an inner shard size dividing the largest transposed chunk (5,6) but not the
    smaller (5,4) is rejected when the metadata is created."""
    with (
        zarr.config.set({"array.rectilinear_chunks": True}),
        pytest.raises(ValueError, match="not\\s+divisible"),
    ):
        _rectilinear_sharding_metadata(filters, (5, 3))


def test_chunk_local_rejects_invalid_representative_chunk() -> None:
    """After an undeclared codec on a rectilinear grid, the representative
    chunk (the largest edge per axis, (6,5)) is still validated when the
    metadata is created, so a shard size that fails it is rejected."""
    with (
        zarr.config.set({"array.rectilinear_chunks": True}),
        pytest.raises(ValueError, match="not\\s+divisible"),
    ):
        _rectilinear_sharding_metadata((OpaqueCodec(),), (4, 5))


@pytest.mark.filterwarnings(
    "ignore:Combining a `sharding_indexed` codec:zarr.errors.ZarrUserWarning"
)
def test_chunk_local_defers_other_shard_shapes_to_run_time() -> None:
    """The documented cost of chunk-local validation: after an undeclared codec
    on a rectilinear grid, an inner shard size dividing the representative
    chunk (6,5) but not the smaller (4,5) is accepted when the array is created.
    The non-dividing shard is rejected when it is first encoded, instead of
    being floor-divided and silently written with the wrong layout, and the
    dividing shard still round-trips."""
    with zarr.config.set({"array.rectilinear_chunks": True}):
        array = zarr.create_array(
            {},
            shape=(10, 5),
            chunks=[[4, 6], [5]],
            dtype="i4",
            filters=[OpaqueCodec()],
            serializer=ShardingCodec(chunk_shape=(3, 5)),
            compressors=None,
        )
        array[4:] = 1
        np.testing.assert_array_equal(array[4:], np.ones((6, 5), dtype="i4"))
        with pytest.raises(ValueError, match="not divisible by the shard's inner chunk shape"):
            array[:4] = 1


def test_chunk_local_does_not_enumerate_to_find_codec_errors() -> None:
    """A shape-changing undeclared codec that accepts the representative chunk
    ((2,2) -> (4,)) but would reject a smaller one ((1,1)) is accepted when the
    metadata is created: validation does not enumerate the chunks of the grid
    to find the failing one. The codec is left to reject that chunk itself when
    it is encoded or decoded."""
    with zarr.config.set({"array.rectilinear_chunks": True}):
        ArrayV3Metadata(
            shape=(3, 3),
            data_type=Int32(),
            chunk_grid=RectilinearChunkGridMetadata(chunk_shapes=((1, 2), (1, 2))),
            chunk_key_encoding={"name": "default"},
            fill_value=0,
            codecs=(ReshapeCodec(shape=(4,)), BytesCodec()),
            attributes={},
            dimension_names=None,
        )


class _ReshapingDelta(Delta):
    """Overrides the resolver of a codec that declares an identity grid."""

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return replace(chunk_spec, shape=chunk_spec.shape[::-1])


class _MisdeclaredCodec(OpaqueCodec):
    """Declares an identity grid while `resolve_metadata` reverses the axes."""

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return replace(chunk_spec, shape=chunk_spec.shape[::-1])

    def resolve_chunk_grid(
        self, *, shape: tuple[int, ...], chunk_grid: ChunkGridMetadata
    ) -> tuple[tuple[int, ...], ChunkGridMetadata]:
        return shape, chunk_grid


@pytest.mark.parametrize(
    ("codec", "expected"),
    [
        (BytesCodec(), ((10, 5), ((4, 6), 5))),
        (Delta(dtype="<i4"), ((10, 5), ((4, 6), 5))),
        (ScaleOffset(offset=1), ((10, 5), ((4, 6), 5))),
        (TransposeCodec(order=(1, 0)), ((5, 10), (5, (4, 6)))),
        (OpaqueCodec(), None),
        (_ReshapingDelta(dtype="<i4"), None),
        (_MisdeclaredCodec(), None),
    ],
    ids=["default", "delta", "scale-offset", "transpose", "undeclared", "subclass", "misdeclared"],
)
def test_declared_chunk_grid(
    codec: Codec,
    expected: tuple[tuple[int, ...], tuple[int | tuple[int, ...], ...]] | None,
) -> None:
    """A codec's declared grid is used only when it is trustworthy: declared
    by the class whose resolver is in effect, and consistent with that
    resolver on the representative chunk."""
    spec = ArraySpec(
        shape=(6, 5),
        dtype=Int32(),
        fill_value=0,
        config=ArrayConfig.from_dict({}),
        prototype=default_buffer_prototype(),
    )
    with zarr.config.set({"array.rectilinear_chunks": True}):
        grid = RectilinearChunkGridMetadata(chunk_shapes=((4, 6), 5))
        declared = _declared_chunk_grid(
            codec, shape=(10, 5), chunk_grid=grid, resolved_spec=codec.resolve_metadata(spec)
        )
        if expected is None:
            assert declared is None
        else:
            shape, chunk_shapes = expected
            assert declared == (shape, RectilinearChunkGridMetadata(chunk_shapes=chunk_shapes))


CHAINS: dict[str, Any] = {
    "bytes": lambda rank: (BytesCodec(),),
    "scale-offset": lambda rank: (ScaleOffset(offset=0), BytesCodec()),
    "delta": lambda rank: (Delta(dtype="<i4"), BytesCodec()),
    "transpose-sharding": lambda rank: (
        TransposeCodec(order=tuple(reversed(range(rank)))),
        ShardingCodec(chunk_shape=(1,) * rank),
    ),
    "undeclared-sharding": lambda rank: (OpaqueCodec(), ShardingCodec(chunk_shape=(1,) * rank)),
}


@pytest.mark.parametrize("rank", [4, 12])
@pytest.mark.parametrize("chain", list(CHAINS))
def test_validation_does_not_visit_chunk_combinations(
    rank: int, chain: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation cost is independent of the number of chunk shapes. The grid
    has 2**rank distinct chunk shapes (4096 at rank 12); resolving metadata once
    per codec, rather than once per chunk shape, keeps the call count small for
    declared, default, and undeclared codecs alike."""
    codecs = CHAINS[chain](rank)
    calls = 0

    def count(resolver: Any) -> Any:
        def counted(self: Any, spec: ArraySpec) -> ArraySpec:
            nonlocal calls
            calls += 1
            assert calls <= 10, "metadata validation enumerated chunk combinations"
            return cast("ArraySpec", resolver(self, spec))

        return counted

    # Patch each resolver on the class that defines it, so that which class
    # defines `resolve_metadata` (and hence which grid declaration is trusted)
    # is unchanged.
    owners = {next(k for k in type(c).__mro__ if "resolve_metadata" in k.__dict__) for c in codecs}
    for owner in owners:
        monkeypatch.setattr(owner, "resolve_metadata", count(owner.__dict__["resolve_metadata"]))
    with zarr.config.set({"array.rectilinear_chunks": True}):
        ArrayV3Metadata(
            shape=(3,) * rank,
            data_type=Int32(),
            chunk_grid=RectilinearChunkGridMetadata(chunk_shapes=((1, 2),) * rank),
            chunk_key_encoding={"name": "default"},
            fill_value=0,
            codecs=codecs,
            attributes={},
            dimension_names=None,
        )
    assert 0 < calls <= 10
