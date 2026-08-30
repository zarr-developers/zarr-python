"""Property-based tests for codec-chain validation with shape-changing codecs.

Two invariants are tested against explicit oracles:

1. Acceptance implies round-trip: any reshape+transpose chain that metadata
   validation accepts must encode and decode data losslessly (and its metadata
   must survive JSON serialization), while a transpose order of the wrong rank
   must be rejected.

2. For a rectilinear grid followed by a shape-changing codec and a
   size-sensitive codec (sharding), acceptance must exactly equal the oracle
   "every chunk shape in the grid, transformed by the chain, satisfies the
   size constraint" — not just the largest chunk (see
   ``evolve_and_validate_codecs``).
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

import zarr
from zarr.codecs import ShardingCodec, TransposeCodec
from zarr.core.dtype import Int32
from zarr.core.metadata.v3 import ArrayV3Metadata, RectilinearChunkGridMetadata
from zarr.registry import _codec_registries, register_codec

from .test_codec_chain_validation import ReshapeCodec

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
from hypothesis import given, settings


@pytest.fixture(scope="module", autouse=True)
def _register_reshape() -> Iterator[None]:
    previous = _codec_registries.get("reshape")
    register_codec("reshape", ReshapeCodec)
    try:
        yield
    finally:
        _codec_registries.pop("reshape", None)
        if previous is not None:
            _codec_registries["reshape"] = previous


@st.composite
def reshape_transpose_cases(
    draw: st.DrawFn,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...] | None, tuple[int, ...]]:
    """(array shape, chunk shape, shard shape or None, reshape target).

    The reshape target is a valid per-chunk factorization: each chunk dimension
    is either kept or split into two factors, so the target always has the same
    total size as the chunk but generally a different rank.
    """
    ndim = draw(st.integers(min_value=1, max_value=3))
    chunks = tuple(draw(st.integers(min_value=1, max_value=4)) for _ in range(ndim))
    if draw(st.booleans()):
        shards = tuple(c * draw(st.integers(min_value=1, max_value=2)) for c in chunks)
    else:
        shards = None
    outer = shards if shards is not None else chunks
    shape = tuple(o * draw(st.integers(min_value=1, max_value=2)) for o in outer)
    target: list[int] = []
    for c in chunks:
        if draw(st.booleans()):
            divisor = draw(st.sampled_from([d for d in range(1, c + 1) if c % d == 0]))
            target.extend([divisor, c // divisor])
        else:
            target.append(c)
    return shape, chunks, shards, tuple(target)


@settings(deadline=None)
@given(case=reshape_transpose_cases(), data=st.data())
def test_accepted_reshape_transpose_chain_roundtrips(
    case: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...] | None, tuple[int, ...]],
    data: st.DataObject,
) -> None:
    """A reshape to any valid chunk factorization, followed by a transpose with
    any permutation of the reshaped rank, is accepted and round-trips."""
    shape, chunks, shards, target = case
    order = tuple(data.draw(st.permutations(range(len(target))), label="order"))
    arr = zarr.create_array(
        {},
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype="i4",
        filters=[ReshapeCodec(shape=target), TransposeCodec(order=order)],
    )
    expected = np.arange(math.prod(shape), dtype="i4").reshape(shape)
    arr[:] = expected
    assert np.array_equal(arr[:], expected)
    # validation must be stable across JSON serialization
    assert ArrayV3Metadata.from_dict(arr.metadata.to_dict()) == arr.metadata


@settings(deadline=None)
@given(case=reshape_transpose_cases(), data=st.data())
def test_wrong_rank_transpose_after_reshape_rejected(
    case: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...] | None, tuple[int, ...]],
    data: st.DataObject,
) -> None:
    """A transpose order whose rank differs from the reshaped rank is rejected."""
    shape, chunks, shards, target = case
    wrong_rank = data.draw(
        st.integers(min_value=1, max_value=len(target) + 2).filter(lambda n: n != len(target)),
        label="wrong_rank",
    )
    order = tuple(data.draw(st.permutations(range(wrong_rank)), label="order"))
    with pytest.raises(ValueError, match="order"):
        zarr.create_array(
            {},
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype="i4",
            filters=[ReshapeCodec(shape=target), TransposeCodec(order=order)],
        )


@st.composite
def rectilinear_transpose_sharding_cases(
    draw: st.DrawFn,
) -> tuple[tuple[int | tuple[int, ...], ...], tuple[int, ...], tuple[int, ...]]:
    """(rectilinear chunk_shapes, transpose order, inner shard shape)."""
    ndim = draw(st.integers(min_value=2, max_value=3))
    chunk_shapes: list[int | tuple[int, ...]] = []
    for _ in range(ndim):
        edges = draw(st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=3))
        # exercise the bare-int (uniform edge) spelling as well
        if len(edges) == 1 and draw(st.booleans()):
            chunk_shapes.append(edges[0])
        else:
            chunk_shapes.append(tuple(edges))
    order = tuple(draw(st.permutations(range(ndim))))
    inner = tuple(draw(st.integers(min_value=1, max_value=6)) for _ in range(ndim))
    return tuple(chunk_shapes), order, inner


@settings(deadline=None)
@given(case=rectilinear_transpose_sharding_cases())
def test_rectilinear_transpose_sharding_matches_oracle(
    case: tuple[tuple[int | tuple[int, ...], ...], tuple[int, ...], tuple[int, ...]],
) -> None:
    """transpose-then-shard over a rectilinear grid is accepted exactly when
    every transposed chunk shape is divisible by the inner shard shape."""
    chunk_shapes, order, inner = case
    per_dim = tuple((e,) if isinstance(e, int) else e for e in chunk_shapes)
    oracle_ok = all(
        all(chunk[order[i]] % inner[i] == 0 for i in range(len(inner)))
        for chunk in itertools.product(*per_dim)
    )
    # array shape: bare-int (uniform) edges cover any extent; explicit edge
    # lists must sum to at least the extent.
    shape = tuple(e if isinstance(e, int) else sum(e) for e in chunk_shapes)

    def build() -> ArrayV3Metadata:
        return ArrayV3Metadata(
            shape=shape,
            data_type=Int32(),
            chunk_grid=RectilinearChunkGridMetadata(chunk_shapes=chunk_shapes),
            chunk_key_encoding={"name": "default"},
            fill_value=0,
            codecs=(TransposeCodec(order=order), ShardingCodec(chunk_shape=inner)),
            attributes=None,
            dimension_names=None,
        )

    with zarr.config.set({"array.rectilinear_chunks": True}):
        if oracle_ok:
            meta = build()
            assert ArrayV3Metadata.from_dict(meta.to_dict()) == meta
        else:
            with pytest.raises(ValueError, match="not\\s+divisible"):
                build()
