"""Property-based tests for codec-chain validation with shape-changing codecs.

Acceptance implies round-trip: any reshape+transpose chain that metadata
validation accepts must encode and decode data losslessly (and its metadata
must survive JSON serialization), while a transpose order of the wrong rank
must be rejected. A fill-changing inner codec must be validated against the
actual fill value. (Transpose ahead of sharding, regular and nested, is
covered by `test_transposed_sharding_chain_validation` in `test_properties.py`.)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

import zarr
from zarr.codecs import TransposeCodec
from zarr.core.metadata.v3 import ArrayV3Metadata
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


@given(offset=st.integers(min_value=1, max_value=254), sharded=st.booleans())
def test_inner_validation_uses_the_actual_fill_value(offset: int, sharded: bool) -> None:
    """Shard validation must not resolve a fill-changing codec against a made-up zero."""
    from zarr.codecs.scale_offset import ScaleOffset

    array = zarr.create_array(
        {},
        shape=(8,),
        chunks=(2,),
        shards=(4,) if sharded else None,
        dtype="u1",
        fill_value=offset,
        filters=[ScaleOffset(offset=offset)],
    )
    array[:] = offset + 1
    assert np.array_equal(array[:], np.full((8,), offset + 1, dtype="u1"))
    reloaded = zarr.open_array(array.store, mode="r")
    assert np.array_equal(reloaded[:], array[:])
