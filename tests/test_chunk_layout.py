"""Tests for the public ``ChunkLayout`` introspection API (design/chunk-layout.md)."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import pytest

import zarr
from zarr.codecs import BytesCodec, GzipCodec, ShardingCodec
from zarr.core.chunk_layouts import ChunkLayout
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Generator

    from zarr.core.array import ShardsLike
    from zarr.core.common import ChunksLike


class _CreateKwargs(TypedDict, total=False):
    """The subset of ``create_array`` keyword arguments these tests vary.

    A ``TypedDict`` (rather than a plain ``dict``) so ``**``-unpacking into
    ``create_array`` is checked per key against that parameter's type.
    """

    shape: tuple[int, ...]
    chunks: ChunksLike
    shards: ShardsLike | None


class _RecreateKwargs(TypedDict):
    """``chunks``/``shards`` reconstructed from a ``ChunkLayout`` (no ``shape``)."""

    chunks: ChunksLike
    shards: ShardsLike | None


@pytest.fixture(autouse=True)
def _enable_rectilinear_chunks() -> Generator[None, None, None]:
    with zarr.config.set({"array.rectilinear_chunks": True}):
        yield


# ---------------------------------------------------------------------------
# value semantics
# ---------------------------------------------------------------------------


def test_canonicalization_uniform_edges_collapse() -> None:
    """A uniform edge tuple normalizes to the bare int, so equal grids compare equal."""
    rect = ChunkLayout(chunks=((20, 20, 20),))
    reg = ChunkLayout(chunks=(20,))
    assert rect == reg
    assert hash(rect) == hash(reg)
    assert rect.is_regular


def test_is_regular_predicate() -> None:
    assert ChunkLayout(chunks=(10, 10)).is_regular
    assert not ChunkLayout(chunks=((10, 20, 30), 10)).is_regular


def test_levels_and_innermost() -> None:
    leaf = ChunkLayout(chunks=(5,))
    mid = ChunkLayout(chunks=(25,), inner=leaf)
    top = ChunkLayout(chunks=(50,), inner=mid)
    assert top.flattened_levels == (top, mid, leaf)
    assert top.flattened_levels[0] is top
    assert top.innermost is leaf
    assert top.is_sharded
    assert not leaf.is_sharded


@pytest.mark.parametrize("bad", [(0,), (-1,), ((10, 0),), ((),)])
def test_validation_rejects_nonpositive_and_empty(bad: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        ChunkLayout(chunks=bad)  # type: ignore[arg-type]


def test_validation_ndim_mismatch() -> None:
    with pytest.raises(ValueError):
        ChunkLayout(chunks=(10, 10), inner=ChunkLayout(chunks=(5,)))


# ---------------------------------------------------------------------------
# construction from real arrays (the four configurations)
# ---------------------------------------------------------------------------


def test_regular_unsharded() -> None:
    a = zarr.create_array(MemoryStore(), shape=(30,), chunks=(10,), dtype="int32")
    layout = a.chunk_layout
    assert layout == ChunkLayout(chunks=(10,))
    assert layout.is_regular
    assert not layout.is_sharded
    assert not a.is_sharded


def test_regular_sharded() -> None:
    a = zarr.create_array(MemoryStore(), shape=(100,), chunks=(10,), shards=(50,), dtype="int32")
    layout = a.chunk_layout
    assert layout == ChunkLayout(chunks=(50,), inner=ChunkLayout(chunks=(10,)))
    assert layout.is_regular
    assert layout.is_sharded
    assert a.is_sharded
    assert layout.innermost.chunks == (10,)


def test_rectilinear_unsharded() -> None:
    a = zarr.create_array(MemoryStore(), shape=(60,), chunks=[[10, 20, 30]], dtype="int32")
    layout = a.chunk_layout
    assert layout == ChunkLayout(chunks=((10, 20, 30),))
    assert not layout.is_regular
    assert not layout.is_sharded
    assert not a.is_sharded


def test_canonical_uniform_rectilinear_array() -> None:
    """A rectilinear-declared grid with uniform edges yields a regular layout."""
    rect = zarr.create_array(MemoryStore(), shape=(60,), chunks=[[20, 20, 20]], dtype="int32")
    reg = zarr.create_array(MemoryStore(), shape=(60,), chunks=(20,), dtype="int32")
    assert rect.chunk_layout == reg.chunk_layout
    assert rect.chunk_layout.is_regular


# ---------------------------------------------------------------------------
# the reconstruction contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        _CreateKwargs(shape=(30,), chunks=(10,)),
        _CreateKwargs(shape=(100,), chunks=(10,), shards=(50,)),
        _CreateKwargs(shape=(60,), chunks=[[10, 20, 30]]),
        _CreateKwargs(shape=(60,), chunks=[[20, 20, 20]]),  # normalizing case
    ],
)
def test_reconstruction_roundtrip(kwargs: _CreateKwargs) -> None:
    src = zarr.create_array(MemoryStore(), dtype="int32", **kwargs)
    layout = src.chunk_layout
    # ChunkLayout.chunks is a per-dimension union (int | edge tuple). create_array
    # accepts that mixed form at runtime, but ChunksLike/ShardsLike model only the
    # all-uniform or all-explicit shapes, so cast at this reconstruction boundary.
    recreate: _RecreateKwargs
    if layout.inner is not None:
        recreate = {
            "chunks": cast("ChunksLike", layout.inner.chunks),
            "shards": cast("ShardsLike", layout.chunks),
        }
    else:
        recreate = {"chunks": cast("ChunksLike", layout.chunks), "shards": None}
    dst = zarr.create_array(MemoryStore(), shape=src.shape, dtype="int32", **recreate)
    assert dst.chunk_layout == layout


# ---------------------------------------------------------------------------
# codec hook
# ---------------------------------------------------------------------------


def test_default_codec_is_opaque() -> None:
    assert BytesCodec().inner_chunk_layout() is None


def test_sharding_codec_reports_inner_layout() -> None:
    codec = ShardingCodec(chunk_shape=(10,), codecs=(BytesCodec(),))
    assert codec.inner_chunk_layout() == ChunkLayout(chunks=(10,))


def test_first_codec_rule_sharding_then_gzip() -> None:
    """[sharding, gzip] is still structurally sharded (diverges from .shards == None)."""
    a = zarr.create_array(
        MemoryStore(),
        shape=(100,),
        chunks=(10,),
        shards=(50,),
        compressors=[GzipCodec()],
        dtype="int32",
    )
    assert a.is_sharded
    assert a.chunk_layout.is_sharded
    assert len(a.chunk_layout.flattened_levels) == 2


def test_is_sharded_agrees_with_layout() -> None:
    for kwargs in (
        {"shape": (30,), "chunks": (10,)},
        {"shape": (100,), "chunks": (10,), "shards": (50,)},
        {"shape": (60,), "chunks": [[10, 20, 30]]},
    ):
        a = zarr.create_array(MemoryStore(), dtype="int32", **kwargs)
        assert a.is_sharded == a.chunk_layout.is_sharded


def test_roundtrip_preserves_data() -> None:
    src = zarr.create_array(MemoryStore(), shape=(60,), chunks=[[10, 20, 30]], dtype="int32")
    src[:] = np.arange(60, dtype="int32")
    np.testing.assert_array_equal(src[:], np.arange(60, dtype="int32"))
