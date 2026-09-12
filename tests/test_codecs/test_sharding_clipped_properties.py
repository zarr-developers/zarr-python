"""NumPy oracle for clipped inner chunks across both pipeline implementations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import zarr

if TYPE_CHECKING:
    from zarr.codecs.sharding import IndexLocation


@st.composite
def clipped_grids(
    draw: st.DrawFn,
) -> tuple[tuple[int, ...], tuple[int | tuple[int, ...], ...], tuple[int, ...]]:
    shape: list[int] = []
    shards: list[int | tuple[int, ...]] = []
    chunks: list[int] = []
    for _ in range(draw(st.integers(1, 3))):
        if draw(st.booleans()):
            edges = tuple(draw(st.lists(st.integers(1, 6), min_size=1, max_size=3)))
            shards.append(edges)
            shape.append(sum(edges))
        else:
            shards.append(draw(st.integers(1, 6)))
            shape.append(draw(st.integers(1, 12)))
        chunks.append(draw(st.integers(1, 8)))
    return tuple(shape), tuple(shards), tuple(chunks)


@pytest.mark.parametrize("pipeline", ["BatchedCodecPipeline", "FusedCodecPipeline"])
@pytest.mark.parametrize("index_location", ["start", "end"])
@settings(max_examples=30, deadline=None)
@given(case=clipped_grids())
def test_clipped_sharding_matches_numpy(
    pipeline: str,
    index_location: IndexLocation,
    case: tuple[tuple[int, ...], tuple[int | tuple[int, ...], ...], tuple[int, ...]],
) -> None:
    shape, shards, chunks = case
    expected = np.arange(math.prod(shape), dtype="i4").reshape(shape)
    selection = tuple(slice(1, None, 2) for _ in shape)
    with zarr.config.set(
        {
            "array.rectilinear_chunks": True,
            "codec_pipeline.path": f"zarr.core.codec_pipeline.{pipeline}",
        }
    ):
        array = zarr.create_array(
            {},
            shape=shape,
            shards={"shape": shards, "index_location": index_location},
            chunks=chunks,
            dtype="i4",
            compressors=None,
            fill_value=0,
        )
        array[:] = expected
        np.testing.assert_array_equal(array[selection], expected[selection])
        array[selection] = -7
        expected[selection] = -7
        np.testing.assert_array_equal(array[:], expected)
        array[selection] = 0
        expected[selection] = 0
        reopened = zarr.open_array(array.store, mode="r")
        np.testing.assert_array_equal(reopened[:], expected)
        assert array.cdata_shape == tuple(len(axis) for axis in array.read_chunk_sizes)
        assert tuple(sum(axis) for axis in array.read_chunk_sizes) == shape
