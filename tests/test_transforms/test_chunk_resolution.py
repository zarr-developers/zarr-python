from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.test_transforms.conftest import Expect
from zarr.core._transforms.chunk_resolution import (
    iter_chunk_transforms,
    sub_transform_to_selections,
)
from zarr.core._transforms.domain import IndexDomain
from zarr.core._transforms.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr.core._transforms.transform import IndexTransform
from zarr.core.chunk_grids import ChunkGrid, FixedDimension

# ---------------------------------------------------------------------------
# iter_chunk_transforms — for a transform composed against a ChunkGrid, yield
# (chunk_coords, sub_transform, out_indices) for each touched chunk.
# ---------------------------------------------------------------------------


def _grid_1d(size: int, extent: int) -> ChunkGrid:
    return ChunkGrid(dimensions=(FixedDimension(size=size, extent=extent),))


def _grid_2d(size0: int, extent0: int, size1: int, extent1: int) -> ChunkGrid:
    return ChunkGrid(
        dimensions=(
            FixedDimension(size=size0, extent=extent0),
            FixedDimension(size=size1, extent=extent1),
        )
    )


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(IndexTransform.from_shape((10,)), _grid_1d(10, 10)),
            expected={"n_chunks": 1, "coords": [(0,)]},
            id="single-chunk-fits-array",
        ),
        Expect(
            input=(IndexTransform.from_shape((30,)), _grid_1d(10, 30)),
            expected={"n_chunks": 3, "coords": [(0,), (1,), (2,)]},
            id="three-chunks-1d",
        ),
        Expect(
            input=(IndexTransform.from_shape((20, 30)), _grid_2d(10, 20, 10, 30)),
            expected={
                "n_chunks": 6,
                "coords": [(i, j) for i in (0, 1) for j in (0, 1, 2)],
            },
            id="six-chunks-2x3",
        ),
        Expect(
            input=(IndexTransform.from_shape((100,))[5:8], _grid_1d(10, 100)),
            expected={"n_chunks": 1, "coords": [(0,)]},
            id="slice-within-chunk",
        ),
        Expect(
            input=(IndexTransform.from_shape((100,))[8:15], _grid_1d(10, 100)),
            expected={"n_chunks": 2, "coords": [(0,), (1,)]},
            id="slice-across-two-chunks",
        ),
    ],
    ids=lambda c: c.id,
)
def test_iter_chunk_transforms_yields_expected_chunks(
    case: Expect[tuple[IndexTransform, ChunkGrid], dict[str, Any]],
) -> None:
    """iter_chunk_transforms enumerates all chunks intersected by the transform."""
    transform, grid = case.input
    results = list(iter_chunk_transforms(transform, grid))
    assert len(results) == case.expected["n_chunks"]
    coords_list = [r[0] for r in results]
    for expected_coord in case.expected["coords"]:
        assert expected_coord in coords_list


def test_iter_chunk_transforms_constant_map_picks_single_chunk_per_dim() -> None:
    """An integer index produces a ConstantMap, fixing the chunk on that dim.

    arr[25, :] over a 10-element chunk grid: chunk index for storage 25 is 2,
    so every chunk yielded has coords[0] == 2. The free dim (the slice) iterates."""
    t = IndexTransform.from_shape((100, 100))[25, :]
    grid = _grid_2d(10, 100, 10, 100)
    results = list(iter_chunk_transforms(t, grid))
    assert len(results) == 10
    for coords, _, _ in results:
        assert coords[0] == 2


def test_iter_chunk_transforms_array_map_lists_chunks_for_array_entries() -> None:
    """An ArrayMap yields chunks for each unique chunk-id of its index_array entries."""
    idx = np.array([5, 15, 25], dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(ArrayMap(index_array=idx),),
    )
    results = list(iter_chunk_transforms(t, _grid_1d(10, 30)))
    coords_list = [r[0] for r in results]
    assert (0,) in coords_list
    assert (1,) in coords_list
    assert (2,) in coords_list


def test_iter_chunk_transforms_within_chunk_offset_is_local() -> None:
    """The yielded sub-transform's output is in chunk-local coordinates,
    so a slice arr[5:8] in chunk 0 yields offset=5 (the offset within the chunk)."""
    t = IndexTransform.from_shape((100,))[5:8]
    grid = _grid_1d(10, 100)
    results = list(iter_chunk_transforms(t, grid))
    assert len(results) == 1
    _, sub_t, _ = results[0]
    assert isinstance(sub_t.output[0], DimensionMap)
    assert sub_t.output[0].offset == 5


# ---------------------------------------------------------------------------
# sub_transform_to_selections — convert a chunk-local sub-transform into
# (chunk_selection, out_selection, drop_axes) tuples for the codec pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=IndexTransform(
                domain=IndexDomain.from_shape((10,)),
                output=(ConstantMap(offset=5),),
            ),
            expected={
                "chunk_sel": (5,),
                "out_sel": (),
                "drop_axes": (),
            },
            id="constant-map-yields-int-selection-no-out",
        ),
        Expect(
            input=IndexTransform(
                domain=IndexDomain.from_shape((10,)),
                output=(DimensionMap(input_dimension=0, offset=3, stride=1),),
            ),
            expected={
                "chunk_sel": (slice(3, 13, 1),),
                "out_sel": (slice(0, 10),),
                "drop_axes": (),
            },
            id="dimension-map-stride-1-yields-contiguous-slice",
        ),
        Expect(
            input=IndexTransform(
                domain=IndexDomain.from_shape((5,)),
                output=(DimensionMap(input_dimension=0, offset=2, stride=3),),
            ),
            expected={
                "chunk_sel": (slice(2, 17, 3),),
                "out_sel": (slice(0, 5),),
                "drop_axes": (),
            },
            id="dimension-map-strided-yields-strided-slice",
        ),
        Expect(
            input=IndexTransform(
                domain=IndexDomain.from_shape((10,)),
                output=(
                    ConstantMap(offset=5),
                    DimensionMap(input_dimension=0, offset=0, stride=1),
                ),
            ),
            expected={
                "chunk_sel_kinds": (int, slice),
                "chunk_sel_values": (5, slice(0, 10, 1)),
                "drop_axes": (),
            },
            id="mixed-2d-constant-and-dimension",
        ),
    ],
    ids=lambda c: c.id,
)
def test_sub_transform_to_selections_basic(case: Expect[IndexTransform, dict[str, Any]]) -> None:
    """sub_transform_to_selections produces the expected (chunk_sel, out_sel, drop_axes) for non-array maps."""
    chunk_sel, out_sel, drop_axes = sub_transform_to_selections(case.input)
    if "chunk_sel" in case.expected:
        assert chunk_sel == case.expected["chunk_sel"]
    if "chunk_sel_kinds" in case.expected:
        for got, expected_kind in zip(chunk_sel, case.expected["chunk_sel_kinds"], strict=True):
            assert isinstance(got, expected_kind)
    if "chunk_sel_values" in case.expected:
        for got, expected_val in zip(chunk_sel, case.expected["chunk_sel_values"], strict=True):
            assert got == expected_val
    if "out_sel" in case.expected:
        assert out_sel == case.expected["out_sel"]
    assert drop_axes == case.expected["drop_axes"]


def test_sub_transform_to_selections_array_map_no_offset() -> None:
    """An ArrayMap with offset=0, stride=1 produces the index_array itself as chunk_sel."""
    arr = np.array([1, 5, 9], dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(ArrayMap(index_array=arr, offset=0, stride=1),),
    )
    chunk_sel, out_sel, drop_axes = sub_transform_to_selections(t)
    assert isinstance(chunk_sel[0], np.ndarray)
    np.testing.assert_array_equal(chunk_sel[0], arr)
    # Without out_indices, out_sel falls back to a domain-derived slice.
    assert out_sel == (slice(0, 3),)
    assert drop_axes == ()


def test_sub_transform_to_selections_array_map_with_offset_stride() -> None:
    """An ArrayMap with non-zero offset/stride is materialized into storage coords."""
    arr = np.array([0, 1, 2], dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(ArrayMap(index_array=arr, offset=10, stride=5),),
    )
    chunk_sel, _out_sel, drop_axes = sub_transform_to_selections(t)
    assert isinstance(chunk_sel[0], np.ndarray)
    np.testing.assert_array_equal(chunk_sel[0], np.array([10, 15, 20]))
    assert drop_axes == ()


def test_sub_transform_to_selections_orthogonal_array_with_out_indices() -> None:
    """When out_indices is supplied with a single ArrayMap (orthogonal mode),
    out_sel uses the supplied scatter indices rather than a domain slice."""
    arr = np.array([1, 5, 9], dtype=np.intp)
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(ArrayMap(index_array=arr),),
    )
    out_indices = np.array([0, 2], dtype=np.intp)
    _chunk_sel, out_sel, _drop_axes = sub_transform_to_selections(t, out_indices)
    assert len(out_sel) == 1
    assert isinstance(out_sel[0], np.ndarray)
    np.testing.assert_array_equal(out_sel[0], out_indices)


def test_sub_transform_to_selections_vectorized_with_out_indices() -> None:
    """When out_indices is supplied with 2+ correlated ArrayMaps (vectorized mode),
    out_sel collapses to a single shared scatter array."""
    t = IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(
            ArrayMap(index_array=np.array([1, 5, 9], dtype=np.intp)),
            ArrayMap(index_array=np.array([10, 11, 12], dtype=np.intp)),
        ),
    )
    out_indices = np.array([0, 1], dtype=np.intp)
    _chunk_sel, out_sel, _drop_axes = sub_transform_to_selections(t, out_indices)
    assert len(out_sel) == 1
    np.testing.assert_array_equal(out_sel[0], out_indices)
