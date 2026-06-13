from __future__ import annotations

import numpy as np

from zarr.core.chunk_grids import ChunkGrid, FixedDimension
from zarr.core.transforms.chunk_resolution import iter_chunk_transforms, sub_transform_to_selections
from zarr.core.transforms.domain import IndexDomain
from zarr.core.transforms.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr.core.transforms.transform import IndexTransform


class TestChunkResolutionIdentity:
    def test_single_chunk(self) -> None:
        """Array fits in one chunk."""
        t = IndexTransform.from_shape((10,))
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=10),))
        results = list(iter_chunk_transforms(t, grid))
        assert len(results) == 1
        coords, sub_t, _ = results[0]
        assert coords == (0,)
        assert sub_t.domain.shape == (10,)

    def test_multiple_chunks_1d(self) -> None:
        """1D array spanning 3 chunks."""
        t = IndexTransform.from_shape((30,))
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=30),))
        results = list(iter_chunk_transforms(t, grid))
        assert len(results) == 3
        coords_list = [r[0] for r in results]
        assert (0,) in coords_list
        assert (1,) in coords_list
        assert (2,) in coords_list

    def test_multiple_chunks_2d(self) -> None:
        """2D array spanning 2x3 chunks."""
        t = IndexTransform.from_shape((20, 30))
        grid = ChunkGrid(
            dimensions=(
                FixedDimension(size=10, extent=20),
                FixedDimension(size=10, extent=30),
            )
        )
        results = list(iter_chunk_transforms(t, grid))
        assert len(results) == 6
        coords_list = [r[0] for r in results]
        assert (0, 0) in coords_list
        assert (1, 2) in coords_list


class TestChunkResolutionSliced:
    def test_slice_within_chunk(self) -> None:
        """Slice that falls within a single chunk."""
        t = IndexTransform.from_shape((100,))[5:8]
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=100),))
        results = list(iter_chunk_transforms(t, grid))
        assert len(results) == 1
        coords, sub_t, _ = results[0]
        assert coords == (0,)
        assert isinstance(sub_t.output[0], DimensionMap)
        assert sub_t.output[0].offset == 5

    def test_slice_across_chunks(self) -> None:
        """Slice that spans two chunks."""
        t = IndexTransform.from_shape((100,))[8:15]
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=100),))
        results = list(iter_chunk_transforms(t, grid))
        assert len(results) == 2
        coords_list = [r[0] for r in results]
        assert (0,) in coords_list
        assert (1,) in coords_list


class TestChunkResolutionConstant:
    def test_integer_index(self) -> None:
        """Integer index produces constant map — single chunk per constant dim."""
        t = IndexTransform.from_shape((100, 100))[25, :]
        grid = ChunkGrid(
            dimensions=(
                FixedDimension(size=10, extent=100),
                FixedDimension(size=10, extent=100),
            )
        )
        results = list(iter_chunk_transforms(t, grid))
        assert len(results) == 10
        for coords, _, _ in results:
            assert coords[0] == 2


class TestChunkResolutionArray:
    def test_array_index(self) -> None:
        """Array index map — chunks determined by array values."""
        idx = np.array([5, 15, 25], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=idx),),
        )
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=30),))
        results = list(iter_chunk_transforms(t, grid))
        coords_list = [r[0] for r in results]
        assert (0,) in coords_list
        assert (1,) in coords_list
        assert (2,) in coords_list


class TestSubTransformToSelections:
    def test_constant_map(self) -> None:
        """ConstantMap produces int selection + drop axis."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(ConstantMap(offset=5),),
        )
        chunk_sel, out_sel, drop_axes = sub_transform_to_selections(t)
        assert chunk_sel == (5,)
        assert out_sel == ()
        assert drop_axes == ()

    def test_dimension_map_stride_1(self) -> None:
        """DimensionMap with stride=1 produces contiguous slice."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(DimensionMap(input_dimension=0, offset=3, stride=1),),
        )
        chunk_sel, out_sel, drop_axes = sub_transform_to_selections(t)
        assert chunk_sel == (slice(3, 13, 1),)
        assert out_sel == (slice(0, 10),)
        assert drop_axes == ()

    def test_dimension_map_strided(self) -> None:
        """DimensionMap with stride>1 produces strided slice."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((5,)),
            output=(DimensionMap(input_dimension=0, offset=2, stride=3),),
        )
        chunk_sel, out_sel, drop_axes = sub_transform_to_selections(t)
        assert chunk_sel == (slice(2, 17, 3),)
        assert out_sel == (slice(0, 5),)
        assert drop_axes == ()

    def test_array_map(self) -> None:
        """ArrayMap produces integer array selection."""
        arr = np.array([1, 5, 9], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=arr, offset=0, stride=1),),
        )
        chunk_sel, out_sel, drop_axes = sub_transform_to_selections(t)
        assert isinstance(chunk_sel[0], np.ndarray)
        np.testing.assert_array_equal(chunk_sel[0], arr)
        # Without chunk_mask, out_sel falls back to domain-based slices
        assert out_sel == (slice(0, 3),)
        assert drop_axes == ()

    def test_array_map_with_offset_stride(self) -> None:
        """ArrayMap with offset and stride computes storage coords."""
        arr = np.array([0, 1, 2], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=arr, offset=10, stride=5),),
        )
        chunk_sel, _out_sel, drop_axes = sub_transform_to_selections(t)
        assert isinstance(chunk_sel[0], np.ndarray)
        np.testing.assert_array_equal(chunk_sel[0], np.array([10, 15, 20]))
        assert drop_axes == ()

    def test_mixed_maps_2d(self) -> None:
        """Mix of ConstantMap and DimensionMap."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(
                ConstantMap(offset=5),
                DimensionMap(input_dimension=0, offset=0, stride=1),
            ),
        )
        chunk_sel, _out_sel, drop_axes = sub_transform_to_selections(t)
        assert chunk_sel[0] == 5
        assert chunk_sel[1] == slice(0, 10, 1)
        # drop_axes is empty — integer in chunk_sel naturally drops the dim via numpy
        assert drop_axes == ()
