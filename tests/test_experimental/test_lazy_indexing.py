"""
Tests for lazy indexing with TensorStore-inspired domain tracking.

Key difference from NumPy: indices are ABSOLUTE coordinates in the domain's
index space, not relative offsets. Negative indices mean negative coordinates,
not "counting from the end".
"""

import numpy as np
import pytest

import zarr
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.experimental.lazy_indexing import (
    Array,
    ArrayDesc,
    ChunkEntry,
    ChunkLayout,
    ChunkMap,
    IndexDomain,
    IndexTransform,
    Layer,
    PendingSelection,
    SelectionKind,
    merge,
)
from zarr.storage import MemoryStore


def _make_test_cmap(
    *,
    domain: IndexDomain,
    chunk_shape: tuple[int, ...],
    storage_shape: tuple[int, ...],
    index_transform: tuple[int, ...],
    encode_chunk_key: object = None,
) -> ChunkMap:
    """Helper to create a ChunkMap for tests using the new API."""
    desc = ArrayDesc(
        shape=storage_shape,
        data_type=np.dtype("float32"),
        chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
        encode_chunk_key=encode_chunk_key,
        fill_value=0,
        codecs=None,
    )

    def _make_child(
        chunk_coords: tuple[int, ...],
        selection: tuple[slice, ...],
        chunk_domain: IndexDomain,
    ) -> ChunkEntry:
        """Return a ChunkEntry for test inspection."""
        ek = encode_chunk_key
        if ek is not None:
            key = ek(chunk_coords)
        else:
            key = "/".join(map(str, ("c",) + chunk_coords))

        chunk_shape_actual = desc.chunk_grid.get_chunk_shape(storage_shape, chunk_coords)

        return ChunkEntry(
            domain=chunk_domain,
            path=key,
            chunk_selection=selection,
            chunk_coords=chunk_coords,
            chunk_shape=chunk_shape_actual,
        )

    return ChunkMap(
        desc=desc,
        domain=domain,
        index_transform=index_transform,
        make_child=_make_child,
    )


class TestIndexDomain:
    """Tests for the IndexDomain class."""

    def test_from_shape(self) -> None:
        """Test creating a domain from a shape."""
        domain = IndexDomain.from_shape((10, 20, 30))
        assert domain.inclusive_min == (0, 0, 0)
        assert domain.exclusive_max == (10, 20, 30)
        assert domain.shape == (10, 20, 30)
        assert domain.origin == (0, 0, 0)
        assert domain.ndim == 3

    def test_non_zero_origin(self) -> None:
        """Test a domain with non-zero origin."""
        domain = IndexDomain(inclusive_min=(5, 10), exclusive_max=(15, 30))
        assert domain.origin == (5, 10)
        assert domain.shape == (10, 20)
        assert domain.ndim == 2

    def test_contains(self) -> None:
        """Test the contains method."""
        domain = IndexDomain(inclusive_min=(5, 10), exclusive_max=(15, 30))

        # Inside the domain
        assert domain.contains((5, 10))
        assert domain.contains((10, 20))
        assert domain.contains((14, 29))

        # Outside the domain
        assert not domain.contains((4, 10))  # x too low
        assert not domain.contains((15, 10))  # x at exclusive max
        assert not domain.contains((5, 30))  # y at exclusive max
        assert not domain.contains((5, 9))  # y too low

    def test_contains_domain(self) -> None:
        """Test that one domain contains another."""
        outer = IndexDomain(inclusive_min=(0, 0), exclusive_max=(100, 100))
        inner = IndexDomain(inclusive_min=(10, 20), exclusive_max=(50, 60))

        assert outer.contains_domain(inner)
        assert not inner.contains_domain(outer)

        # Partially overlapping
        partial = IndexDomain(inclusive_min=(50, 50), exclusive_max=(150, 150))
        assert not outer.contains_domain(partial)

    def test_invalid_domain(self) -> None:
        """Test that invalid domains raise errors."""
        # min > max
        with pytest.raises(ValueError, match="inclusive_min must be <= exclusive_max"):
            IndexDomain(inclusive_min=(10,), exclusive_max=(5,))

        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            IndexDomain(inclusive_min=(0, 0), exclusive_max=(10,))

    def test_repr(self) -> None:
        """Test string representation."""
        domain = IndexDomain(inclusive_min=(5, 10), exclusive_max=(15, 30))
        assert repr(domain) == "IndexDomain([5, 15), [10, 30))"

    def test_translate_basic(self) -> None:
        """Test basic translation of a domain."""
        domain = IndexDomain(inclusive_min=(10, 20), exclusive_max=(30, 40))
        translated = domain.translate((-10, -20))

        assert translated.inclusive_min == (0, 0)
        assert translated.exclusive_max == (20, 20)
        assert translated.shape == domain.shape  # Shape unchanged

    def test_translate_positive_offset(self) -> None:
        """Test translation with positive offset."""
        domain = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 20))
        translated = domain.translate((5, 10))

        assert translated.inclusive_min == (5, 10)
        assert translated.exclusive_max == (15, 30)

    def test_translate_to_negative_coords(self) -> None:
        """Test translation that results in negative coordinates."""
        domain = IndexDomain(inclusive_min=(5, 10), exclusive_max=(15, 20))
        translated = domain.translate((-10, -15))

        assert translated.inclusive_min == (-5, -5)
        assert translated.exclusive_max == (5, 5)

    def test_translate_preserves_shape(self) -> None:
        """Test that translation preserves shape."""
        domain = IndexDomain(inclusive_min=(100, 200), exclusive_max=(150, 300))
        original_shape = domain.shape

        for offset in [(-100, -200), (50, 100), (-50, 50)]:
            translated = domain.translate(offset)
            assert translated.shape == original_shape

    def test_translate_wrong_ndim_raises(self) -> None:
        """Test that translate raises for mismatched dimensions."""
        domain = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 10))

        with pytest.raises(ValueError, match="same length"):
            domain.translate((5,))

        with pytest.raises(ValueError, match="same length"):
            domain.translate((5, 5, 5))

    def test_translate_identity(self) -> None:
        """Test that translating by zero offset is identity."""
        domain = IndexDomain(inclusive_min=(5, 10), exclusive_max=(15, 30))
        translated = domain.translate((0, 0))

        assert translated == domain

    def test_translate_1d(self) -> None:
        """Test translation in 1D."""
        domain = IndexDomain(inclusive_min=(50,), exclusive_max=(100,))
        translated = domain.translate((-50,))

        assert translated.inclusive_min == (0,)
        assert translated.exclusive_max == (50,)


class TestChunkMap:
    """Tests for the ChunkMap class.

    Keys are Region tuples: ``(inclusive_min, exclusive_max)``.
    """

    # Helper for concise Region creation
    @staticmethod
    def D(
        lo: tuple[int, ...], hi: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (lo, hi)

    def test_single_chunk_full_domain(self) -> None:
        """ChunkMap with domain covering exactly one chunk."""
        cmap = _make_test_cmap(
            domain=IndexDomain.from_shape((10,)),
            chunk_shape=(10,),
            storage_shape=(10,),
            index_transform=(0,),
        )
        assert len(cmap) == 1
        assert list(cmap) == [self.D((0,), (10,))]
        assert self.D((0,), (10,)) in cmap

        child = cmap[self.D((0,), (10,))]
        assert child.path == "c/0"
        assert child.chunk_selection == (slice(0, 10),)

    def test_multiple_chunks(self) -> None:
        """ChunkMap spanning multiple chunks."""
        cmap = _make_test_cmap(
            domain=IndexDomain(inclusive_min=(25,), exclusive_max=(75,)),
            chunk_shape=(10,),
            storage_shape=(100,),
            index_transform=(0,),
        )
        assert len(cmap) == 6
        keys = list(cmap)
        assert keys == [self.D((i * 10,), ((i + 1) * 10,)) for i in range(2, 8)]

        child = cmap[self.D((20,), (30,))]
        assert child.path == "c/2"
        assert child.chunk_selection == (slice(5, 10),)

        child = cmap[self.D((70,), (80,))]
        assert child.path == "c/7"
        assert child.chunk_selection == (slice(0, 5),)

        child = cmap[self.D((40,), (50,))]
        assert child.path == "c/4"
        assert child.chunk_selection == (slice(0, 10),)

    def test_with_index_transform(self) -> None:
        """ChunkMap with non-zero offset."""
        cmap = _make_test_cmap(
            domain=IndexDomain(inclusive_min=(10,), exclusive_max=(20,)),
            chunk_shape=(5,),
            storage_shape=(10,),
            index_transform=(10,),
        )
        assert len(cmap) == 2
        child0 = cmap[self.D((0,), (5,))]
        assert child0.path == "c/0"
        assert child0.chunk_selection == (slice(0, 5),)

        child1 = cmap[self.D((5,), (10,))]
        assert child1.path == "c/1"
        assert child1.chunk_selection == (slice(0, 5),)

    def test_domain_partially_outside_storage(self) -> None:
        """Domain extending beyond storage bounds gets clipped."""
        cmap = _make_test_cmap(
            domain=IndexDomain(inclusive_min=(5,), exclusive_max=(15,)),
            chunk_shape=(5,),
            storage_shape=(10,),
            index_transform=(0,),
        )
        assert len(cmap) == 1
        child = cmap[self.D((5,), (10,))]
        assert child.chunk_selection == (slice(0, 5),)

    def test_domain_completely_outside_storage(self) -> None:
        """Empty ChunkMap when domain doesn't overlap storage."""
        cmap = _make_test_cmap(
            domain=IndexDomain(inclusive_min=(20,), exclusive_max=(30,)),
            chunk_shape=(5,),
            storage_shape=(10,),
            index_transform=(0,),
        )
        assert len(cmap) == 0
        assert list(cmap) == []

    def test_multidimensional(self) -> None:
        """ChunkMap for 2D array."""
        cmap = _make_test_cmap(
            domain=IndexDomain(inclusive_min=(5, 15), exclusive_max=(15, 25)),
            chunk_shape=(10, 10),
            storage_shape=(20, 30),
            index_transform=(0, 0),
        )
        assert len(cmap) == 4
        keys = list(cmap)
        assert keys == [
            self.D((0, 10), (10, 20)),
            self.D((0, 20), (10, 30)),
            self.D((10, 10), (20, 20)),
            self.D((10, 20), (20, 30)),
        ]

        child = cmap[self.D((0, 10), (10, 20))]
        assert child.path == "c/0/1"
        assert child.chunk_selection == (slice(5, 10), slice(5, 10))

    def test_negative_domain_with_offset(self) -> None:
        """ChunkMap with negative domain coordinates."""
        cmap = _make_test_cmap(
            domain=IndexDomain(inclusive_min=(-5,), exclusive_max=(5,)),
            chunk_shape=(5,),
            storage_shape=(10,),
            index_transform=(-5,),
        )
        assert len(cmap) == 2
        assert cmap[self.D((0,), (5,))].path == "c/0"
        assert cmap[self.D((5,), (10,))].path == "c/1"

    def test_getitem_key_error(self) -> None:
        """KeyError for regions outside the map."""
        cmap = _make_test_cmap(
            domain=IndexDomain(inclusive_min=(25,), exclusive_max=(35,)),
            chunk_shape=(10,),
            storage_shape=(100,),
            index_transform=(0,),
        )
        assert self.D((20,), (30,)) in cmap
        assert self.D((30,), (40,)) in cmap
        with pytest.raises(KeyError):
            cmap[self.D((0,), (10,))]
        with pytest.raises(KeyError):
            cmap[self.D((40,), (50,))]

    def test_mapping_methods(self) -> None:
        """Inherited Mapping methods (keys, values, items) work."""
        cmap = _make_test_cmap(
            domain=IndexDomain.from_shape((20,)),
            chunk_shape=(10,),
            storage_shape=(20,),
            index_transform=(0,),
        )
        assert list(cmap.keys()) == [self.D((0,), (10,)), self.D((10,), (20,))]
        values = list(cmap.values())
        assert len(values) == 2

        items = list(cmap.items())
        assert len(items) == 2
        assert items[0][0] == self.D((0,), (10,))
        assert items[0][1].path == "c/0"

    def test_composition_with_array_getitem(self) -> None:
        """arr[10:20].chunk_map reflects the narrowed domain."""
        store = MemoryStore()
        zarr.create_array(
            store=store,
            shape=(100,),
            dtype="float32",
            chunks=(10,),
            fill_value=0,
        )
        arr = Array.open(store)

        # Full array: 10 chunks
        assert arr.chunk_map is not None
        assert len(arr.chunk_map) == 10

        # Sliced: [25, 75) -> chunks 2-7
        sliced = arr[25:75]
        assert sliced.chunk_map is not None
        assert len(sliced.chunk_map) == 6
        assert self.D((20,), (30,)) in sliced.chunk_map
        assert self.D((70,), (80,)) in sliced.chunk_map
        assert self.D((10,), (20,)) not in sliced.chunk_map
        assert self.D((80,), (90,)) not in sliced.chunk_map

        # Verify first chunk — child is a ChunkEntry now
        child = sliced.chunk_map[self.D((20,), (30,))]
        # Chunk 2 covers [20, 30). Sliced to [25, 75) => intersection is [25, 30)
        assert child.domain == IndexDomain(inclusive_min=(25,), exclusive_max=(30,))
        assert child.path.endswith("c/2")

    def test_custom_key_encoder(self) -> None:
        """ChunkMap with custom key encoder (e.g., v2 format)."""
        cmap = _make_test_cmap(
            domain=IndexDomain.from_shape((20, 30)),
            chunk_shape=(10, 10),
            storage_shape=(20, 30),
            index_transform=(0, 0),
            encode_chunk_key=lambda coords: ".".join(map(str, coords)),
        )
        assert cmap[self.D((0, 0), (10, 10))].path == "0.0"
        assert cmap[self.D((10, 20), (20, 30))].path == "1.2"

    def test_repr(self) -> None:
        """ChunkMap repr is informative."""
        cmap = _make_test_cmap(
            domain=IndexDomain(inclusive_min=(25,), exclusive_max=(75,)),
            chunk_shape=(10,),
            storage_shape=(100,),
            index_transform=(0,),
        )
        r = repr(cmap)
        assert "ChunkMap" in r
        assert "len=6" in r

    def test_3d(self) -> None:
        """ChunkMap for 3D array."""
        cmap = _make_test_cmap(
            domain=IndexDomain.from_shape((30, 20, 10)),
            chunk_shape=(10, 10, 10),
            storage_shape=(30, 20, 10),
            index_transform=(0, 0, 0),
        )
        assert len(cmap) == 6  # 3 * 2 * 1
        keys = list(cmap)
        assert len(keys) == 6
        assert keys[0] == self.D((0, 0, 0), (10, 10, 10))
        assert keys[-1] == self.D((20, 10, 0), (30, 20, 10))
        assert cmap[self.D((10, 0, 0), (20, 10, 10))].path == "c/1/0/0"

    def test_child_is_chunk_entry(self) -> None:
        """ChunkMap values are ChunkEntry records."""
        cmap = _make_test_cmap(
            domain=IndexDomain.from_shape((100,)),
            chunk_shape=(10,),
            storage_shape=(100,),
            index_transform=(0,),
        )
        child = cmap[self.D((0,), (10,))]
        assert isinstance(child, ChunkEntry)
        assert child.chunk_shape == (10,)

    def test_child_domain_reflects_intersection(self) -> None:
        """Child's domain is the intersection of parent domain with chunk region."""
        cmap = _make_test_cmap(
            domain=IndexDomain(inclusive_min=(25,), exclusive_max=(75,)),
            chunk_shape=(10,),
            storage_shape=(100,),
            index_transform=(0,),
        )
        # Chunk 2 covers [20, 30), intersection with [25, 75) is [25, 30)
        child = cmap[self.D((20,), (30,))]
        assert child.domain.inclusive_min == (25,)
        assert child.domain.exclusive_max == (30,)

        # Chunk 4 covers [40, 50), fully inside [25, 75)
        child = cmap[self.D((40,), (50,))]
        assert child.domain.inclusive_min == (40,)
        assert child.domain.exclusive_max == (50,)

    def test_sharded_via_array(self) -> None:
        """Array.chunk_map for sharded arrays has correct structure."""
        store = MemoryStore()
        zarr.create_array(
            store=store,
            shape=(100,),
            dtype="float32",
            chunks=(5,),
            shards=(20,),
            fill_value=0,
        )
        arr = Array.open(store)
        cmap = arr.chunk_map
        assert cmap is not None
        assert len(cmap) == 5  # 100 / 20 = 5 shards

        # Each value is a ChunkEntry
        child = cmap[self.D((0,), (20,))]
        assert isinstance(child, ChunkEntry)
        assert child.chunk_shape == (20,)


class TestChunkLayout:
    """Tests for the ChunkLayout class."""

    def test_from_chunk_shape(self) -> None:
        """Test creating a layout with zero origin."""
        layout = ChunkLayout.from_chunk_shape((10, 20))
        assert layout.grid_origin == (0, 0)
        assert layout.chunk_shape == (10, 20)
        assert layout.ndim == 2

    def test_is_aligned(self) -> None:
        """Test chunk alignment checking."""
        layout = ChunkLayout(grid_origin=(0, 0), chunk_shape=(10, 10))

        # On chunk boundaries
        assert layout.is_aligned((0, 0))
        assert layout.is_aligned((10, 0))
        assert layout.is_aligned((0, 10))
        assert layout.is_aligned((10, 10))
        assert layout.is_aligned((100, 200))

        # Not on chunk boundaries
        assert not layout.is_aligned((5, 0))
        assert not layout.is_aligned((0, 5))
        assert not layout.is_aligned((5, 5))
        assert not layout.is_aligned((15, 25))

    def test_is_aligned_nonzero_origin(self) -> None:
        """Test alignment with non-zero grid origin."""
        layout = ChunkLayout(grid_origin=(5, 5), chunk_shape=(10, 10))

        # Aligned relative to origin at (5, 5)
        assert layout.is_aligned((5, 5))
        assert layout.is_aligned((15, 5))
        assert layout.is_aligned((5, 15))
        assert layout.is_aligned((15, 15))

        # Not aligned
        assert not layout.is_aligned((0, 0))  # Would be aligned if origin was 0
        assert not layout.is_aligned((10, 10))
        assert not layout.is_aligned((7, 5))

    def test_chunk_domain(self) -> None:
        """Test getting the domain of a specific chunk."""
        layout = ChunkLayout(grid_origin=(0, 0), chunk_shape=(10, 10))

        # First chunk
        dom = layout.chunk_domain((0, 0))
        assert dom.inclusive_min == (0, 0)
        assert dom.exclusive_max == (10, 10)

        # Another chunk
        dom = layout.chunk_domain((2, 3))
        assert dom.inclusive_min == (20, 30)
        assert dom.exclusive_max == (30, 40)

    def test_chunk_domain_nonzero_origin(self) -> None:
        """Test chunk domain with non-zero grid origin."""
        layout = ChunkLayout(grid_origin=(5, 5), chunk_shape=(10, 10))

        # First chunk starts at grid origin
        dom = layout.chunk_domain((0, 0))
        assert dom.inclusive_min == (5, 5)
        assert dom.exclusive_max == (15, 15)

        # Second chunk in each dimension
        dom = layout.chunk_domain((1, 1))
        assert dom.inclusive_min == (15, 15)
        assert dom.exclusive_max == (25, 25)

    def test_chunk_coords_for_point(self) -> None:
        """Test finding which chunk contains a point."""
        layout = ChunkLayout(grid_origin=(0, 0), chunk_shape=(10, 10))

        assert layout.chunk_coords_for_point((0, 0)) == (0, 0)
        assert layout.chunk_coords_for_point((5, 5)) == (0, 0)
        assert layout.chunk_coords_for_point((9, 9)) == (0, 0)
        assert layout.chunk_coords_for_point((10, 10)) == (1, 1)
        assert layout.chunk_coords_for_point((25, 35)) == (2, 3)

    def test_chunk_coords_for_point_nonzero_origin(self) -> None:
        """Test chunk coords with non-zero grid origin."""
        layout = ChunkLayout(grid_origin=(5, 5), chunk_shape=(10, 10))

        # Point at grid origin is in chunk (0, 0)
        assert layout.chunk_coords_for_point((5, 5)) == (0, 0)
        assert layout.chunk_coords_for_point((14, 14)) == (0, 0)
        assert layout.chunk_coords_for_point((15, 15)) == (1, 1)

        # Point before grid origin is in chunk (-1, -1)
        assert layout.chunk_coords_for_point((0, 0)) == (-1, -1)
        assert layout.chunk_coords_for_point((4, 4)) == (-1, -1)

    def test_iter_chunk_coords(self) -> None:
        """Test iterating over chunks overlapping a domain."""
        layout = ChunkLayout(grid_origin=(0, 0), chunk_shape=(10, 10))
        domain = IndexDomain(inclusive_min=(5, 15), exclusive_max=(25, 35))

        coords = list(layout.iter_chunk_coords(domain))

        # Domain [5, 25) x [15, 35) overlaps:
        # Dim 0: chunks 0, 1, 2 (0-10, 10-20, 20-30)
        # Dim 1: chunks 1, 2, 3 (10-20, 20-30, 30-40)
        expected = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 2),
            (2, 3),
        ]
        assert coords == expected

    def test_iter_chunk_domains(self) -> None:
        """Test iterating over chunk domains overlapping a region."""
        layout = ChunkLayout(grid_origin=(0,), chunk_shape=(10,))
        domain = IndexDomain(inclusive_min=(15,), exclusive_max=(35,))

        results = list(layout.iter_chunk_domains(domain))

        # Domain [15, 35) overlaps chunks 1, 2, 3
        assert len(results) == 3

        # Chunk 1: [10, 20) intersected with [15, 35) = [15, 20)
        coords, dom = results[0]
        assert coords == (1,)
        assert dom.inclusive_min == (15,)
        assert dom.exclusive_max == (20,)

        # Chunk 2: [20, 30) fully contained
        coords, dom = results[1]
        assert coords == (2,)
        assert dom.inclusive_min == (20,)
        assert dom.exclusive_max == (30,)

        # Chunk 3: [30, 40) intersected with [15, 35) = [30, 35)
        coords, dom = results[2]
        assert coords == (3,)
        assert dom.inclusive_min == (30,)
        assert dom.exclusive_max == (35,)

    def test_aligned_domain(self) -> None:
        """Test finding the largest aligned subdomain."""
        layout = ChunkLayout(grid_origin=(0,), chunk_shape=(10,))

        # Domain that's already aligned
        domain = IndexDomain(inclusive_min=(10,), exclusive_max=(30,))
        aligned = layout.aligned_domain(domain)
        assert aligned.inclusive_min == (10,)
        assert aligned.exclusive_max == (30,)

        # Domain that needs rounding
        domain = IndexDomain(inclusive_min=(5,), exclusive_max=(35,))
        aligned = layout.aligned_domain(domain)
        assert aligned.inclusive_min == (10,)  # Rounded up from 5
        assert aligned.exclusive_max == (30,)  # Rounded down from 35

        # Domain smaller than a chunk
        domain = IndexDomain(inclusive_min=(12,), exclusive_max=(18,))
        aligned = layout.aligned_domain(domain)
        assert aligned.inclusive_min == (20,)  # Rounded up
        assert aligned.exclusive_max == (20,)  # Empty (rounded down < rounded up)

    def test_aligned_domain_nonzero_origin(self) -> None:
        """Test aligned_domain with non-zero grid origin."""
        layout = ChunkLayout(grid_origin=(5,), chunk_shape=(10,))

        # Domain [7, 28) -> aligned to [15, 25) (boundaries at 5, 15, 25, 35...)
        domain = IndexDomain(inclusive_min=(7,), exclusive_max=(28,))
        aligned = layout.aligned_domain(domain)
        assert aligned.inclusive_min == (15,)
        assert aligned.exclusive_max == (25,)

    def test_invalid_chunk_shape(self) -> None:
        """Test that zero or negative chunk shapes raise errors."""
        with pytest.raises(ValueError, match="positive"):
            ChunkLayout(grid_origin=(0,), chunk_shape=(0,))

        with pytest.raises(ValueError, match="positive"):
            ChunkLayout(grid_origin=(0,), chunk_shape=(-5,))

    def test_mismatched_dimensions(self) -> None:
        """Test that mismatched dimensions raise errors."""
        with pytest.raises(ValueError, match="same length"):
            ChunkLayout(grid_origin=(0, 0), chunk_shape=(10,))

    def test_repr(self) -> None:
        """Test string representation."""
        layout = ChunkLayout(grid_origin=(5, 10), chunk_shape=(10, 20))
        assert repr(layout) == "ChunkLayout(grid_origin=(5, 10), chunk_shape=(10, 20))"


class TestArrayChunkLayout:
    """Tests for chunk_layout property on Array."""

    @pytest.fixture
    def base_array(self) -> Array:
        """Create a base array for testing."""
        store = MemoryStore()
        zarr.create_array(store, shape=(100,), chunks=(10,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(100, dtype="i4"))
        return arr

    def test_chunk_layout_basic(self, base_array: Array) -> None:
        """Test chunk_layout on a basic array."""
        layout = base_array.chunk_layout
        assert layout.grid_origin == (0,)
        assert layout.chunk_shape == (10,)

    def test_chunk_layout_is_aligned(self, base_array: Array) -> None:
        """Test using chunk_layout to check alignment."""
        layout = base_array.chunk_layout
        assert layout.is_aligned((0,))
        assert layout.is_aligned((10,))
        assert layout.is_aligned((50,))
        assert not layout.is_aligned((5,))
        assert not layout.is_aligned((25,))

    def test_chunk_layout_after_slice(self, base_array: Array) -> None:
        """Test that chunk_layout reflects the sliced domain's coordinate system."""
        # Slice the array - domain becomes [25, 75)
        sliced = base_array[25:75]

        # The chunk layout should still have the same grid boundaries
        # but expressed in the original coordinate system (since slicing
        # preserves index_transform)
        layout = sliced.chunk_layout
        assert layout.chunk_shape == (10,)
        assert layout.grid_origin == (0,)  # Slicing preserves index_transform=(0,)

        # Check alignment in the sliced domain's coordinates
        assert layout.is_aligned((30,))  # 30 is a chunk boundary
        assert layout.is_aligned((40,))
        assert not layout.is_aligned((25,))  # 25 is not a chunk boundary
        assert not layout.is_aligned((35,))

    def test_chunk_layout_after_with_domain(self, base_array: Array) -> None:
        """Test chunk_layout after with_domain shifts the grid."""
        # with_domain sets index_transform = domain.origin
        # So domain coordinate 10 maps to storage coordinate 0
        new_domain = IndexDomain(inclusive_min=(10,), exclusive_max=(20,))
        view = base_array.with_domain(new_domain)

        layout = view.chunk_layout
        assert layout.chunk_shape == (10,)
        # Grid origin is at index_transform = (10,)
        # So chunk boundaries are at 10, 20, 30, ...
        assert layout.grid_origin == (10,)

        assert layout.is_aligned((10,))  # Domain origin is aligned
        assert layout.is_aligned((20,))
        assert not layout.is_aligned((15,))

    def test_chunk_layout_iter_chunk_domains(self, base_array: Array) -> None:
        """Test using chunk_layout to iterate over chunks."""
        layout = base_array.chunk_layout

        # Get chunks overlapping [25, 55)
        domain = IndexDomain(inclusive_min=(25,), exclusive_max=(55,))
        chunks = list(layout.iter_chunk_domains(domain))

        # Should overlap chunks 2, 3, 4, 5 (covering 20-60)
        assert len(chunks) == 4

        # First chunk: coords (2,), intersection [25, 30)
        coords, dom = chunks[0]
        assert coords == (2,)
        assert dom == IndexDomain(inclusive_min=(25,), exclusive_max=(30,))

        # Last chunk: coords (5,), intersection [50, 55)
        coords, dom = chunks[-1]
        assert coords == (5,)
        assert dom == IndexDomain(inclusive_min=(50,), exclusive_max=(55,))

    def test_chunk_layout_aligned_domain(self, base_array: Array) -> None:
        """Test finding aligned subdomain."""
        layout = base_array.chunk_layout

        # Find aligned subdomain of [25, 75)
        domain = IndexDomain(inclusive_min=(25,), exclusive_max=(75,))
        aligned = layout.aligned_domain(domain)

        # Should round to [30, 70)
        assert aligned.inclusive_min == (30,)
        assert aligned.exclusive_max == (70,)


class TestArrayDomain:
    """Tests for Array with domain tracking."""

    @pytest.fixture
    def base_array(self) -> Array:
        """Create a base array for testing."""
        store = MemoryStore()
        zarr.create_array(store, shape=(100,), chunks=(10,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        # Fill with test data
        arr.setitem(slice(None), np.arange(100, dtype="i4"))
        return arr

    @pytest.fixture
    def multidim_array(self) -> Array:
        """Create a multi-dimensional array for testing."""
        store = MemoryStore()
        zarr.create_array(store, shape=(10, 20, 30), chunks=(5, 10, 15), dtype="i4", fill_value=0)
        arr = Array.open(store)
        data = np.arange(10 * 20 * 30, dtype="i4").reshape((10, 20, 30))
        arr.setitem(slice(None), data)
        return arr

    def test_array_domain(self, base_array: Array) -> None:
        """Test that slicing an array changes the origin and domain of the array."""
        # Check initial domain
        assert base_array.origin == (0,)
        assert base_array.domain == IndexDomain.from_shape((100,))
        assert base_array.shape == (100,)

        # Slice the array using absolute coordinates
        sliced = base_array[20:40]

        # Check that we got a new Array, not data
        assert isinstance(sliced, Array)

        # Check that the domain reflects the slice
        assert sliced.origin == (20,)
        assert sliced.domain.inclusive_min == (20,)
        assert sliced.domain.exclusive_max == (40,)
        assert sliced.shape == (20,)

    def test_chained_slicing(self, base_array: Array) -> None:
        """Test that chained slicing works correctly with absolute coordinates."""
        # First slice: [20:60) -> domain [20, 60)
        first = base_array[20:60]
        assert first.origin == (20,)
        assert first.shape == (40,)

        # Second slice: [30:40) in absolute coordinates
        # (these coordinates are within the domain [20, 60))
        second = first[30:40]
        assert second.origin == (30,)
        assert second.shape == (10,)

        # Verify we can resolve to the correct data
        data = second.resolve()
        expected = np.arange(30, 40, dtype="i4")
        np.testing.assert_array_equal(data, expected)

    def test_multidim_slicing(self, multidim_array: Array) -> None:
        """Test slicing in multiple dimensions."""
        # Slice in all dimensions using absolute coordinates
        sliced = multidim_array[2:8, 5:15, 10:25]

        assert sliced.origin == (2, 5, 10)
        assert sliced.shape == (6, 10, 15)
        assert sliced.domain.inclusive_min == (2, 5, 10)
        assert sliced.domain.exclusive_max == (8, 15, 25)

    def test_partial_slicing(self, multidim_array: Array) -> None:
        """Test slicing only some dimensions."""
        # Slice only first dimension
        sliced = multidim_array[3:7]

        assert sliced.origin == (3, 0, 0)
        assert sliced.shape == (4, 20, 30)

    def test_ellipsis_slicing(self, multidim_array: Array) -> None:
        """Test slicing with ellipsis."""
        # Ellipsis at the end
        sliced = multidim_array[3:7, ...]
        assert sliced.origin == (3, 0, 0)
        assert sliced.shape == (4, 20, 30)

        # Ellipsis at the start
        sliced = multidim_array[..., 10:20]
        assert sliced.origin == (0, 0, 10)
        assert sliced.shape == (10, 20, 10)


class TestWithDomain:
    """Tests for the with_domain() method."""

    @pytest.fixture
    def base_array(self) -> Array:
        """Create a base array for testing."""
        store = MemoryStore()
        zarr.create_array(store, shape=(100,), chunks=(10,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(100, dtype="i4"))
        return arr

    def test_with_domain(self, base_array: Array) -> None:
        """Test that with_domain remaps domain coordinates to storage coordinates.

        with_domain() creates a view where domain.origin maps to storage coordinate 0.
        This follows TensorStore's IndexTransform semantic.
        """
        # Create new domain [10, 20) - this will MAP to storage [0, 10)
        new_domain = IndexDomain(inclusive_min=(10,), exclusive_max=(20,))

        # Use with_domain to create a new view
        view = base_array.with_domain(new_domain)

        # Check properties
        assert view.origin == (10,)
        assert view.shape == (10,)
        assert view.domain == new_domain

        # Resolve and check data
        # Domain [10, 20) maps to storage [0, 10), so we get storage[0:10]
        data = view.resolve()
        expected = np.arange(0, 10, dtype="i4")
        np.testing.assert_array_equal(data, expected)

    def test_with_domain_beyond_bounds(self, base_array: Array) -> None:
        """Test that with_domain allows domains beyond storage bounds.

        When domain extends beyond storage (after coordinate remapping),
        out-of-bounds regions are filled with fill_value.
        """
        # Create a domain [90, 110) which maps to storage [0, 20)
        # But storage only has 100 elements, so storage [0, 20) is valid
        # This test should use a domain that goes beyond the remapped storage bounds
        # Let's use domain [0, 110) which maps to storage [0, 110) - last 10 are OOB
        extended_domain = IndexDomain(inclusive_min=(0,), exclusive_max=(110,))
        view = base_array.with_domain(extended_domain)

        assert view.origin == (0,)
        assert view.shape == (110,)

        # Domain [0, 110) maps to storage [0, 110)
        # Storage only has [0, 100), so last 10 values are fill_value (0)
        data = view.resolve()
        expected = np.concatenate([np.arange(0, 100, dtype="i4"), np.zeros(10, dtype="i4")])
        np.testing.assert_array_equal(data, expected)

    def test_with_domain_negative_origin(self, base_array: Array) -> None:
        """Test with_domain with negative origin.

        Domain with negative origin maps negative coords to storage coordinates.
        Domain.origin maps to storage 0, so domain -5 maps to storage 0.
        """
        # Create a domain with negative origin [-5, 5)
        # This maps to storage [0, 10)
        neg_domain = IndexDomain(inclusive_min=(-5,), exclusive_max=(5,))
        view = base_array.with_domain(neg_domain)

        assert view.origin == (-5,)
        assert view.shape == (10,)

        # Domain [-5, 5) maps to storage [0, 10)
        # So we get storage[0:10] = [0, 1, 2, ..., 9]
        data = view.resolve()
        expected = np.arange(0, 10, dtype="i4")
        np.testing.assert_array_equal(data, expected)

    def test_with_domain_wrong_ndim(self, base_array: Array) -> None:
        """Test that with_domain raises error for wrong number of dimensions."""
        wrong_ndim = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 10))

        with pytest.raises(ValueError, match="same number of dimensions"):
            base_array.with_domain(wrong_ndim)

    def test_with_domain_preserves_store(self, base_array: Array) -> None:
        """Test that with_domain preserves the store reference."""
        new_domain = IndexDomain(inclusive_min=(50,), exclusive_max=(60,))
        view = base_array.with_domain(new_domain)

        # Should share the same source
        assert view.source is base_array.source
        assert view.source.store_path == base_array.source.store_path


class TestAbsoluteIndexing:
    """Tests for TensorStore-style absolute coordinate indexing.

    Key insight: indices are ABSOLUTE coordinates in the domain, not offsets.
    Negative indices mean negative coordinates, not "from the end".
    """

    @pytest.fixture
    def standard_array(self) -> Array:
        """Create a standard array with domain [0, 10)."""
        store = MemoryStore()
        zarr.create_array(store, shape=(10,), chunks=(5,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(10, dtype="i4"))
        return arr

    @pytest.fixture
    def shifted_array(self) -> Array:
        """Create an array with domain [10, 20)."""
        store = MemoryStore()
        zarr.create_array(store, shape=(10,), chunks=(5,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(10, dtype="i4"))
        # Shift domain to [10, 20)
        return arr.with_domain(IndexDomain(inclusive_min=(10,), exclusive_max=(20,)))

    @pytest.fixture
    def negative_domain_array(self) -> Array:
        """Create an array with domain [-5, 5)."""
        store = MemoryStore()
        zarr.create_array(store, shape=(10,), chunks=(5,), dtype="i4", fill_value=-1)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(10, dtype="i4"))
        # Shift domain to [-5, 5)
        return arr.with_domain(IndexDomain(inclusive_min=(-5,), exclusive_max=(5,)))

    def test_absolute_integer_index(self, standard_array: Array) -> None:
        """Test that integer indices are absolute coordinates."""
        # arr[5] means coordinate 5, not "5th element"
        sliced = standard_array[5]
        assert sliced.origin == (5,)
        assert sliced.shape == (1,)

        data = sliced.resolve()
        assert data == 5

    def test_absolute_index_shifted_domain(self, shifted_array: Array) -> None:
        """Test absolute indexing with a shifted domain."""
        # Domain is [10, 20), so arr[15] selects coordinate 15
        sliced = shifted_array[15]
        assert sliced.origin == (15,)
        assert sliced.shape == (1,)

        # Coordinate 15 maps to storage index 5 (since domain starts at 10)
        data = sliced.resolve()
        assert data == 5

    def test_index_below_domain_raises(self, shifted_array: Array) -> None:
        """Test that indexing below domain raises error."""
        # Domain is [10, 20), so coordinate 5 is out of bounds
        with pytest.raises(IndexError, match="out of bounds"):
            shifted_array[5]

    def test_negative_index_is_coordinate(self, negative_domain_array: Array) -> None:
        """Test that negative indices are actual coordinates, not 'from end'."""
        # Domain is [-5, 5), so arr[-3] means coordinate -3
        sliced = negative_domain_array[-3]
        assert sliced.origin == (-3,)
        assert sliced.shape == (1,)

        # Coordinate -3 maps to storage index 2 (since domain starts at -5)
        data = sliced.resolve()
        assert data == 2

    def test_negative_index_out_of_bounds(self, standard_array: Array) -> None:
        """Test that negative indices outside domain raise errors."""
        # Domain is [0, 10), so -1 is out of bounds (it's not "last element")
        with pytest.raises(IndexError, match="out of bounds"):
            standard_array[-1]

    def test_absolute_slice(self, shifted_array: Array) -> None:
        """Test that slice bounds are absolute coordinates."""
        # Domain is [10, 20), slice [12:18)
        sliced = shifted_array[12:18]
        assert sliced.origin == (12,)
        assert sliced.shape == (6,)

        data = sliced.resolve()
        # Coordinates 12-17 map to storage indices 2-7
        expected = np.arange(2, 8, dtype="i4")
        np.testing.assert_array_equal(data, expected)

    def test_slice_with_negative_coordinates(self, negative_domain_array: Array) -> None:
        """Test slicing with negative coordinate bounds."""
        # Domain is [-5, 5), slice [-3:2)
        sliced = negative_domain_array[-3:2]
        assert sliced.origin == (-3,)
        assert sliced.shape == (5,)

        data = sliced.resolve()
        # Coordinates -3 to 1 map to storage indices 2-6
        expected = np.arange(2, 7, dtype="i4")
        np.testing.assert_array_equal(data, expected)

    def test_slice_clamps_to_domain(self, standard_array: Array) -> None:
        """Test that slices clamp to domain bounds (no error for OOB slices)."""
        # Domain is [0, 10), slice [5:100) clamps to [5:10)
        sliced = standard_array[5:100]
        assert sliced.origin == (5,)
        assert sliced.shape == (5,)

    def test_slice_before_domain_clamps(self, shifted_array: Array) -> None:
        """Test that slices starting before domain clamp correctly."""
        # Domain is [10, 20), slice [0:15) clamps to [10:15)
        sliced = shifted_array[0:15]
        assert sliced.origin == (10,)
        assert sliced.shape == (5,)

    def test_chained_absolute_indexing(self, standard_array: Array) -> None:
        """Test chaining with absolute coordinates."""
        # First slice: [2:8) -> domain [2, 8)
        first = standard_array[2:8]
        assert first.origin == (2,)
        assert first.shape == (6,)

        # Second slice: [4:6) - these are absolute coordinates within [2, 8)
        second = first[4:6]
        assert second.origin == (4,)
        assert second.shape == (2,)

        data = second.resolve()
        expected = np.arange(4, 6, dtype="i4")
        np.testing.assert_array_equal(data, expected)


class TestResolve:
    """Tests for the resolve() method that materializes data."""

    @pytest.fixture
    def filled_array(self) -> Array:
        """Create an array filled with sequential data."""
        store = MemoryStore()
        zarr.create_array(store, shape=(100,), chunks=(10,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(100, dtype="i4"))
        return arr

    def test_resolve_full_array(self, filled_array: Array) -> None:
        """Test resolving the full array."""
        data = filled_array.resolve()
        expected = np.arange(100, dtype="i4")
        np.testing.assert_array_equal(data, expected)

    def test_resolve_sliced_array(self, filled_array: Array) -> None:
        """Test resolving a sliced array."""
        sliced = filled_array[25:75]
        data = sliced.resolve()
        expected = np.arange(25, 75, dtype="i4")
        np.testing.assert_array_equal(data, expected)

    def test_resolve_chain_slices(self, filled_array: Array) -> None:
        """Test resolving after chaining multiple slices."""
        # Chain multiple slices with absolute coordinates
        result = filled_array[10:90][30:70][40:60]
        data = result.resolve()
        expected = np.arange(40, 60, dtype="i4")
        np.testing.assert_array_equal(data, expected)


class TestIntegerIndexing:
    """Tests for integer (single element) indexing."""

    @pytest.fixture
    def array_1d(self) -> Array:
        """Create a 1D array."""
        store = MemoryStore()
        zarr.create_array(store, shape=(10,), chunks=(5,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(10, dtype="i4"))
        return arr

    @pytest.fixture
    def array_3d(self) -> Array:
        """Create a 3D array."""
        store = MemoryStore()
        zarr.create_array(store, shape=(5, 6, 7), chunks=(2, 3, 4), dtype="i4", fill_value=0)
        arr = Array.open(store)
        data = np.arange(5 * 6 * 7, dtype="i4").reshape((5, 6, 7))
        arr.setitem(slice(None), data)
        return arr

    def test_integer_index_preserves_dimension(self, array_1d: Array) -> None:
        """Test that integer indexing preserves the dimension (unlike NumPy)."""
        # In lazy indexing, arr[5] should give a length-1 array, not drop the dimension
        sliced = array_1d[5]
        assert sliced.ndim == 1
        assert sliced.shape == (1,)
        assert sliced.origin == (5,)

    def test_integer_index_3d(self, array_3d: Array) -> None:
        """Test integer indexing in 3D."""
        # Single integer should give a length-1 slice in that dimension
        sliced = array_3d[2]
        assert sliced.shape == (1, 6, 7)
        assert sliced.origin == (2, 0, 0)

    def test_mixed_integer_slice(self, array_3d: Array) -> None:
        """Test mixing integer and slice indexing."""
        sliced = array_3d[2, 1:4, 3]
        assert sliced.shape == (1, 3, 1)
        assert sliced.origin == (2, 1, 3)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def array_1d(self) -> Array:
        """Create a 1D array."""
        store = MemoryStore()
        zarr.create_array(store, shape=(10,), chunks=(5,), dtype="i4", fill_value=0)
        return Array.open(store)

    def test_empty_slice(self, array_1d: Array) -> None:
        """Test that an empty slice creates a zero-size array."""
        sliced = array_1d[5:5]
        assert sliced.shape == (0,)
        assert sliced.origin == (5,)

    def test_slice_step_not_one_raises(self, array_1d: Array) -> None:
        """Test that step != 1 raises an error."""
        with pytest.raises(IndexError, match="only supports step=1"):
            array_1d[::2]

    def test_too_many_indices(self, array_1d: Array) -> None:
        """Test that too many indices raises an error."""
        with pytest.raises(IndexError, match="too many indices"):
            array_1d[1, 2]

    def test_slice_clamps_to_bounds(self, array_1d: Array) -> None:
        """Test that slices clamp to array bounds (like NumPy)."""
        # Slice extends beyond bounds
        sliced = array_1d[5:100]
        assert sliced.shape == (5,)  # Clamped to (5, 10)
        assert sliced.origin == (5,)
        assert sliced.domain.exclusive_max == (10,)

    def test_open_with_custom_domain(self) -> None:
        """Test opening an array with a custom domain."""
        store = MemoryStore()
        zarr.create_array(store, shape=(100,), chunks=(10,), dtype="i4", fill_value=0)

        # Open with a custom domain
        custom_domain = IndexDomain(inclusive_min=(25,), exclusive_max=(75,))
        arr = Array.open(store, domain=custom_domain)

        assert arr.domain == custom_domain
        assert arr.origin == (25,)
        assert arr.shape == (50,)


class TestMerge:
    """Tests for merge with unified Array type."""

    @pytest.fixture
    def base_array(self) -> Array:
        """Create a base array for testing."""
        store = MemoryStore()
        zarr.create_array(store, shape=(100,), chunks=(10,), dtype="i4", fill_value=-1)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(100, dtype="i4"))
        return arr

    def test_merge_basic(self, base_array: Array) -> None:
        """Test basic concatenation of two slices."""
        left = base_array[0:30]
        right = base_array[70:100]

        combined = merge([left, right])

        assert combined.domain == IndexDomain(inclusive_min=(0,), exclusive_max=(100,))
        assert combined.shape == (100,)
        assert combined.dtype == base_array.dtype

    def test_merge_resolve(self, base_array: Array) -> None:
        """Test that merge resolves correctly with gaps filled."""
        left = base_array[0:30]
        right = base_array[70:100]

        combined = merge([left, right])
        data = combined.resolve()

        # Check that we got the expected data
        np.testing.assert_array_equal(data[0:30], np.arange(0, 30, dtype="i4"))
        np.testing.assert_array_equal(data[70:100], np.arange(70, 100, dtype="i4"))
        # Gap should be filled with fill_value (-1)
        np.testing.assert_array_equal(data[30:70], np.full(40, -1, dtype="i4"))

    def test_merge_inverts_slicing(self, base_array: Array) -> None:
        """Test that merge is the inverse of slicing."""
        # Slice into chunks
        chunks = [base_array[i : i + 10] for i in range(0, 100, 10)]

        # Reassemble
        reassembled = merge(chunks)

        # Should be identical to original
        np.testing.assert_array_equal(reassembled.resolve(), base_array.resolve())

    def test_merge_overlapping_last_wins(self, base_array: Array) -> None:
        """Test that overlapping regions use last-write-wins."""
        # Create two overlapping slices
        a = base_array[0:60]
        b = base_array[40:100]

        # b comes after a, so b's data should win in [40, 60)
        combined = merge([a, b])
        data = combined.resolve()

        # All data should match original since both come from same source
        np.testing.assert_array_equal(data, base_array.resolve())

    def test_merge_with_explicit_domain(self, base_array: Array) -> None:
        """Test merge with explicitly specified domain."""
        left = base_array[10:30]
        right = base_array[70:90]

        # Specify a smaller domain than bounding box
        explicit_domain = IndexDomain(inclusive_min=(20,), exclusive_max=(80,))
        combined = merge([left, right], domain=explicit_domain)

        assert combined.domain == explicit_domain
        assert combined.shape == (60,)

        data = combined.resolve()
        # left contributes [20, 30), right contributes [70, 80)
        np.testing.assert_array_equal(data[0:10], np.arange(20, 30, dtype="i4"))
        np.testing.assert_array_equal(data[50:60], np.arange(70, 80, dtype="i4"))
        # Gap is fill_value
        np.testing.assert_array_equal(data[10:50], np.full(40, -1, dtype="i4"))

    def test_merge_custom_fill_value(self, base_array: Array) -> None:
        """Test merge with custom fill value."""
        left = base_array[0:30]
        right = base_array[70:100]

        combined = merge([left, right], fill_value=999)
        data = combined.resolve()

        # Gap should be filled with custom fill_value
        np.testing.assert_array_equal(data[30:70], np.full(40, 999, dtype="i4"))

    def test_merge_preserves_dtype(self, base_array: Array) -> None:
        """Test that merge preserves dtype."""
        left = base_array[0:50]
        right = base_array[50:100]

        combined = merge([left, right])
        assert combined.dtype == base_array.dtype

        data = combined.resolve()
        assert data.dtype == base_array.dtype

    def test_merge_single_array(self, base_array: Array) -> None:
        """Test merge with a single array."""
        sliced = base_array[25:75]
        combined = merge([sliced])

        assert combined.domain == sliced.domain
        np.testing.assert_array_equal(combined.resolve(), sliced.resolve())

    def test_merge_empty_raises(self) -> None:
        """Test that merge with no arrays raises."""
        with pytest.raises(ValueError, match="at least one array"):
            merge([])

    def test_merge_mismatched_ndim_raises(self, base_array: Array) -> None:
        """Test that merge with mismatched dimensions raises."""
        store = MemoryStore()
        zarr.create_array(store, shape=(10, 10), chunks=(5, 5), dtype="i4", fill_value=0)
        arr_2d = Array.open(store)

        with pytest.raises(ValueError, match="same number of dimensions"):
            merge([base_array[0:10], arr_2d[0:5, 0:5]])

    def test_merge_mismatched_dtype_raises(self) -> None:
        """Test that merge with mismatched dtypes raises."""
        store1 = MemoryStore()
        zarr.create_array(store1, shape=(10,), chunks=(5,), dtype="i4", fill_value=0)
        arr1 = Array.open(store1)

        store2 = MemoryStore()
        zarr.create_array(store2, shape=(10,), chunks=(5,), dtype="f8", fill_value=0)
        arr2 = Array.open(store2)

        with pytest.raises(ValueError, match="same dtype"):
            merge([arr1, arr2])

    def test_merge_2d(self) -> None:
        """Test merge with 2D arrays."""
        store = MemoryStore()
        zarr.create_array(store, shape=(20, 20), chunks=(10, 10), dtype="i4", fill_value=-1)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(400, dtype="i4").reshape(20, 20))

        # Slice into quadrants
        top_left = arr[0:10, 0:10]
        top_right = arr[0:10, 10:20]
        bottom_left = arr[10:20, 0:10]
        bottom_right = arr[10:20, 10:20]

        # Reassemble
        combined = merge([top_left, top_right, bottom_left, bottom_right])

        assert combined.domain == arr.domain
        np.testing.assert_array_equal(combined.resolve(), arr.resolve())

    def test_merge_2d_with_gap(self) -> None:
        """Test 2D merge with a gap."""
        store = MemoryStore()
        zarr.create_array(store, shape=(20, 20), chunks=(10, 10), dtype="i4", fill_value=-1)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(400, dtype="i4").reshape(20, 20))

        # Only top-left and bottom-right
        top_left = arr[0:10, 0:10]
        bottom_right = arr[10:20, 10:20]

        combined = merge([top_left, bottom_right])

        assert combined.domain == IndexDomain(inclusive_min=(0, 0), exclusive_max=(20, 20))

        data = combined.resolve()
        # Top-left should have data
        np.testing.assert_array_equal(data[0:10, 0:10], arr.resolve()[0:10, 0:10])
        # Bottom-right should have data
        np.testing.assert_array_equal(data[10:20, 10:20], arr.resolve()[10:20, 10:20])
        # Gaps should be fill_value
        np.testing.assert_array_equal(data[0:10, 10:20], np.full((10, 10), -1, dtype="i4"))
        np.testing.assert_array_equal(data[10:20, 0:10], np.full((10, 10), -1, dtype="i4"))

    def test_merge_repr(self, base_array: Array) -> None:
        """Test concatenated Array string representation."""
        combined = merge([base_array[0:30], base_array[70:100]])
        repr_str = repr(combined)
        assert "Array" in repr_str
        assert "sources=2" in repr_str

    def test_merge_returns_array(self, base_array: Array) -> None:
        """Test that merge returns an Array."""
        combined = merge([base_array[0:50], base_array[50:100]])
        assert isinstance(combined, Array)

    def test_merge_nested(self, base_array: Array) -> None:
        """Test that concatenated Arrays can be nested."""
        # Create two concatenated Arrays
        left_concat = merge([base_array[0:20], base_array[20:40]])
        right_concat = merge([base_array[60:80], base_array[80:100]])

        # Concat them together
        combined = merge([left_concat, right_concat])

        assert combined.domain == IndexDomain(inclusive_min=(0,), exclusive_max=(100,))

        data = combined.resolve()
        np.testing.assert_array_equal(data[0:40], np.arange(0, 40, dtype="i4"))
        np.testing.assert_array_equal(data[60:100], np.arange(60, 100, dtype="i4"))
        # Gap filled
        np.testing.assert_array_equal(data[40:60], np.full(20, -1, dtype="i4"))

    def test_merge_slicing(self, base_array: Array) -> None:
        """Test slicing a concatenated Array."""
        combined = merge([base_array[0:30], base_array[70:100]])

        # Slice the combined array
        sliced = combined[20:80]

        assert sliced.domain == IndexDomain(inclusive_min=(20,), exclusive_max=(80,))
        assert sliced.shape == (60,)

        data = sliced.resolve()
        np.testing.assert_array_equal(data[0:10], np.arange(20, 30, dtype="i4"))
        np.testing.assert_array_equal(data[50:60], np.arange(70, 80, dtype="i4"))
        np.testing.assert_array_equal(data[10:50], np.full(40, -1, dtype="i4"))

    def test_merge_from_chunk_layout(self, base_array: Array) -> None:
        """Test reassembling an array from its chunks using chunk_layout."""
        layout = base_array.chunk_layout

        # Get each chunk as a slice
        chunks = []
        for _chunk_coords, chunk_domain in layout.iter_chunk_domains(base_array.domain):
            chunk_slice = base_array[chunk_domain.inclusive_min[0] : chunk_domain.exclusive_max[0]]
            chunks.append(chunk_slice)

        # Reassemble
        reassembled = merge(chunks)

        np.testing.assert_array_equal(reassembled.resolve(), base_array.resolve())


# ---------------------------------------------------------------------------
# Advanced indexing tests
# ---------------------------------------------------------------------------


class TestAdvancedIndexingDetection:
    """Test that advanced indexing creates the correct PendingSelection."""

    @pytest.fixture
    def arr_1d(self) -> Array:
        store = MemoryStore()
        zarr.create_array(store, shape=(20,), chunks=(5,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(20, dtype="i4"))
        return arr

    @pytest.fixture
    def arr_2d(self) -> Array:
        store = MemoryStore()
        zarr.create_array(store, shape=(10, 20), chunks=(5, 10), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(200, dtype="i4").reshape(10, 20))
        return arr

    def test_bool_mask_creates_pending(self, arr_1d: Array) -> None:
        mask = np.array([i % 3 == 0 for i in range(20)])
        result = arr_1d[mask]
        assert result._pending_selection is not None
        assert result._pending_selection.kind == SelectionKind.MASK

    def test_int_array_creates_pending(self, arr_1d: Array) -> None:
        result = arr_1d[np.array([1, 5, 10])]
        assert result._pending_selection is not None
        # 1D int array on 1D array → treated as fancy indexing
        assert result._pending_selection.kind in (
            SelectionKind.COORDINATE,
            SelectionKind.ORTHOGONAL,
        )

    def test_basic_indexing_unchanged(self, arr_1d: Array) -> None:
        result = arr_1d[5:10]
        assert result._pending_selection is None
        assert result.domain == IndexDomain(inclusive_min=(5,), exclusive_max=(10,))

    def test_oindex_creates_orthogonal(self, arr_2d: Array) -> None:
        result = arr_2d.oindex[np.array([1, 3]), :]
        assert result._pending_selection is not None
        assert result._pending_selection.kind == SelectionKind.ORTHOGONAL

    def test_vindex_creates_coordinate(self, arr_2d: Array) -> None:
        result = arr_2d.vindex[np.array([1, 3]), np.array([5, 10])]
        assert result._pending_selection is not None
        assert result._pending_selection.kind == SelectionKind.COORDINATE


class TestAdvancedShape:
    """Test that shape is computed correctly for advanced selections."""

    @pytest.fixture
    def arr_1d(self) -> Array:
        store = MemoryStore()
        zarr.create_array(store, shape=(20,), chunks=(5,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(20, dtype="i4"))
        return arr

    @pytest.fixture
    def arr_2d(self) -> Array:
        store = MemoryStore()
        zarr.create_array(store, shape=(10, 20), chunks=(5, 10), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(200, dtype="i4").reshape(10, 20))
        return arr

    def test_mask_shape_is_1d(self, arr_1d: Array) -> None:
        mask = np.zeros(20, dtype=bool)
        mask[3] = True
        mask[7] = True
        mask[15] = True
        result = arr_1d[mask]
        assert result.shape == (3,)
        assert result.ndim == 1

    def test_mask_shape_2d(self, arr_2d: Array) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[0, 0] = True
        mask[5, 10] = True
        mask[9, 19] = True
        result = arr_2d[mask]
        assert result.shape == (3,)
        assert result.ndim == 1

    def test_oindex_shape(self, arr_2d: Array) -> None:
        result = arr_2d.oindex[np.array([1, 3, 5]), :]
        assert result.shape == (3, 20)

    def test_oindex_shape_both_dims(self, arr_2d: Array) -> None:
        result = arr_2d.oindex[np.array([1, 3]), np.array([5, 10, 15])]
        assert result.shape == (2, 3)

    def test_vindex_shape(self, arr_2d: Array) -> None:
        result = arr_2d.vindex[np.array([1, 3, 5]), np.array([5, 10, 15])]
        assert result.shape == (3,)

    def test_empty_mask_shape(self, arr_1d: Array) -> None:
        mask = np.zeros(20, dtype=bool)
        result = arr_1d[mask]
        assert result.shape == (0,)


class TestAdvancedResolve:
    """Test that advanced indexing resolve produces correct data."""

    @pytest.fixture
    def arr_1d(self) -> Array:
        store = MemoryStore()
        zarr.create_array(store, shape=(20,), chunks=(5,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(20, dtype="i4"))
        return arr

    @pytest.fixture
    def np_1d(self) -> np.ndarray:
        return np.arange(20, dtype="i4")

    @pytest.fixture
    def arr_2d(self) -> Array:
        store = MemoryStore()
        zarr.create_array(store, shape=(10, 20), chunks=(5, 10), dtype="i4", fill_value=0)
        arr = Array.open(store)
        data = np.arange(200, dtype="i4").reshape(10, 20)
        arr.setitem(slice(None), data)
        return arr

    @pytest.fixture
    def np_2d(self) -> np.ndarray:
        return np.arange(200, dtype="i4").reshape(10, 20)

    def test_resolve_bool_mask_1d(self, arr_1d: Array, np_1d: np.ndarray) -> None:
        mask = np.array([i % 3 == 0 for i in range(20)])
        result = arr_1d[mask].resolve()
        expected = np_1d[mask]
        np.testing.assert_array_equal(result, expected)

    def test_resolve_bool_mask_2d(self, arr_2d: Array, np_2d: np.ndarray) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[0, 0] = True
        mask[3, 15] = True
        mask[9, 19] = True
        result = arr_2d[mask].resolve()
        expected = np_2d[mask]
        np.testing.assert_array_equal(result, expected)

    def test_resolve_oindex_int_arrays(self, arr_2d: Array, np_2d: np.ndarray) -> None:
        rows = np.array([1, 3, 7])
        cols = np.array([0, 5, 10, 15, 19])
        result = arr_2d.oindex[rows, cols].resolve()
        expected = np_2d[np.ix_(rows, cols)]
        np.testing.assert_array_equal(result, expected)

    def test_resolve_oindex_with_slice(self, arr_2d: Array, np_2d: np.ndarray) -> None:
        rows = np.array([2, 5, 8])
        result = arr_2d.oindex[rows, :].resolve()
        expected = np_2d[rows, :]
        np.testing.assert_array_equal(result, expected)

    def test_resolve_oindex_bool_per_dim(self, arr_2d: Array, np_2d: np.ndarray) -> None:
        row_mask = np.array([True, False, True, False, True, False, True, False, True, False])
        col_sel = np.array([0, 5, 10])
        result = arr_2d.oindex[row_mask, col_sel].resolve()
        expected = np_2d[np.ix_(np.nonzero(row_mask)[0], col_sel)]
        np.testing.assert_array_equal(result, expected)

    def test_resolve_vindex_coordinate(self, arr_2d: Array, np_2d: np.ndarray) -> None:
        rows = np.array([0, 3, 9])
        cols = np.array([1, 10, 19])
        result = arr_2d.vindex[rows, cols].resolve()
        expected = np_2d[rows, cols]
        np.testing.assert_array_equal(result, expected)

    def test_resolve_empty_mask(self, arr_1d: Array) -> None:
        mask = np.zeros(20, dtype=bool)
        result = arr_1d[mask].resolve()
        assert result.shape == (0,)

    def test_resolve_1d_int_array(self, arr_1d: Array, np_1d: np.ndarray) -> None:
        idx = np.array([0, 5, 10, 15, 19])
        result = arr_1d[idx].resolve()
        expected = np_1d[idx]
        np.testing.assert_array_equal(result, expected)

    def test_resolve_with_shifted_domain(self) -> None:
        """Advanced indexing on an array with non-zero domain origin."""
        store = MemoryStore()
        zarr.create_array(store, shape=(20,), chunks=(5,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(20, dtype="i4"))

        # Shift domain to [100, 120)
        shifted = arr.with_domain(IndexDomain(inclusive_min=(100,), exclusive_max=(120,)))

        # Boolean mask on shifted domain
        mask = np.array([i % 5 == 0 for i in range(20)])
        result = shifted[mask].resolve()
        expected = np.arange(20, dtype="i4")[mask]
        np.testing.assert_array_equal(result, expected)

    def test_resolve_with_sharded_array(self) -> None:
        """Advanced indexing on a sharded array."""
        store = MemoryStore()
        zarr.create_array(
            store,
            shape=(20,),
            chunks=(5,),
            shards=(10,),
            dtype="i4",
            fill_value=0,
        )
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(20, dtype="i4"))

        mask = np.array([i % 3 == 0 for i in range(20)])
        result = arr[mask].resolve()
        expected = np.arange(20, dtype="i4")[mask]
        np.testing.assert_array_equal(result, expected)


class TestAdvancedGuards:
    """Test guards that prevent invalid operations on advanced-indexed arrays."""

    @pytest.fixture
    def arr_1d(self) -> Array:
        store = MemoryStore()
        zarr.create_array(store, shape=(20,), chunks=(5,), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(20, dtype="i4"))
        return arr

    @pytest.fixture
    def arr_2d(self) -> Array:
        store = MemoryStore()
        zarr.create_array(store, shape=(10, 20), chunks=(5, 10), dtype="i4", fill_value=0)
        arr = Array.open(store)
        arr.setitem(slice(None), np.arange(200, dtype="i4").reshape(10, 20))
        return arr

    def test_chaining_advanced_raises(self, arr_1d: Array) -> None:
        mask = np.zeros(20, dtype=bool)
        mask[0] = True
        mask[5] = True
        advanced = arr_1d[mask]
        with pytest.raises(IndexError, match="pending advanced selection"):
            advanced[np.array([0])]

    def test_basic_after_advanced_raises(self, arr_1d: Array) -> None:
        mask = np.zeros(20, dtype=bool)
        mask[0] = True
        advanced = arr_1d[mask]
        with pytest.raises(IndexError, match="pending.*selection"):
            advanced[0:1]

    def test_merge_with_pending_raises(self, arr_1d: Array) -> None:
        mask = np.zeros(20, dtype=bool)
        mask[0] = True
        advanced = arr_1d[mask]
        with pytest.raises(ValueError, match="pending advanced selection"):
            merge([advanced])

    def test_with_domain_after_advanced_raises(self, arr_1d: Array) -> None:
        mask = np.zeros(20, dtype=bool)
        mask[0] = True
        advanced = arr_1d[mask]
        with pytest.raises(IndexError, match="pending.*selection"):
            advanced.with_domain(IndexDomain(inclusive_min=(0,), exclusive_max=(1,)))

    def test_basic_after_basic_works(self, arr_1d: Array) -> None:
        """Chaining basic indexing should still work."""
        result = arr_1d[5:15][7:12]
        assert result._pending_selection is None
        data = result.resolve()
        expected = np.arange(7, 12, dtype="i4")
        np.testing.assert_array_equal(data, expected)


class TestIndexTransform:
    """Tests for the IndexTransform dataclass."""

    def test_identity(self) -> None:
        domain = IndexDomain(inclusive_min=(10,), exclusive_max=(20,))
        t = IndexTransform.identity(domain)
        assert t.domain == domain
        assert t.offset == (0,)

    def test_from_shape(self) -> None:
        t = IndexTransform.from_shape((5, 10))
        assert t.domain == IndexDomain.from_shape((5, 10))
        assert t.offset == (0, 0)
        assert t.ndim == 2

    def test_ndim(self) -> None:
        t = IndexTransform(
            domain=IndexDomain(inclusive_min=(0, 0, 0), exclusive_max=(1, 2, 3)),
            offset=(0, 0, 0),
        )
        assert t.ndim == 3

    def test_offset_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            IndexTransform(
                domain=IndexDomain(inclusive_min=(0,), exclusive_max=(10,)),
                offset=(0, 0),
            )

    def test_storage_origin(self) -> None:
        # domain origin = 10, offset = 5 → storage origin = 10 - 5 = 5
        t = IndexTransform(
            domain=IndexDomain(inclusive_min=(10,), exclusive_max=(20,)),
            offset=(5,),
        )
        assert t.storage_origin == (5,)

    def test_storage_origin_identity(self) -> None:
        t = IndexTransform.from_shape((100,))
        assert t.storage_origin == (0,)

    def test_storage_origin_negative_offset(self) -> None:
        # domain origin = -5, offset = -5 → storage origin = -5 - (-5) = 0
        t = IndexTransform(
            domain=IndexDomain(inclusive_min=(-5,), exclusive_max=(5,)),
            offset=(-5,),
        )
        assert t.storage_origin == (0,)

    def test_narrow(self) -> None:
        t = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(100,)),
            offset=(0,),
        )
        narrowed = t.narrow(slice(10, 20))
        assert narrowed.domain == IndexDomain(inclusive_min=(10,), exclusive_max=(20,))
        # Offset preserved
        assert narrowed.offset == (0,)

    def test_narrow_preserves_offset(self) -> None:
        t = IndexTransform(
            domain=IndexDomain(inclusive_min=(10,), exclusive_max=(50,)),
            offset=(10,),
        )
        narrowed = t.narrow(slice(20, 30))
        assert narrowed.domain == IndexDomain(inclusive_min=(20,), exclusive_max=(30,))
        assert narrowed.offset == (10,)

    def test_compose_identity(self) -> None:
        """Composing two identity transforms over the same domain."""
        domain = IndexDomain(inclusive_min=(0,), exclusive_max=(100,))
        outer = IndexTransform.identity(domain)
        inner = IndexTransform.identity(domain)
        composed = outer.compose_or_none(inner)
        assert composed is not None
        assert composed.domain == domain
        assert composed.offset == (0,)

    def test_compose_narrowed_outer(self) -> None:
        """Outer has a narrowed domain, inner is identity over full range."""
        outer = IndexTransform(
            domain=IndexDomain(inclusive_min=(10,), exclusive_max=(20,)),
            offset=(0,),
        )
        inner = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(100,)),
            offset=(0,),
        )
        composed = outer.compose_or_none(inner)
        assert composed is not None
        assert composed.domain == IndexDomain(inclusive_min=(10,), exclusive_max=(20,))
        assert composed.offset == (0,)

    def test_compose_with_offset(self) -> None:
        """Outer has an offset (from with_domain), inner is identity."""
        # with_domain([-5, 5)) sets offset=(-5,)
        # intermediate = user - (-5) = user + 5
        # For user [-5,5), intermediate = [0, 10)
        # Inner domain [0,100) covers that fully
        outer = IndexTransform(
            domain=IndexDomain(inclusive_min=(-5,), exclusive_max=(5,)),
            offset=(-5,),
        )
        inner = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(100,)),
            offset=(0,),
        )
        composed = outer.compose_or_none(inner)
        assert composed is not None
        # Full outer domain is valid (intermediate [0,10) ⊂ [0,100))
        assert composed.domain == IndexDomain(inclusive_min=(-5,), exclusive_max=(5,))
        assert composed.offset == (-5,)

    def test_compose_partial_overlap(self) -> None:
        """Outer domain partially overlaps inner domain in intermediate space."""
        # outer: domain [0,20), offset 0 → intermediate [0,20)
        # inner: domain [10,30), offset 0
        # intersection in intermediate: [10,20)
        # back to user space: [10,20)
        outer = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(20,)),
            offset=(0,),
        )
        inner = IndexTransform(
            domain=IndexDomain(inclusive_min=(10,), exclusive_max=(30,)),
            offset=(0,),
        )
        composed = outer.compose_or_none(inner)
        assert composed is not None
        assert composed.domain == IndexDomain(inclusive_min=(10,), exclusive_max=(20,))
        assert composed.offset == (0,)

    def test_compose_disjoint(self) -> None:
        """Disjoint domains produce None."""
        outer = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(10,)),
            offset=(0,),
        )
        inner = IndexTransform(
            domain=IndexDomain(inclusive_min=(20,), exclusive_max=(30,)),
            offset=(0,),
        )
        assert outer.compose_or_none(inner) is None

    def test_compose_disjoint_with_offset(self) -> None:
        """Offset can make seemingly overlapping domains disjoint in intermediate space."""
        # outer: domain [0,10), offset 20 → intermediate [-20,-10)
        # inner: domain [0,10), offset 0
        # No overlap in intermediate space
        outer = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(10,)),
            offset=(20,),
        )
        inner = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(10,)),
            offset=(0,),
        )
        assert outer.compose_or_none(inner) is None

    def test_compose_offset_makes_overlap(self) -> None:
        """Offset can make seemingly disjoint domains overlap in intermediate space."""
        # outer: domain [100,110), offset 100 → intermediate [0,10)
        # inner: domain [0,100), offset 0
        # Overlap in intermediate: [0,10)
        # Back to user space: [100,110)
        outer = IndexTransform(
            domain=IndexDomain(inclusive_min=(100,), exclusive_max=(110,)),
            offset=(100,),
        )
        inner = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(100,)),
            offset=(0,),
        )
        composed = outer.compose_or_none(inner)
        assert composed is not None
        assert composed.domain == IndexDomain(inclusive_min=(100,), exclusive_max=(110,))
        assert composed.offset == (100,)

    def test_compose_both_offsets(self) -> None:
        """Both transforms have non-zero offsets."""
        outer = IndexTransform(
            domain=IndexDomain(inclusive_min=(10,), exclusive_max=(20,)),
            offset=(5,),
        )
        inner = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(100,)),
            offset=(3,),
        )
        # intermediate = user - 5. For user [10,20), intermediate = [5,15)
        # inner domain [0,100) covers that
        # composed offset = 5 + 3 = 8
        composed = outer.compose_or_none(inner)
        assert composed is not None
        assert composed.domain == IndexDomain(inclusive_min=(10,), exclusive_max=(20,))
        assert composed.offset == (8,)

    def test_compose_2d(self) -> None:
        """Composition works in multiple dimensions."""
        outer = IndexTransform(
            domain=IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 20)),
            offset=(0, 0),
        )
        inner = IndexTransform(
            domain=IndexDomain(inclusive_min=(0, 0), exclusive_max=(50, 50)),
            offset=(0, 0),
        )
        composed = outer.compose_or_none(inner)
        assert composed is not None
        assert composed.domain == IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 20))
        assert composed.offset == (0, 0)

    def test_repr(self) -> None:
        t = IndexTransform.from_shape((10,))
        r = repr(t)
        assert "IndexTransform" in r
        assert "domain=" in r
        assert "offset=" in r


class TestLayer:
    """Tests for the Layer dataclass."""

    def test_construction(self, base_array: Array) -> None:
        """Layer can be constructed from a transform and source."""
        source = base_array._layers[0].source
        transform = IndexTransform.from_shape((100,))
        layer = Layer(transform=transform, source=source)
        assert layer.transform == transform
        assert layer.source is source

    @pytest.fixture
    def base_array(self) -> Array:
        store = MemoryStore()
        arr = zarr.create(shape=(100,), chunks=(10,), dtype="i4", store=store)
        arr[:] = np.arange(100, dtype="i4")
        return Array.open(store=store)

    def test_layer_in_array(self, base_array: Array) -> None:
        """Array should have layers after construction."""
        assert len(base_array._layers) == 1
        layer = base_array._layers[0]
        assert isinstance(layer, Layer)
        assert isinstance(layer.transform, IndexTransform)

    def test_layers_immutable_after_slice(self, base_array: Array) -> None:
        """Slicing should not change layers, only the outer transform."""
        original_layers = base_array._layers
        sliced = base_array[10:20]
        assert sliced._layers is original_layers
