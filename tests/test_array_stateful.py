"""A stateful test of one array's life: create, append, resize, write, reopen.

The model is a NumPy array of what the store holds, not a second zarr array, so a bug
in zarr's chunk grid logic cannot hide by being made on both sides. Zero-length axes
are drawn on purpose, both at creation and by resizing and appending, and so are
stored chunk sizes of 0.

The model tracks cells beyond the array's shape too, because chunks do: a shrinking
`resize` deletes exactly the chunks outside the new grid, a chunk it keeps keeps its
cells beyond the new shape (and they come back if the array grows again), and a write
that covers every in-bounds cell of an unsharded chunk rewrites the whole chunk,
resetting its cells beyond the shape to the fill value. A sharded array rewrites a
shard through its inner chunks, each judged against the shard rather than the array
shape, so a write never resets cells beyond the shape.
"""

from __future__ import annotations

import itertools
import json
import warnings
from typing import Any, Literal

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import event, note
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

import zarr
from zarr.core.buffer import cpu, default_buffer_prototype
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.sync import sync
from zarr.errors import ZarrUserWarning
from zarr.storage import MemoryStore
from zarr.testing.strategies import rectilinear_chunks

pytestmark = [
    pytest.mark.slow_hypothesis,
    pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning"),
]

DTYPE = np.dtype("int16")
METADATA_KEYS = (".zarray", ".zattrs", "zarr.json")
MAX_SIDE = 6


async def _list(store: MemoryStore, prefix: str) -> list[str]:
    return [key async for key in store.list_prefix(prefix)]


class ArrayLifecycle(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self._rectilinear = zarr.config.set({"array.rectilinear_chunks": True})
        self._rectilinear.__enter__()
        self.store = MemoryStore()
        self.path = "a"
        self.shape: tuple[int, ...] = (0,)
        self.fill = 0
        # What the store holds, indexed like the array and extending past its shape.
        self.stored: np.ndarray[Any, np.dtype[np.int16]] = np.zeros((0,), dtype=DTYPE)
        # Axes stored with chunk size 0; the store warns until its metadata is re-saved.
        self.legacy_axes: list[int] = []

    # -------------------------------------------------------------- creation
    @initialize(data=st.data())
    def create(self, data: st.DataObject) -> None:
        zarr_format: Literal[2, 3] = data.draw(st.sampled_from([3, 2]), label="zarr_format")
        shape = data.draw(
            npst.array_shapes(min_dims=1, max_dims=3, min_side=0, max_side=MAX_SIDE),
            label="shape",
        )
        self.fill = data.draw(st.integers(-3, 3), label="fill_value")
        # sampled_from favours early entries; the less common spellings go first.
        spellings = ["ints", "-1", "False", "auto"]
        if zarr_format == 3:
            spellings[:0] = ["sharded", "rectilinear"]
        spelling = data.draw(st.sampled_from(spellings), label="chunk spelling")
        event(f"chunks: {spelling}")

        chunks: Any
        shards: Any = None
        if spelling in ("-1", "False"):
            chunks = {"-1": -1, "False": False}[spelling]
        elif spelling == "auto":
            chunks = "auto"
        elif spelling == "rectilinear":
            chunks = data.draw(rectilinear_chunks(shape=shape), label="rectilinear chunks")
        else:
            chunks = tuple(data.draw(st.integers(1, 3)) for _ in shape)
            if spelling == "sharded":
                shards = tuple(c * data.draw(st.integers(1, 3)) for c in chunks)
        note(f"create {shape=} {chunks=} {shards=} {zarr_format=} fill={self.fill}")
        zarr.create_array(
            self.store,
            name=self.path,
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype=DTYPE,
            fill_value=self.fill,
            zarr_format=zarr_format,
        )
        self.shape = shape
        self.stored = np.full(shape, self.fill, dtype=DTYPE)

        if spelling != "rectilinear" and data.draw(st.booleans(), label="legacy zero"):
            # A stored chunk size of 0, as zarr-python wrote for arrays created with a
            # zero-length axis; older releases could then grow the axis without storing
            # a chunk, so any extent is possible, but zero-length axes come first.
            # Sharded arrays store it in the outer grid.
            axes = st.lists(st.integers(0, len(shape) - 1), min_size=1, unique=True)
            if zero_axes := [axis for axis, extent in enumerate(shape) if extent == 0]:
                axes = st.lists(st.sampled_from(zero_axes), min_size=1, unique=True) | axes
            self.legacy_axes = data.draw(axes, label="axes stored with chunk size 0")
            stored_zero = data.draw(st.sampled_from([0, False]), label="stored zero")
            self._rewrite_stored_chunks(zarr_format, self.legacy_axes, stored_zero)
            event("legacy zero chunk size")
        else:
            # Re-saving valid metadata, as the warning tells users to, changes nothing.
            arr = self._open()
            arr.update_attributes({})
            assert self._open().metadata == arr.metadata

    def _rewrite_stored_chunks(
        self, zarr_format: Literal[2, 3], axes: list[int], value: Any
    ) -> None:
        key = f"{self.path}/{'.zarray' if zarr_format == 2 else 'zarr.json'}"
        buf = sync(self.store.get(key, prototype=default_buffer_prototype()))
        assert buf is not None
        doc = json.loads(buf.to_bytes())
        sizes = (
            doc["chunks"] if zarr_format == 2 else doc["chunk_grid"]["configuration"]["chunk_shape"]
        )
        for axis in axes:
            sizes[axis] = value
        sync(self.store.set(key, cpu.Buffer.from_bytes(json.dumps(doc).encode())))

    def _open(self) -> zarr.Array[Any]:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", ZarrUserWarning)
            arr = zarr.open_array(self.store, path=self.path, mode="r+")
        # A stored chunk size of 0 is read silently on an empty axis; on a non-empty one
        # it warns that the axis holds only the fill value. Either way it is upgraded.
        warned = any(issubclass(w.category, ZarrUserWarning) for w in record)
        must_warn = any(self.shape[axis] > 0 for axis in self.legacy_axes)
        assert warned is must_warn, [str(w.message) for w in record]
        assert (arr.metadata._stored_document is not None) is bool(self.legacy_axes)
        return arr

    # ----------------------------------------------------------------- model
    def _cover(self, shape: tuple[int, ...]) -> None:
        """Grow `stored` with the fill value so that it covers `shape`."""
        pad = [(0, max(0, n - s)) for n, s in zip(shape, self.stored.shape, strict=True)]
        if any(after for _, after in pad):
            self.stored = np.pad(self.stored, pad, constant_values=self.fill)

    def _model_resize(self, grid: ChunkGrid, new_shape: tuple[int, ...]) -> None:
        """Delete exactly the chunks of `grid` outside the grid for `new_shape`."""
        kept = []
        for dim, old, new in zip(grid.dimensions, self.shape, new_shape, strict=True):
            if new >= old:
                kept.append(slice(None))
            elif new == 0:
                kept.append(slice(0, 0))
            else:
                last = dim.index_to_chunk(new - 1)
                kept.append(slice(0, dim.chunk_offset(last) + dim.chunk_size(last)))
        stored = np.full_like(self.stored, self.fill)
        stored[tuple(kept)] = self.stored[tuple(kept)]
        self.stored = stored
        self.shape = new_shape
        self._cover(new_shape)

    def _model_write(self, arr: zarr.Array[Any], region: tuple[slice, ...], values: Any) -> None:
        """Write `values`; an unsharded chunk whose in-bounds cells are all written is
        rewritten whole."""
        grid = ChunkGrid.from_metadata(arr.metadata)
        if arr.shards is None and all(r.stop > r.start for r in region):
            # Per axis, each chunk the region touches: (start, stop, all in-bounds cells written).
            per_axis: list[list[tuple[int, int, bool]]] = []
            for dim, r, extent in zip(grid.dimensions, region, self.shape, strict=True):
                spans = []
                for c in range(dim.index_to_chunk(r.start), dim.index_to_chunk(r.stop - 1) + 1):
                    lo = dim.chunk_offset(c)
                    hi = lo + dim.chunk_size(c)
                    spans.append((lo, hi, r.start <= lo and r.stop >= min(hi, extent)))
                per_axis.append(spans)
            self._cover(tuple(max(hi for _, hi, _ in axis) for axis in per_axis))
            for combo in itertools.product(*per_axis):
                if all(complete for *_, complete in combo):
                    self.stored[tuple(slice(lo, hi) for lo, hi, _ in combo)] = self.fill
        self.stored[region] = values

    # ----------------------------------------------------------------- rules
    @rule(data=st.data())
    def append(self, data: st.DataObject) -> None:
        arr = self._open()
        axes = st.integers(0, len(self.shape) - 1)
        if self.legacy_axes:
            # What a user of an older release did next: grow an axis stored with chunk size 0.
            axes = st.sampled_from(self.legacy_axes) | axes
        axis = data.draw(axes, label="axis")
        block_shape = list(self.shape)
        block_shape[axis] = data.draw(st.integers(0, 4), label="rows")
        block = data.draw(npst.arrays(DTYPE, tuple(block_shape)), label="block")
        note(f"append {block.shape} along {axis} to {self.shape}")
        if self.shape[axis] == 0 and block.shape[axis]:
            event("append to a zero-length axis")
        if axis in self.legacy_axes and block.shape[axis]:
            event("grow an axis stored with chunk size 0")
        old_extent = self.shape[axis]
        arr.append(block, axis=axis)
        self._model_resize(ChunkGrid.from_metadata(arr.metadata), arr.shape)
        region = tuple(
            slice(old_extent, s) if i == axis else slice(0, s) for i, s in enumerate(arr.shape)
        )
        self._model_write(arr, region, block)
        # Growing the array rewrites its metadata, which stores any correction.
        self.legacy_axes = []

    @rule(data=st.data())
    def resize(self, data: st.DataObject) -> None:
        arr = self._open()
        extents = [st.integers(0, MAX_SIDE) for _ in self.shape]
        for axis in self.legacy_axes:
            # What a user of an older release did next: grow an axis stored with chunk size 0.
            extents[axis] = st.integers(self.shape[axis] + 1, MAX_SIDE + 1) | extents[axis]
        new_shape = data.draw(st.tuples(*extents), label="new shape")
        note(f"resize {self.shape} -> {new_shape}")
        if any(new_shape[axis] > self.shape[axis] for axis in self.legacy_axes):
            event("grow an axis stored with chunk size 0")
        grid = ChunkGrid.from_metadata(arr.metadata)
        arr.resize(new_shape)
        self._model_resize(grid, new_shape)
        self.legacy_axes = []

    @rule(data=st.data())
    def write(self, data: st.DataObject) -> None:
        arr = self._open()
        region = tuple(
            slice(*sorted(data.draw(st.tuples(st.integers(0, s), st.integers(0, s)))))
            for s in self.shape
        )
        shape = tuple(r.stop - r.start for r in region)
        values = data.draw(npst.arrays(DTYPE, shape), label="values")
        note(f"write {region}")
        arr[region] = values
        self._model_write(arr, region, values)
        if all(shape):
            # Writing chunks first stores the metadata they are written under; a write
            # of nothing stores nothing.
            self.legacy_axes = []

    @precondition(lambda self: self.legacy_axes)
    @rule()
    def resave_metadata(self) -> None:
        """What the warning for an invalid stored chunk size tells users to do: store
        the metadata as read."""
        arr = self._open()
        read = arr.metadata
        arr.update_attributes({})
        self.legacy_axes = []
        assert self._open().metadata == read

    def teardown(self) -> None:
        self._rectilinear.__exit__(None, None, None)

    # ------------------------------------------------------------ invariants
    @invariant()
    def no_chunk_under_an_invalid_document(self) -> None:
        """While the stored document is still one that is upgraded on read, which other
        readers may reject or read differently, no chunk is stored under it."""
        if self.legacy_axes:
            keys = sync(_list(self.store, f"{self.path}/"))
            assert set(keys) <= {f"{self.path}/{name}" for name in METADATA_KEYS}, keys

    @invariant()
    def matches_model(self) -> None:
        arr = self._open()
        assert arr.shape == self.shape
        expected = self.stored[tuple(slice(0, s) for s in self.shape)]
        np.testing.assert_array_equal(np.asarray(arr[...]), expected)


TestArrayLifecycle = ArrayLifecycle.TestCase
