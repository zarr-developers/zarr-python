"""A stateful test of one array's life: create, append, resize, write, reopen.

The model is a NumPy array, not a second zarr array, so a bug in zarr's chunk
grid logic cannot hide by being made on both sides. Zero-length axes are drawn
on purpose, both at creation and by resizing and appending, and so are the
stored chunk sizes of 0 that zarr-python wrote for empty arrays before 3.4.

`resize` deletes only the chunks that fall entirely outside the new shape, so
cells cut off by a shrink can come back with their old values when the axis
grows again. The model does not encode that chunk-level behaviour: a cell cut
off and brought back is unknown until it is written.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Literal

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import event, note, settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

import zarr
from zarr.core.buffer import cpu, default_buffer_prototype
from zarr.core.sync import sync
from zarr.errors import ZarrUserWarning
from zarr.storage import MemoryStore

pytestmark = pytest.mark.filterwarnings(
    "ignore::zarr.core.dtype.common.UnstableSpecificationWarning"
)

DTYPE = np.dtype("int16")
MAX_SIDE = 6


def _rectilinear_dim(extent: int) -> st.SearchStrategy[int | list[int]]:
    """A bare step, or an edge list covering `extent` (any edges for extent 0)."""
    steps = st.integers(min_value=1, max_value=MAX_SIDE)
    if extent == 0:
        return steps | st.lists(steps, min_size=1, max_size=3)
    if extent == 1:
        return steps | st.just([1])
    cuts = st.lists(st.integers(min_value=1, max_value=extent - 1), unique=True, max_size=3)
    edges = cuts.map(
        lambda c: [b - a for a, b in zip([0, *sorted(c)], [*sorted(c), extent], strict=True)]
    )
    return steps | edges


class ArrayLifecycle(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self._rectilinear = zarr.config.set({"array.rectilinear_chunks": True})
        self._rectilinear.__enter__()
        self.store = MemoryStore()
        self.path = "a"
        self.model: np.ndarray[Any, np.dtype[np.int16]] = np.zeros((0,), dtype=DTYPE)
        # Cells whose value the model knows; see the module docstring.
        self.known: np.ndarray[Any, np.dtype[np.bool_]] = np.ones((0,), dtype=bool)
        # Every shape the array has had, to find cells a resize brings back.
        self.past_shapes: list[tuple[int, ...]] = []
        self.fill = 0
        # A legacy store warns until its metadata is re-saved.
        self.expect_open_warning = False

    # -------------------------------------------------------------- creation
    @initialize(data=st.data())
    def create(self, data: st.DataObject) -> None:
        zarr_format: Literal[2, 3] = data.draw(st.sampled_from([2, 3]), label="zarr_format")
        shape = data.draw(
            npst.array_shapes(min_dims=1, max_dims=3, min_side=0, max_side=MAX_SIDE),
            label="shape",
        )
        self.fill = data.draw(st.integers(-3, 3), label="fill_value")
        # sampled_from favours early entries; the less common spellings go first.
        spellings = ["ints", "legacy-zero", "-1", "False", "auto"]
        if zarr_format == 3:
            spellings.insert(1, "rectilinear")
        spelling = data.draw(st.sampled_from(spellings), label="chunk spelling")
        event(f"chunks: {spelling}")
        if any(s == 0 for s in shape):
            event("created with a zero-length axis")

        chunks: Any
        shards: Any = None
        if spelling == "-1":
            chunks = -1
        elif spelling == "False":
            chunks = False
        elif spelling == "auto":
            chunks = "auto"
        elif spelling == "rectilinear":
            chunks = [data.draw(_rectilinear_dim(s)) for s in shape]
            if not any(isinstance(c, list) for c in chunks):
                chunks[0] = [chunks[0]] if shape[0] == 0 else [shape[0]]
        else:
            chunks = tuple(data.draw(st.integers(1, 4)) for _ in shape)
            if (
                spelling == "ints"
                and zarr_format == 3
                and data.draw(st.booleans(), label="sharded")
            ):
                shards = tuple(c * data.draw(st.integers(1, 2)) for c in chunks)
                event("sharded")
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
        self.model = np.full(shape, self.fill, dtype=DTYPE)
        self.known = np.ones(shape, dtype=bool)
        self.past_shapes = [shape]

        if spelling == "legacy-zero":
            # What zarr-python wrote before 3.4 for an array created with a
            # zero-length axis and one chunk spanning it; older releases could
            # grow that axis without storing a chunk, so any extent is possible.
            zero_axes = data.draw(
                st.lists(st.integers(0, len(shape) - 1), min_size=1, unique=True),
                label="axes stored with chunk size 0",
            )
            stored_zero = data.draw(st.sampled_from([0, False]), label="stored zero")
            self._rewrite_stored_chunks(zarr_format, zero_axes, stored_zero)
            self.expect_open_warning = True
            if any(shape[i] > 0 for i in zero_axes):
                event("legacy zero chunk on a grown axis")

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
        warned = any(issubclass(w.category, ZarrUserWarning) for w in record)
        assert warned is self.expect_open_warning, [str(w.message) for w in record]
        return arr

    # ----------------------------------------------------------------- rules
    @rule(data=st.data())
    def append(self, data: st.DataObject) -> None:
        arr = self._open()
        axis = data.draw(st.integers(0, self.model.ndim - 1), label="axis")
        block_shape = list(self.model.shape)
        block_shape[axis] = data.draw(st.integers(0, 4), label="rows")
        block = data.draw(npst.arrays(DTYPE, tuple(block_shape)), label="block")
        note(f"append {block.shape} along {axis} to {self.model.shape}")
        if self.model.shape[axis] == 0:
            event("append to a zero-length axis")
        arr.append(block, axis=axis)
        self._reshape_model(arr.shape)
        tail = tuple(
            slice(-block.shape[axis], None) if i == axis and block.shape[axis] else slice(None)
            for i in range(self.model.ndim)
        )
        if block.shape[axis]:
            self.model[tail] = block
            self.known[tail] = True
        # Growing the array rewrites its metadata, which stores any correction.
        self.expect_open_warning = False

    @rule(data=st.data())
    def resize(self, data: st.DataObject) -> None:
        arr = self._open()
        new_shape = data.draw(
            st.tuples(*(st.integers(0, MAX_SIDE) for _ in self.model.shape)), label="new shape"
        )
        note(f"resize {self.model.shape} -> {new_shape}")
        if any(o == 0 and n > 0 for o, n in zip(self.model.shape, new_shape, strict=True)):
            event("resize grows a zero-length axis")
        arr.resize(new_shape)
        self._reshape_model(new_shape)
        self.expect_open_warning = False

    def _reshape_model(self, new_shape: tuple[int, ...]) -> None:
        """Resize the model: kept cells keep their values, new cells hold the fill
        value, and cells that an earlier shape held but the current one cut off
        become unknown."""
        overlap = tuple(
            slice(0, min(o, n)) for o, n in zip(self.model.shape, new_shape, strict=True)
        )
        model = np.full(new_shape, self.fill, dtype=DTYPE)
        known = np.ones(new_shape, dtype=bool)
        for past in self.past_shapes:
            known[tuple(slice(0, min(p, n)) for p, n in zip(past, new_shape, strict=True))] = False
        model[overlap] = self.model[overlap]
        known[overlap] = self.known[overlap]
        if not known.all():
            event("resize brings back cells cut off earlier")
        self.model, self.known = model, known
        self.past_shapes.append(tuple(new_shape))

    @rule(data=st.data())
    def write(self, data: st.DataObject) -> None:
        arr = self._open()
        region = tuple(
            slice(*sorted(data.draw(st.tuples(st.integers(0, s), st.integers(0, s)))))
            for s in self.model.shape
        )
        values = data.draw(npst.arrays(DTYPE, self.model[region].shape), label="values")
        note(f"write {region}")
        arr[region] = values
        self.model[region] = values
        self.known[region] = True

    @rule()
    def resave_metadata(self) -> None:
        """What the legacy warning tells users to do."""
        self._open().update_attributes({})
        self.expect_open_warning = False

    def teardown(self) -> None:
        self._rectilinear.__exit__(None, None, None)

    # ------------------------------------------------------------ invariants
    @invariant()
    def matches_model(self) -> None:
        arr = self._open()
        assert arr.shape == self.model.shape
        actual = np.asarray(arr[...])
        np.testing.assert_array_equal(actual[self.known], self.model[self.known])


ArrayLifecycle.TestCase.settings = settings(max_examples=200, stateful_step_count=12, deadline=None)
TestArrayLifecycle = ArrayLifecycle.TestCase
