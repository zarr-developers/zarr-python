"""This package's own use of the state machine it exports.

`ChainedIndexingStateMachine` composes indexing steps onto a `LazyArray` and
checks each step against NumPy — see
`zarr_indexing.testing.stateful` for what the invariants assert and why.

Two sources: a NumPy array, which is the machine's default and reads at
`VECTORIZED`, and a real zarr array, whose partitioning is discovered from the
store rather than declared. The zarr case runs a smaller budget: it reads
through a store, and it exercises the same code paths.

This replaces a seeded `_random_chain` sweep in `test_lazy_array` that read
chained selections through `parts()`. That sweep did reach the states it was
meant to, but a rank-0 correlated view was absorbed by a reshape in `result()`
and mirrored into the sweep rather than read as a failure; asserting the
documented assembly literally makes that impossible to paper over.
`test_lazy_array` keeps its `result()`-based sweep, which is the deterministic
cross-flavor coverage this does not attempt.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
from hypothesis import settings

from zarr_indexing.support import IndexingSupport
from zarr_indexing.testing import (
    DEFAULT_SETTINGS,
    ChainedIndexingStateMachine,
    state_machine_test,
)

TestNumpyIndexing = state_machine_test(ChainedIndexingStateMachine)


class OneDimensionalIndexing(ChainedIndexingStateMachine):
    """The same chains over a rank-1 source.

    The sorted one-dimensional fancy path in `chunk_resolution` is entered only
    when both the input and output ranks are 1, and the output rank is the
    *source's* — so the rank-3 default walls that path off from the machine
    entirely, and from every downstream project told to subclass it. That path
    is where reordering and duplicate coordinates are partitioned, which is the
    corruption class this whole harness exists to catch.
    """

    data = np.arange(30, dtype=np.int64)
    partitionings: ClassVar[tuple[Any, ...]] = (None, (4,), (30,), ((7, 8, 15),))


TestOneDimensionalIndexing = state_machine_test(OneDimensionalIndexing)


class SingletonAxisIndexing(ChainedIndexingStateMachine):
    """A source with an extent-1 axis, which the code must not read as a broadcast one.

    An index array's axis is a singleton either because the map broadcasts over
    it or because the domain is genuinely one cell wide there, and the two are
    told apart by the domain rather than the array. Nothing generated the second
    kind.
    """

    data = np.arange(2 * 1 * 3, dtype=np.int64).reshape(2, 1, 3)
    partitionings: ClassVar[tuple[Any, ...]] = (None, (1, 1, 1), (2, 1, 2))


TestSingletonAxisIndexing = state_machine_test(SingletonAxisIndexing)


class ZarrIndexing(ChainedIndexingStateMachine):
    """The same chains against a zarr array, read at every level it serves.

    A zarr array carries `oindex` and `vindex`, so `VECTORIZED` is what
    `LazyArray` detects; naming the levels explicitly also draws the two outer
    ones, which route an orthogonal request to `oindex` one array axis at a time.
    """

    supports = (
        IndexingSupport.BASIC,
        IndexingSupport.OUTER_1VECTOR,
        IndexingSupport.OUTER,
        IndexingSupport.VECTORIZED,
    )

    def make_source(self, data: Any) -> Any:
        zarr = pytest.importorskip("zarr")
        array = zarr.create_array({}, shape=data.shape, chunks=(3, 2, 3), dtype=data.dtype)
        array[:] = data
        return array


TestZarrIndexing = state_machine_test(
    ZarrIndexing, config=settings(DEFAULT_SETTINGS, max_examples=50)
)
