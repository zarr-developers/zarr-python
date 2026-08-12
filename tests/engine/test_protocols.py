from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from zarr.abc.engine import ArrayEngine, AsyncArrayEngine, SelectionRequest
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.indexing import (
    BasicIndexer,
    BlockIndexer,
    CoordinateIndexer,
    MaskIndexer,
    OrthogonalIndexer,
)
from zarr.errors import UnsupportedEngineError

if TYPE_CHECKING:
    import numpy.typing as npt

    from zarr.abc.engine import SelectionKind
    from zarr.core.buffer import BufferPrototype, NDArrayLikeOrScalar, NDBuffer
    from zarr.core.indexing import Fields, Indexer
    from zarr.core.metadata import ArrayMetadata

SHAPE = (10, 9)
GRID = ChunkGrid.from_sizes(SHAPE, (3, 4))


def _request(kind: SelectionKind, selection: Any) -> SelectionRequest:
    return SelectionRequest(kind=kind, selection=selection, shape=SHAPE, chunk_grid=GRID)


@pytest.mark.parametrize(
    ("kind", "selection", "expected"),
    [
        ("basic", (slice(1, 5), slice(None)), BasicIndexer),
        ("orthogonal", (np.array([0, 3]), slice(None)), OrthogonalIndexer),
        ("coordinate", (np.array([0, 3]), np.array([1, 2])), CoordinateIndexer),
        ("mask", np.zeros(SHAPE, dtype=bool), MaskIndexer),
        ("block", (1, 2), BlockIndexer),
    ],
)
def test_request_builds_the_indexer_for_its_kind(
    kind: SelectionKind, selection: Any, expected: type[Indexer]
) -> None:
    """`kind` alone decides which indexing dialect the raw selection is read in."""
    assert isinstance(_request(kind, selection).indexer, expected)


def test_request_indexer_is_built_on_demand_and_cached() -> None:
    # An engine that never asks for the zarr-native view must not pay to build
    # it; one that asks twice must pay once.
    request = _request("basic", (slice(1, 5), slice(None)))
    assert "indexer" not in request.__dict__
    assert request.indexer is request.indexer


def test_request_keeps_the_selection_unresolved() -> None:
    # The reason the request carries the raw selection: `CoordinateIndexer`
    # stores its coordinates in chunk-sorted order, so an engine that
    # reconstructed the selection from the indexer would silently reorder the
    # points -- and with them the user's results.
    coords = (np.array([9, 0, 3]), np.array([8, 0, 2]))
    request = _request("coordinate", coords)
    assert request.selection is coords
    assert not np.array_equal(request.indexer.selection[0], coords[0])  # type: ignore[attr-defined]


class _FakeSyncEngine:
    def read_selection(
        self,
        request: SelectionRequest,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> NDArrayLikeOrScalar:
        return np.zeros(request.indexer.shape)

    def write_selection(
        self,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None:
        return None

    def with_metadata(self, metadata: ArrayMetadata) -> _FakeSyncEngine:
        return self


class _FakeAsyncEngine:
    async def read_selection(
        self,
        request: SelectionRequest,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> NDArrayLikeOrScalar:
        return np.zeros(request.indexer.shape)

    async def write_selection(
        self,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None:
        return None

    def with_metadata(self, metadata: ArrayMetadata) -> _FakeAsyncEngine:
        return self


def test_runtime_checkable_protocols() -> None:
    """isinstance checks verify method presence only; mypy is authoritative.

    Both fakes therefore satisfy both protocols -- the sync/async distinction
    is drawn at runtime by `classify_engine_arg`, not by isinstance.
    """
    assert isinstance(_FakeSyncEngine(), ArrayEngine)
    assert isinstance(_FakeAsyncEngine(), AsyncArrayEngine)
    assert not isinstance(object(), ArrayEngine)
    assert not isinstance(object(), AsyncArrayEngine)


def test_unsupported_engine_error_is_value_error() -> None:
    with pytest.raises(ValueError):
        raise UnsupportedEngineError("nope")
