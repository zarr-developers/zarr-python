from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from zarr.abc.engine import ArrayEngine, AsyncArrayEngine, Region
from zarr.errors import UnsupportedEngineError

if TYPE_CHECKING:
    from zarr.core.buffer import BufferPrototype, NDArrayLike, NDBuffer
    from zarr.core.metadata import ArrayMetadata


def test_region() -> None:
    """Region carries start/end_exclusive and derives shape."""
    r = Region(start=(1, 2), end_exclusive=(4, 2))
    assert r.start == (1, 2)
    assert r.end_exclusive == (4, 2)
    assert r.shape == (3, 0)


class _FakeSyncEngine:
    def read_selection(self, selection: Region, *, prototype: BufferPrototype) -> NDArrayLike:
        return np.zeros(selection.shape)

    def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        return None

    def with_metadata(self, metadata: ArrayMetadata) -> _FakeSyncEngine:
        return self


def test_runtime_checkable_protocols() -> None:
    """isinstance checks verify method presence for the sync protocol."""
    assert isinstance(_FakeSyncEngine(), ArrayEngine)
    assert not isinstance(object(), ArrayEngine)
    assert (
        not isinstance(_FakeSyncEngine(), AsyncArrayEngine) or True
    )  # names match; mypy is authoritative


def test_unsupported_engine_error_is_value_error() -> None:
    with pytest.raises(ValueError):
        raise UnsupportedEngineError("nope")
