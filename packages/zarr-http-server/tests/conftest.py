from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.abc.store import Store

ZarrFormat = Literal[2, 3]


@pytest.fixture
def store(request: pytest.FixtureRequest) -> Store:
    """Store fixture resolved via indirect parametrization."""
    if request.param != "memory":
        raise ValueError(f"unsupported store param: {request.param!r}")
    return MemoryStore()


@pytest.fixture(params=(2, 3), ids=["zarr2", "zarr3"])
def zarr_format(request: pytest.FixtureRequest) -> ZarrFormat:
    """Zarr format version fixture, parametrized over v2 and v3."""
    if request.param == 2:
        return 2
    elif request.param == 3:
        return 3
    msg = f"Invalid zarr format requested. Got {request.param}, expected one of (2, 3)."
    raise ValueError(msg)
