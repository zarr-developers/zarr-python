from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.storage import LocalStore, MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from zarr.abc.store import Store


def _zarrs_available() -> bool:
    """Return True only if the zarrs CrudBackend is fully usable (registered)."""
    try:
        import _zarrs_bindings  # noqa: F401
    except ImportError:
        return False
    try:
        import zarr.zarrs
    except ImportError:
        return False
    # The module might exist but not yet register the zarrs CrudBackend (e.g.
    # Task 4 not yet merged). Verify registration before enabling the param.
    try:
        import zarr.crud

        zarr.crud.get_backend("zarrs")
    except (ImportError, KeyError):
        return False
    return True


@pytest.fixture(
    params=[
        "reference",
        pytest.param(
            "zarrs",
            marks=pytest.mark.skipif(
                not _zarrs_available(), reason="zarrs-bindings is not installed"
            ),
        ),
    ]
)
def backend(request: pytest.FixtureRequest) -> str:
    """A CRUD backend name. The zarrs param is skipped when the extension is absent."""
    import zarr.crud

    if request.param == "zarrs":
        import zarr.zarrs  # noqa: F401  (registers the zarrs backend)
    return str(request.param)


@pytest.fixture(params=["memory", "local"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Store]:
    if request.param == "memory":
        s: Store = await MemoryStore.open()
    else:
        s = await LocalStore.open(root=tmp_path / "store")
    try:
        yield s
    finally:
        s.close()


def array_metadata(**kwargs: Any) -> dict[str, Any]:
    """An array metadata document built via zarr-python itself."""
    params: dict[str, Any] = {
        "shape": (8, 8),
        "chunks": (4, 4),
        "dtype": "uint16",
        "zarr_format": 3,
    } | kwargs
    arr = zarr.create_array(store=MemoryStore(), **params)
    doc = dict(arr.metadata.to_dict())
    if params["zarr_format"] == 2:
        doc.pop("attributes", None)
    return doc


def filled(store: Store, **kwargs: Any) -> tuple[np.ndarray[Any, np.dtype[Any]], dict[str, Any]]:
    """Create an 8x8 array 'a', fill it with a ramp, return (data, metadata)."""
    params: dict[str, Any] = {"shape": (8, 8), "chunks": (4, 4), "dtype": "uint16"} | kwargs
    arr = zarr.create_array(store=store, name="a", **params)
    data = np.arange(64, dtype=params["dtype"]).reshape(8, 8)
    arr[:, :] = data
    doc = dict(arr.metadata.to_dict())
    if params.get("zarr_format") == 2:
        doc.pop("attributes", None)
    return data, doc
