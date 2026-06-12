from __future__ import annotations

import pytest

pytest.importorskip(
    "_zarrs_bindings", reason="zarrs-bindings is not installed", exc_type=ImportError
)


def test_import() -> None:
    import zarr.zarrs

    assert isinstance(zarr.zarrs.__version__, str)
