from __future__ import annotations


def test_import() -> None:
    import zarr.zarrs

    assert isinstance(zarr.zarrs.__version__, str)
