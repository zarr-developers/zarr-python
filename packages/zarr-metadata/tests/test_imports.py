"""
Smoke test: the package imports and its top-level public surface is reachable.
"""

from __future__ import annotations


def test_package_imports() -> None:
    """The package and its top-level union types load without errors."""
    import zarr_metadata

    # Touch the cross-version unions to confirm both v2 and v3 submodules
    # load and the top-level __init__ wires the union types correctly.
    assert zarr_metadata.ArrayMetadata is not None
    assert zarr_metadata.GroupMetadata is not None
