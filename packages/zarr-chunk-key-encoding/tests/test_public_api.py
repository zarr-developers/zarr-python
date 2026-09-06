"""Tests for the shape of the public API."""

from __future__ import annotations

import pathlib

import zarr_chunk_key_encoding

EXPECTED_PUBLIC_API = {
    "BoundedChunkKeyEncoding",
    "ChunkKeyConfigurationError",
    "ChunkKeyDecodeError",
    "ChunkKeyEncoding",
    "ChunkKeyEncodingError",
    "ChunkKeyEncodingJSON",
    "DefaultChunkKeyEncoding",
    "InvalidChunkCoordsError",
    "Separator",
    "UnknownChunkKeyEncodingError",
    "V2ChunkKeyEncoding",
    "__version__",
    "chunk_key_encoding_from_json",
}


def test_public_api_is_minimal() -> None:
    """The unreleased package exposes only the behavior its core consumers need."""
    assert set(zarr_chunk_key_encoding.__all__) == EXPECTED_PUBLIC_API


def test_every_module_is_private() -> None:
    """`__all__` in the top-level package is the entire public API.

    Every module beneath it is named with a leading underscore, so import
    paths like `zarr_chunk_key_encoding.abc` are not part of the contract and
    can be reorganized freely. A new module added without the underscore
    would silently widen that contract, so it is checked rather than trusted.
    """
    package_dir = pathlib.Path(zarr_chunk_key_encoding.__file__).parent
    public = sorted(
        p.name
        for p in package_dir.glob("*.py")
        if p.stem != "__init__" and not p.stem.startswith("_")
    )
    assert public == []


def test_all_names_resolve() -> None:
    """Every name in `__all__` is an attribute of the package.

    Ordering of `__all__` is owned by ruff (RUF022), so it is not asserted
    here.
    """
    for name in zarr_chunk_key_encoding.__all__:
        assert hasattr(zarr_chunk_key_encoding, name)


def test_no_plugin_surface() -> None:
    """The package exposes no registration or discovery API.

    Chunk key encoding is an extension point, but that machinery is common
    to every v3 extension point and belongs in a shared package. Adding it
    here later is compatible; removing it after release would not be, so
    this pins the absence.
    """
    for name in zarr_chunk_key_encoding.__all__:
        assert "register" not in name
        assert "ENTRY_POINT" not in name
        assert "Support" not in name
