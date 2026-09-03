"""Drift tests tying the hand-written judgment registries to the package's
type modules: adding a codec, chunk grid, or data type module without
registering it in the corresponding judgment surface must fail a test
rather than silently weaken validation (an unregistered codec, for
example, would suppress the exactly-one-array->bytes check for every
pipeline containing it)."""

from __future__ import annotations

import importlib
import pkgutil

import zarr_metadata.v3.chunk_grid
import zarr_metadata.v3.chunk_key_encoding
import zarr_metadata.v3.codec
import zarr_metadata.v3.data_type
from zarr_metadata.rules._v3_array import (
    _check_fill_for_dtype,  # pyright: ignore[reportPrivateUsage]
)
from zarr_metadata.v3._extension_points import RAW_BYTES_FAMILY
from zarr_metadata.v3._shape import (  # pyright: ignore[reportPrivateUsage]
    _CHUNK_GRID_SHAPES,
    _CHUNK_KEY_ENCODING_SHAPES,
    _CODEC_SHAPES,
    _DATA_TYPE_SHAPES,
)
from zarr_metadata.v3.codec.kind import codec_kind_of_name


def _module_constants(package: object, suffix: str) -> set[str]:
    """Values of `*<suffix>` constants across a package's public modules."""
    names: set[str] = set()
    for info in pkgutil.iter_modules(package.__path__):  # type: ignore[attr-defined]
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package.__name__}.{info.name}")  # type: ignore[attr-defined]
        names.update(
            value
            for attribute, value in vars(module).items()
            if attribute.endswith(suffix) and isinstance(value, str)
        )
    return names


def test_every_codec_module_is_kind_classified() -> None:
    codec_names = _module_constants(zarr_metadata.v3.codec, "_CODEC_NAME")
    assert codec_names, "constant scan found nothing — the naming convention moved?"
    unclassified = {name for name in codec_names if codec_kind_of_name(name) is None}
    assert not unclassified


def test_every_codec_module_has_a_shape_validator() -> None:
    codec_names = _module_constants(zarr_metadata.v3.codec, "_CODEC_NAME")
    assert codec_names == set(_CODEC_SHAPES)


def test_every_chunk_grid_module_has_a_shape_validator() -> None:
    grid_names = _module_constants(zarr_metadata.v3.chunk_grid, "_CHUNK_GRID_NAME")
    assert grid_names, "constant scan found nothing — the naming convention moved?"
    assert grid_names == set(_CHUNK_GRID_SHAPES)


def test_every_chunk_key_encoding_module_has_a_shape_validator() -> None:
    names = _module_constants(zarr_metadata.v3.chunk_key_encoding, "_CHUNK_KEY_ENCODING_NAME")
    assert names, "constant scan found nothing — the naming convention moved?"
    assert names == set(_CHUNK_KEY_ENCODING_SHAPES)


def test_every_data_type_module_has_a_shape_validator() -> None:
    names = _module_constants(zarr_metadata.v3.data_type, "_DATA_TYPE_NAME")
    assert names, "constant scan found nothing — the naming convention moved?"
    assert names | {RAW_BYTES_FAMILY} == set(_DATA_TYPE_SHAPES)


def test_every_data_type_has_a_fill_value_branch() -> None:
    # object() is a valid fill value for no data type this package
    # defines, so a known name must produce a complaint; only genuinely
    # unknown names may decline (extension openness). The parameterized
    # r<N> family has no name constant and is represented by "r8".
    dtype_names = _module_constants(zarr_metadata.v3.data_type, "_DATA_TYPE_NAME")
    assert dtype_names, "constant scan found nothing — the naming convention moved?"
    unjudged = {
        name for name in {*dtype_names, "r8"} if _check_fill_for_dtype(name, object()) is None
    }
    assert not unjudged
