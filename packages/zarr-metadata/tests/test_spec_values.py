"""Test the spec's closed value sets and defaults, as the narrowing helpers and default constants spell them."""

from collections.abc import Callable

import pytest

from zarr_metadata import (
    DEFAULT_CHUNK_KEY_ENCODING_DEFAULT_SEPARATOR,
    DEFAULT_CHUNK_KEY_ENCODING_NAME,
    DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR,
    UNSET,
    V2_CHUNK_KEY_ENCODING_DEFAULT_SEPARATOR,
    V2_CHUNK_KEY_ENCODING_NAME,
    V2_CHUNK_KEY_ENCODING_SEPARATOR,
    ZARR_V2_ARRAY_DEFAULT_DIMENSION_SEPARATOR,
    ZARR_V2_ARRAY_DIMENSION_SEPARATOR,
    ZarrV2ArrayMetadata,
)
from zarr_metadata.definition import canonical_of, resolve
from zarr_metadata.v2.array import is_zarr_v2_array_dimension_separator, is_zarr_v2_array_order
from zarr_metadata.v3 import is_zarr_v3_node_type
from zarr_metadata.v3.definition import CORE_CONTEXT, ChunkKeyEncodingDefinition


@pytest.mark.parametrize(
    ("helper", "value", "expected"),
    [
        (is_zarr_v2_array_order, "C", True),
        (is_zarr_v2_array_order, "F", True),
        (is_zarr_v2_array_order, "c", False),
        (is_zarr_v2_array_order, "A", False),
        (is_zarr_v2_array_order, None, False),
        (is_zarr_v2_array_dimension_separator, ".", True),
        (is_zarr_v2_array_dimension_separator, "/", True),
        (is_zarr_v2_array_dimension_separator, "\\", False),
        (is_zarr_v2_array_dimension_separator, None, False),
        (is_zarr_v3_node_type, "array", True),
        (is_zarr_v3_node_type, "group", True),
        (is_zarr_v3_node_type, "Array", False),
        (is_zarr_v3_node_type, ("array",), False),
        (is_zarr_v3_node_type, 3, False),
    ],
)
def test_narrowing_helpers(helper: Callable[[object], bool], value: object, expected: bool) -> None:
    """Each helper accepts exactly the members of its value set."""
    assert helper(value) is expected


def test_v2_array_default_dimension_separator() -> None:
    """An array document without a `dimension_separator` means the default one, which is a permitted separator."""
    assert ZARR_V2_ARRAY_DEFAULT_DIMENSION_SEPARATOR in ZARR_V2_ARRAY_DIMENSION_SEPARATOR
    metadata = ZarrV2ArrayMetadata.create_default(dimension_separator=UNSET)
    assert "dimension_separator" not in metadata.to_json()
    assert metadata.dimension_separator == ZARR_V2_ARRAY_DEFAULT_DIMENSION_SEPARATOR


@pytest.mark.parametrize(
    ("name", "default", "permitted"),
    [
        (
            DEFAULT_CHUNK_KEY_ENCODING_NAME,
            DEFAULT_CHUNK_KEY_ENCODING_DEFAULT_SEPARATOR,
            DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR,
        ),
        (
            V2_CHUNK_KEY_ENCODING_NAME,
            V2_CHUNK_KEY_ENCODING_DEFAULT_SEPARATOR,
            V2_CHUNK_KEY_ENCODING_SEPARATOR,
        ),
    ],
)
def test_chunk_key_encoding_default_separator(
    name: str, default: str, permitted: tuple[str, ...]
) -> None:
    """A chunk key encoding's default separator is permitted, and its simplest spelling omits it."""
    assert default in permitted
    declaration = {"name": name, "configuration": {"separator": default}}
    resolved, problems = resolve(declaration, ChunkKeyEncodingDefinition, CORE_CONTEXT)
    assert canonical_of(resolved, problems) == {"name": name}
