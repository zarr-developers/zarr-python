"""Property tests for composition-rule boundaries and API agreement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zarr_metadata.builder import create_zarr_v3_array_metadata_json
from zarr_metadata.model import MetadataValidationError
from zarr_metadata.rules import (
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_group_metadata_v2,
    validate_group_metadata_v3,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from zarr_metadata.model import ValidationProblem


JSON_SCALARS = st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text()
JSON_VALUES = st.recursive(
    JSON_SCALARS,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=12), children, max_size=4)
    ),
    max_leaves=20,
)


DOCUMENT_VALIDATORS = (
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_group_metadata_v2,
    validate_group_metadata_v3,
)


@pytest.mark.parametrize("validator", DOCUMENT_VALIDATORS, ids=lambda validator: validator.__name__)
@given(JSON_VALUES)
def test_document_validators_are_total_for_arbitrary_json(
    validator: Callable[[object], tuple[ValidationProblem, ...]], value: object
) -> None:
    """Untrusted JSON always produces a verdict; it never crashes the validator."""
    assert isinstance(validator(value), tuple)


@given(
    extent=st.integers(min_value=0, max_value=200),
    chunk_shapes=st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=8),
)
def test_rectilinear_explicit_chunks_may_overflow_extent(
    extent: int, chunk_shapes: list[int]
) -> None:
    """The final explicit chunk may extend past the array boundary."""
    if sum(chunk_shapes) < extent:
        return
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (extent,),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": (tuple(chunk_shapes),)},
        },
        "chunk_key_encoding": "default",
        "codecs": ("bytes",),
    }
    assert validate_array_metadata_v3(document) == ()


@given(depth=st.integers(min_value=1, max_value=5), leaf_fill=st.integers(0, 255))
def test_nested_struct_fill_values_are_checked_recursively(depth: int, leaf_fill: int) -> None:
    data_type: object = "uint8"
    fill_value: object = leaf_fill
    path: list[str] = []
    for level in range(depth):
        name = f"level_{level}"
        data_type = {
            "name": "struct",
            "configuration": {"fields": ({"name": name, "data_type": data_type},)},
        }
        fill_value = {name: fill_value}
        path.insert(0, name)

    document: dict[str, object] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (1,),
        "data_type": data_type,
        "fill_value": fill_value,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1,)}},
        "chunk_key_encoding": "default",
        "codecs": ("bytes",),
    }
    assert validate_array_metadata_v3(document) == ()

    invalid: object = 256
    for name in reversed(path):
        invalid = {name: invalid}
    document["fill_value"] = invalid
    problems = validate_array_metadata_v3(document)
    assert any(problem.loc == ("fill_value", *path) for problem in problems)


@given(
    exponents=st.lists(st.integers(min_value=0, max_value=6), min_size=1, max_size=4, unique=True)
)
def test_nested_sharding_pipelines_accept_divisible_inner_chunks(exponents: list[int]) -> None:
    """Every generated nesting level is checked against the level enclosing it."""
    inner_shapes = [2**exponent for exponent in sorted(exponents, reverse=True)]
    codecs: tuple[object, ...] = ("bytes",)
    for inner_shape in reversed(inner_shapes):
        codecs = (
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": (inner_shape,),
                    "codecs": codecs,
                    "index_codecs": (
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        "crc32c",
                    ),
                },
            },
        )
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (64,),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (64,)}},
        "chunk_key_encoding": "default",
        "codecs": codecs,
    }
    assert validate_array_metadata_v3(document) == ()


@given(
    data_type=st.sampled_from(("int8", "uint8", "int16", "uint16", "int32", "uint32")),
    fill_value=st.integers(min_value=-(2**40), max_value=2**40),
)
def test_validator_and_factory_agree(data_type: str, fill_value: int) -> None:
    document: Mapping[str, object] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4,),
        "data_type": data_type,
        "fill_value": fill_value,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2,)}},
        "chunk_key_encoding": "default",
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    }
    validator_accepts = validate_array_metadata_v3(document) == ()

    try:
        create_zarr_v3_array_metadata_json(**document)  # type: ignore[arg-type]
    except MetadataValidationError:
        factory_accepts = False
    else:
        factory_accepts = True

    assert validator_accepts == factory_accepts
