"""Tests for how a name reaches the entity that answers for it."""

from __future__ import annotations

import pytest

from zarr_metadata.rules import validate_array_metadata_v3
from zarr_metadata.v3._extension_points import (
    CODECS,
    DATA_TYPE,
    RAW_BYTES_FAMILY,
)
from zarr_metadata.v3.codec.blosc import BloscCodec
from zarr_metadata.v3.data_type.numpy_datetime64 import NumpyDatetime64DataType
from zarr_metadata.v3.data_type.raw import RawBytesDataType
from zarr_metadata.v3.data_type.uint8 import Uint8DataType
from zarr_metadata.v3.entity import CORE_AND_EXTENSIONS, MetadataEntity

# (field, name, the entity that answers for it — None when nothing does)
RESOLUTIONS: dict[str, tuple[str, str, type[MetadataEntity] | None]] = {
    "plain-dtype": (DATA_TYPE, "uint8", Uint8DataType),
    "dotted-dtype": (DATA_TYPE, "numpy.datetime64", NumpyDatetime64DataType),
    "raw-8": (DATA_TYPE, "r8", RawBytesDataType),
    "raw-24": (DATA_TYPE, "r24", RawBytesDataType),
    # A malformed member reaches the family too: a misspelling of
    # something we model must be reported as such, not pass as an unknown
    # third-party extension.
    "raw-not-multiple-of-8": (DATA_TYPE, "r12", RawBytesDataType),
    "raw-zero": (DATA_TYPE, "r0", RawBytesDataType),
    # Tables are per point, so the family cannot be reached from another.
    "raw-shaped-codec-name": (CODECS, "r8", None),
    "codec": (CODECS, "blosc", BloscCodec),
    "unknown": (CODECS, "zfpy", None),
    # The family's key is invented, so no document may write it.
    "the-family-key-itself": (DATA_TYPE, RAW_BYTES_FAMILY, None),
}


@pytest.mark.parametrize(("field", "name", "expected"), RESOLUTIONS.values(), ids=list(RESOLUTIONS))
def test_a_name_resolves_to_the_entity_that_answers_for_it(
    field: str, name: str, expected: type[MetadataEntity] | None
) -> None:
    assert CORE_AND_EXTENSIONS.resolve(field, name) is expected  # type: ignore[arg-type]


def test_squatted_names_are_judged_against_the_definition_they_squat() -> None:
    # Zarr identifiers are registry-allocated. A private codec named
    # `bytes` has left the compatibility contract, and saying so is the
    # correct answer rather than a limitation, so nothing here defends
    # against collisions.
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4, 4),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
        "chunk_key_encoding": "default",
        "codecs": ({"name": "bytes", "configuration": {"width": 7}},),
    }
    problems = validate_array_metadata_v3(document)
    assert [(p.loc, p.kind) for p in problems] == [
        (("codecs", 0, "configuration", "width"), "unknown_key")
    ]


def test_forging_the_family_sentinel_cannot_change_a_verdict() -> None:
    # A literal "r<N>" data type mislabels nothing: the family claims its
    # names through `accepts`, not through the table key, so no validation
    # verdict depends on the sentinel being unforgeable.
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4, 4),
        "data_type": RAW_BYTES_FAMILY,
        "fill_value": (1,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
        "chunk_key_encoding": "default",
        "codecs": ("bytes",),
    }
    # Unjudged as an unknown data type, exactly as any unmodelled name is.
    assert validate_array_metadata_v3(document) == ()
