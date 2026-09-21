"""How a name reaches the entity that answers for it.

`Context.resolve` tries the name as a key and, when nothing is keyed by
it, asks each entity at that point whether the name is one of its own --
which is how a family such as `r<N>` covers an unbounded set of names
from one registration. The examples pin the cases that matter; the
properties cover the family, which no example-based test can sample.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zarr_metadata.rules import validate_array_metadata_v3
from zarr_metadata.v3.codec.blosc import BloscCodec
from zarr_metadata.v3.data_type.numpy_datetime64 import NumpyDatetime64DataType
from zarr_metadata.v3.data_type.raw import RAW_BYTES_FAMILY, RawBytesDataType
from zarr_metadata.v3.data_type.uint8 import Uint8DataType
from zarr_metadata.v3.entity import (
    CORE_AND_EXTENSIONS,
    ChunkGridEntity,
    CodecEntity,
    DataTypeEntity,
    MetadataEntity,
)

# (field, name, the entity that answers for it — None when nothing does)
RESOLUTIONS: dict[str, tuple[type[MetadataEntity], str, type[MetadataEntity] | None]] = {
    "plain-dtype": (DataTypeEntity, "uint8", Uint8DataType),
    "dotted-dtype": (DataTypeEntity, "numpy.datetime64", NumpyDatetime64DataType),
    "raw-8": (DataTypeEntity, "r8", RawBytesDataType),
    "raw-24": (DataTypeEntity, "r24", RawBytesDataType),
    # A malformed member reaches the family too: a misspelling of
    # something we model must be reported as such, not pass as an unknown
    # third-party extension.
    "raw-not-multiple-of-8": (DataTypeEntity, "r12", RawBytesDataType),
    "raw-zero": (DataTypeEntity, "r0", RawBytesDataType),
    # Tables are per point, so the family cannot be reached from another.
    "raw-shaped-codec-name": (CodecEntity, "r8", None),
    "codec": (CodecEntity, "blosc", BloscCodec),
    "unknown": (CodecEntity, "zfpy", None),
    # The family's key is invented, so no document may write it.
    "the-family-key-itself": (DataTypeEntity, RAW_BYTES_FAMILY, None),
}


@pytest.mark.parametrize(("field", "name", "expected"), RESOLUTIONS.values(), ids=list(RESOLUTIONS))
def test_a_name_resolves_to_the_entity_that_answers_for_it(
    field: type[MetadataEntity], name: str, expected: type[MetadataEntity] | None
) -> None:
    assert CORE_AND_EXTENSIONS.resolve(field, name) is expected


@given(width=st.integers(min_value=0, max_value=2**32))
def test_every_numeric_r_spelling_resolves_to_the_family(width: int) -> None:
    # Including malformed widths (0, 12, anything not a multiple of 8):
    # the family claims a name by grammar shape, not by validity, so a
    # misspelled member of a family we model is reported as a misspelling
    # rather than passing as an unknown third-party extension.
    assert CORE_AND_EXTENSIONS.resolve(DataTypeEntity, f"r{width}") is RawBytesDataType


OTHER_KINDS: tuple[type[MetadataEntity], ...] = (CodecEntity, ChunkGridEntity)


@given(width=st.integers(min_value=0, max_value=2**32), field=st.sampled_from(OTHER_KINDS))
def test_r_shaped_names_resolve_to_nothing_outside_data_types(
    width: int, field: type[MetadataEntity]
) -> None:
    # The family belongs to `data_type`; a codec that happens to be named
    # `r8` must not reach it.
    assert CORE_AND_EXTENSIONS.resolve(field, f"r{width}") is None


# The scan `resolve` falls back to asks every entity, so a name no entity
# claims has to come back as nothing however many are registered.
_UNCLAIMED = st.text(min_size=1).filter(
    lambda name: (
        not (name.startswith("r") and name[1:].isdigit())
        and name not in CORE_AND_EXTENSIONS.tables[DataTypeEntity]
    )
)


@given(name=_UNCLAIMED)
def test_a_name_no_entity_claims_resolves_to_nothing(name: str) -> None:
    assert CORE_AND_EXTENSIONS.resolve(DataTypeEntity, name) is None


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
