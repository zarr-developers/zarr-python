"""Composition rules for the core ``bytes`` codec."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.rules._storage_class import data_type_name, storage_class
from zarr_metadata.v3._extension_points import CODECS, DATA_TYPE
from zarr_metadata.v3._shape import blocking_problems, validate_known_entity_metadata
from zarr_metadata.v3.codec.bytes import BYTES_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.rules._spec import ArraySpec

_ARRAY_V3 = "zarr_v3_array"


@entity_rule(_ARRAY_V3, CODECS, BYTES_CODEC_NAME, reads_optional=frozenset({"endian"}))
def data_type_has_a_raw_byte_representation(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    if incoming.data_type is None:
        return ()
    shape_verdict = validate_known_entity_metadata(DATA_TYPE, incoming.data_type)
    if shape_verdict is not None and len(blocking_problems(shape_verdict)) != 0:
        return ()
    found = storage_class(incoming.data_type)
    name = data_type_name(incoming.data_type)
    if found == "variable_length":
        return (
            ValidationProblem(
                (),
                f"bytes codec is not compatible with variable-length data_type {name!r}",
                "invalid_value",
            ),
        )
    if found == "multi_byte" and "endian" not in configuration:
        # Name the type: inside a shard's `index_codecs` the array is the
        # shard index, whose uint64 type appears nowhere in the document.
        return (
            ValidationProblem(
                ("endian",),
                f"endian is required for data type {name!r}, which contains multi-byte values",
                "missing_key",
            ),
        )
    return ()
