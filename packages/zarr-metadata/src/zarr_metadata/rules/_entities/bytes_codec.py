"""Composition rules for the core ``bytes`` codec."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._engine import as_string_mapping
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.v3._extension_points import CODECS, DATA_TYPE
from zarr_metadata.v3._shape import blocking_problems, validate_known_entity_metadata
from zarr_metadata.v3.codec.bytes import BYTES_CODEC_NAME
from zarr_metadata.v3.data_type.raw import RAW_BYTES_NAME_PATTERN

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.rules._spec import ArraySpec

_ARRAY_V3 = "zarr_v3_array"

_SINGLE_BYTE = frozenset({"bool", "int8", "uint8"})
_MULTI_BYTE = frozenset(
    {
        "int16",
        "int32",
        "int64",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "numpy.datetime64",
        "numpy.timedelta64",
    }
)
_VARIABLE_LENGTH = frozenset({"bytes", "string"})
_StorageClass = Literal["single_byte", "multi_byte", "variable_length"]


def _data_type_name(data_type: object) -> str | None:
    if isinstance(data_type, str):
        return data_type
    mapping = as_string_mapping(data_type)
    if mapping is None:
        return None
    name = mapping.get("name")
    return name if isinstance(name, str) else None


def _storage_class(data_type: object) -> _StorageClass | None:
    """Classify known data types by their raw byte representation."""
    name = _data_type_name(data_type)
    if name in _SINGLE_BYTE or (name is not None and RAW_BYTES_NAME_PATTERN.fullmatch(name)):
        return "single_byte"
    if name in _MULTI_BYTE:
        return "multi_byte"
    if name in _VARIABLE_LENGTH:
        return "variable_length"
    if name != "struct":
        return None

    envelope = as_string_mapping(data_type)
    configuration = (
        as_string_mapping(envelope.get("configuration")) if envelope is not None else None
    )
    fields = configuration.get("fields") if configuration is not None else None
    if not isinstance(fields, tuple):
        return None
    classes: list[_StorageClass] = []
    for field in cast("tuple[object, ...]", fields):
        field_mapping = as_string_mapping(field)
        if field_mapping is None or "data_type" not in field_mapping:
            return None
        field_class = _storage_class(field_mapping["data_type"])
        if field_class is None:
            return None
        classes.append(field_class)
    if "variable_length" in classes:
        return "variable_length"
    if "multi_byte" in classes:
        return "multi_byte"
    return "single_byte"


@entity_rule(_ARRAY_V3, CODECS, BYTES_CODEC_NAME)
def data_type_has_a_raw_byte_representation(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    if incoming.data_type is None:
        return ()
    shape_verdict = validate_known_entity_metadata(DATA_TYPE, incoming.data_type)
    if shape_verdict is not None and len(blocking_problems(shape_verdict)) != 0:
        return ()
    storage_class = _storage_class(incoming.data_type)
    if storage_class == "variable_length":
        name = _data_type_name(incoming.data_type)
        return (
            ValidationProblem(
                (),
                f"bytes codec is not compatible with variable-length data_type {name!r}",
                "invalid_value",
            ),
        )
    if storage_class == "multi_byte" and "endian" not in configuration:
        return (
            ValidationProblem(
                ("endian",),
                "endian is required for a data type containing multi-byte values",
                "missing_key",
            ),
        )
    return ()
