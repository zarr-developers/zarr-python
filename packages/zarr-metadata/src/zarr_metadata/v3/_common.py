"""The v3 metadata field: its JSON type, the type aliases for members that hold a field, and the validators that check a single field.

This module is private. It sits below both code paths that read a field:
the model, which checks the fields of a document, and the definitions,
which read a field's configuration. Public consumers import
`ZarrV3MetadataFieldJSON` from `zarr_metadata.v3`, the aliases from
`zarr_metadata.v3.definition`, and the validators from
`zarr_metadata.model`.
"""

from typing import TypeAlias, TypeGuard, cast

from typing_extensions import TypeAliasType

from zarr_metadata._common import ZarrV3NamedConfigJSON
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    is_canonical_json,
    is_object,
    with_input,
)
from zarr_metadata.definition._envelope import (
    configuration_json_problems,
    envelope_problems,
    well_named,
)

ZarrV3MetadataFieldJSON: TypeAlias = str | ZarrV3NamedConfigJSON
"""The JSON shape of any v3 metadata extension-point entry: either a bare
short-hand name string or a `{name, configuration, must_understand}` envelope.

Used for `data_type`, `chunk_grid`, `chunk_key_encoding`, individual
codec entries, and `storage_transformers` in v3 array metadata, and for
the inner `codecs` / `index_codecs` lists of the `sharding_indexed`
codec.
"""


# A member that holds a metadata field is annotated with the alias for its
# kind. A scope uses that alias to decide which kind to read the field as.
# Each alias is defined as the field's JSON type, so a type checker and
# `check` both see it as `ZarrV3MetadataFieldJSON`.

ZarrV3DataTypeField = TypeAliasType("ZarrV3DataTypeField", ZarrV3MetadataFieldJSON)
"""A member that holds a data type: a document's `data_type`, or a struct field's data type. It is read in the same scope as the object that contains it."""
ZarrV3ChunkGridField = TypeAliasType("ZarrV3ChunkGridField", ZarrV3MetadataFieldJSON)
"""A member that holds a chunk grid: a document's `chunk_grid`."""
ZarrV3ChunkKeyEncodingField = TypeAliasType("ZarrV3ChunkKeyEncodingField", ZarrV3MetadataFieldJSON)
"""A member that holds a chunk key encoding: a document's `chunk_key_encoding`."""
ZarrV3CodecField = TypeAliasType("ZarrV3CodecField", ZarrV3MetadataFieldJSON)
"""A member that holds a codec. A document's `codecs` is `tuple[ZarrV3CodecField, ...]`, and so is a shard's."""
ZarrV3StaticCodecField = TypeAliasType("ZarrV3StaticCodecField", ZarrV3MetadataFieldJSON)
"""A member that holds a codec with a static encoded size. A shard's `index_codecs` uses
this, because a reader locates the index from a size it knows before reading it.
"""
ZarrV3StorageTransformerField = TypeAliasType(
    "ZarrV3StorageTransformerField", ZarrV3MetadataFieldJSON
)
"""A member that holds a storage transformer. A document's `storage_transformers` is `tuple[ZarrV3StorageTransformerField, ...]`."""


def validate_metadata_field_v3(
    value: object, *, allow_must_understand_false: bool = True
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a v3 metadata field.

    A metadata field is either a bare name or an envelope around a
    configuration whose members are JSON. The envelope is an object with a
    `name` that is a valid extension name under the spec, a `configuration`
    that is an object with string keys, a boolean `must_understand`, and no
    other keys.
    """
    envelope = envelope_problems(value, allow_must_understand_false=allow_must_understand_false)
    return with_input((*envelope, *configuration_json_problems(value)), value)


def is_metadata_field_v3(value: object) -> TypeGuard[ZarrV3MetadataFieldJSON]:
    """Whether `value` is a v3 metadata field: a bare name that is a valid extension name under the spec, or a named config."""
    if isinstance(value, str):
        return well_named(value)
    if not is_object(value) or not isinstance(value, dict):
        return False
    return is_canonical_json(value) and not validate_metadata_field_v3(value)


def parse_metadata_field_v3(value: object) -> ZarrV3MetadataFieldJSON:
    """Return `value` narrowed to `ZarrV3MetadataFieldJSON`, or raise `MetadataValidationError`."""
    problems = validate_metadata_field_v3(value)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast(ZarrV3MetadataFieldJSON, arrays_to_tuples(value))


__all__ = [
    "ZarrV3ChunkGridField",
    "ZarrV3ChunkKeyEncodingField",
    "ZarrV3CodecField",
    "ZarrV3DataTypeField",
    "ZarrV3MetadataFieldJSON",
    "ZarrV3StaticCodecField",
    "ZarrV3StorageTransformerField",
    "is_metadata_field_v3",
    "parse_metadata_field_v3",
    "validate_metadata_field_v3",
]
