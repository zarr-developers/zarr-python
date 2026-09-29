"""The v3 metadata field: its JSON, the aliases a member holding one is annotated with, and the validators that judge one on its own.

Private, and below both readers of a field: the model, which judges the
fields of a document, and the definitions, which read a field's
configuration. Public consumers import `ZarrV3MetadataFieldJSON` from
`zarr_metadata.v3`, the aliases from `zarr_metadata.v3.definition`, and
the validators from `zarr_metadata.model`.
"""

from collections.abc import Mapping
from typing import TypeGuard, cast

from typing_extensions import TypeAliasType

from zarr_metadata._common import ZarrV3NamedConfigJSON
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    is_canonical_json,
    prefixed,
    validate_json,
    with_input,
)

ZarrV3MetadataFieldJSON = str | ZarrV3NamedConfigJSON
"""The JSON shape of any v3 metadata extension-point entry: either a bare
short-hand name string or a `{name, configuration, must_understand}` envelope.

Used for `data_type`, `chunk_grid`, `chunk_key_encoding`, individual
codec entries, and `storage_transformers` in v3 array metadata, and for
the inner `codecs` / `index_codecs` lists of the `sharding_indexed`
codec.
"""


# A member holding a metadata field is annotated with the alias of its
# kind, which a scope reads it as. Each alias is the JSON a field is, so to
# a type checker, and to `check`, it is `ZarrV3MetadataFieldJSON`.

DataTypeField = TypeAliasType("DataTypeField", ZarrV3MetadataFieldJSON)
"""A member holding a data type: a document's `data_type`, or a struct field's; read in the scope what holds it is read in."""
ChunkGridField = TypeAliasType("ChunkGridField", ZarrV3MetadataFieldJSON)
"""A member holding a chunk grid: a document's `chunk_grid`."""
ChunkKeyEncodingField = TypeAliasType("ChunkKeyEncodingField", ZarrV3MetadataFieldJSON)
"""A member holding a chunk key encoding: a document's `chunk_key_encoding`."""
CodecField = TypeAliasType("CodecField", ZarrV3MetadataFieldJSON)
"""A member holding a codec: a document's `codecs` is `tuple[CodecField, ...]`, and so is a shard's."""
StaticCodecField = TypeAliasType("StaticCodecField", ZarrV3MetadataFieldJSON)
"""A member holding a codec of static size: a shard's `index_codecs` is one,
since a reader finds the index by a size it knows before reading it.
"""
StorageTransformerField = TypeAliasType("StorageTransformerField", ZarrV3MetadataFieldJSON)
"""A member holding a storage transformer: a document's `storage_transformers` is `tuple[StorageTransformerField, ...]`."""


def validate_metadata_field_v3(
    value: object, *, allow_must_understand_false: bool = True
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a v3 metadata field.

    A metadata field is a bare name, or an envelope around a configuration
    whose members are JSON: an object of a string `name`, a `configuration`
    that is an object of string keys, a boolean `must_understand`, and
    nothing else.
    """
    envelope = envelope_problems(value, allow_must_understand_false=allow_must_understand_false)
    return with_input((*envelope, *_configuration_json_problems(value)), value)


ENVELOPE_KEYS: frozenset[str] = frozenset({"name", "configuration", "must_understand"})
"""The members a metadata field's envelope declares: a key beside them is an unknown key."""


def envelope_problems(
    value: object, *, allow_must_understand_false: bool
) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a v3 metadata field's envelope, what its configuration holds left unjudged.

    The envelope is what sits around the configuration: a string `name`, a
    `configuration` that is an object of string keys, a boolean
    `must_understand`, and nothing else. `resolve` asks this of a field it
    has refined to JSON already, so a configuration is walked once.
    """
    if isinstance(value, str):
        return ()
    if not isinstance(value, Mapping):
        return (
            ValidationProblem(
                (),
                "expected a metadata field (string or extension object)",
                "invalid_type",
            ),
        )
    field = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = []
    for key in field:
        if not isinstance(key, str):
            problems.append(
                ValidationProblem((), f"non-string metadata field key {key!r}", "invalid_type")
            )
        elif key not in ENVELOPE_KEYS:
            problems.append(ValidationProblem((key,), f"unexpected key {key!r}", "unknown_key"))
    if not isinstance(field.get("name"), str):
        problems.append(ValidationProblem(("name",), "expected a string name", "invalid_type"))
    if "configuration" in field:
        configuration = field["configuration"]
        if not isinstance(configuration, Mapping):
            problems.append(
                ValidationProblem(("configuration",), "expected an object", "invalid_type")
            )
        elif not all(isinstance(k, str) for k in cast("Mapping[object, object]", configuration)):
            problems.append(
                ValidationProblem(("configuration",), "expected string keys", "invalid_type")
            )
    if "must_understand" in field:
        must_understand = field["must_understand"]
        if not isinstance(must_understand, bool):
            problems.append(
                ValidationProblem(("must_understand",), "expected a boolean", "invalid_type")
            )
        elif not allow_must_understand_false and not must_understand:
            problems.append(
                ValidationProblem(
                    ("must_understand",),
                    "false is not supported at this extension point",
                    "invalid_value",
                )
            )
    return tuple(problems)


def _configuration_json_problems(value: object) -> tuple[ValidationProblem, ...]:
    """Each member of `value`'s configuration that is not JSON, located; nothing where no object of string keys is there to walk."""
    if not isinstance(value, Mapping):
        return ()
    configuration = cast("Mapping[object, object]", value).get("configuration")
    if not isinstance(configuration, Mapping):
        return ()
    members = cast("Mapping[object, object]", configuration)
    if not all(isinstance(key, str) for key in members):
        return ()
    return tuple(
        found
        for key, item in cast("Mapping[str, object]", members).items()
        for found in prefixed("configuration", prefixed(key, validate_json(item)))
    )


def is_metadata_field_v3(value: object) -> TypeGuard[ZarrV3MetadataFieldJSON]:
    """Whether `value` is a v3 metadata field: a bare name or a named config."""
    if isinstance(value, str):
        return True
    if not isinstance(value, dict):
        return False
    field = cast("dict[object, object]", value)
    return is_canonical_json(field) and not validate_metadata_field_v3(field)


def parse_metadata_field_v3(value: object) -> ZarrV3MetadataFieldJSON:
    """Return `value` narrowed to `ZarrV3MetadataFieldJSON`, or raise `MetadataValidationError`."""
    problems = validate_metadata_field_v3(value)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast(ZarrV3MetadataFieldJSON, arrays_to_tuples(value))


__all__ = [
    "ENVELOPE_KEYS",
    "ChunkGridField",
    "ChunkKeyEncodingField",
    "CodecField",
    "DataTypeField",
    "StaticCodecField",
    "StorageTransformerField",
    "ZarrV3MetadataFieldJSON",
    "envelope_problems",
    "is_metadata_field_v3",
    "parse_metadata_field_v3",
    "validate_metadata_field_v3",
]
