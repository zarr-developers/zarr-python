"""The v3 metadata field: its JSON, the aliases a member holding one is annotated with, and the validators that judge one on its own.

Private, and below both readers of a field: the model, which judges the
fields of a document, and the definitions, which read a field's
configuration. Public consumers import `ZarrV3MetadataFieldJSON` from
`zarr_metadata.v3`, the aliases from `zarr_metadata.v3.definition`, and
the validators from `zarr_metadata.model`.
"""

import re
from collections.abc import Mapping
from typing import Final, TypeAlias, TypeGuard, cast

from typing_extensions import TypeAliasType

from zarr_metadata._common import ZarrV3NamedConfigJSON
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    is_canonical_json,
    shown,
    shown_key,
    validate_json,
    with_input,
)

ZarrV3MetadataFieldJSON: TypeAlias = str | ZarrV3NamedConfigJSON
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
    whose members are JSON: an object of a `name` as the spec names an
    extension, a `configuration` that is an object of string keys, a
    boolean `must_understand`, and nothing else.
    """
    envelope = envelope_problems(value, allow_must_understand_false=allow_must_understand_false)
    return with_input((*envelope, *_configuration_json_problems(value)), value)


ENVELOPE_KEYS: frozenset[str] = frozenset({"name", "configuration", "must_understand"})
"""The members a metadata field's envelope declares: a key beside them is an unknown key."""


_EXTENSION_NAME: Final = re.compile(r"[a-z][a-z0-9_.-]+")
"""A registered extension name: the spec's `^[a-z][a-z0-9-_.]+$`, its hyphen last so no engine reads it as a range."""

_URI: Final = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")
"""A URI, as far as a name is judged: RFC 3986's scheme, its colon, and one or more of the characters a URI is written in -- its unreserved and reserved sets, and the `%` of percent-encoding (https://www.rfc-editor.org/rfc/rfc3986#section-2) -- the names earlier versions of the spec required.

The characters are listed rather than written `\\S`, which Python's `re`
and ECMA-262, the dialect a JSON Schema `pattern` is read in, disagree
on: `\\x1c`-`\\x1f` and `\\x85` are whitespace to one and `\\ufeff` to the
other, so a name this reader refused, a validator of the schema would
accept, or the other way round.
"""

EXTENSION_NAME_SCHEMA_PATTERN: Final = rf"^({_EXTENSION_NAME.pattern}|{_URI.pattern})(?![\s\S])"
r"""What `well_named` accepts, as a JSON Schema `pattern`: the two patterns it matches whole, `(?![\s\S])` where `$` would take a final newline, as `_RAW_BYTES_SCHEMA_PATTERN` has it."""


def well_named(name: str) -> bool:
    """Whether `name` is an extension name as the spec names one.

    A registered name "MUST start with one lower case letter a-z and then
    be followed by only lower case letters a-z, numerals 0-9, underscores,
    dots and dashes", regex `^[a-z][a-z0-9-_.]+$`
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1603-L1606);
    a URI, which earlier versions of the spec required, is "still
    permitted"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1614-L1618).
    """
    return _EXTENSION_NAME.fullmatch(name) is not None or _URI.fullmatch(name) is not None


def name_problem(name: str, at: tuple[str | int, ...]) -> ValidationProblem | None:
    """The problem `name`, at `at`, is when the spec gives no extension such a name; None when it does, as `well_named` says."""
    if well_named(name):
        return None
    message = (
        "expected an extension name -- lower-case letters, digits, '-', '_' and '.', "
        f"starting with a letter -- or a URI, got {shown(name)}"
    )
    return ValidationProblem(at, message, "invalid_value")


def envelope_problems(
    value: object, *, allow_must_understand_false: bool
) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a v3 metadata field's envelope, what its configuration holds left unjudged.

    The envelope is what sits around the configuration: a `name` as the
    spec names an extension, a `configuration` that is an object of string
    keys, a boolean `must_understand`, and nothing else. `resolve` asks
    this of a field it has refined to JSON already, so a configuration is
    walked once.
    """
    if isinstance(value, str):
        bad = name_problem(value, ())
        return () if bad is None else (bad,)
    if not isinstance(value, Mapping):
        return (
            ValidationProblem(
                (),
                f"expected a metadata field (string or extension object), got {shown(value)}",
                "invalid_type",
            ),
        )
    field = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = []
    for key in field:
        if not isinstance(key, str):
            problems.append(
                ValidationProblem(
                    (), f"non-string metadata field key {shown_key(key)}", "invalid_type"
                )
            )
        elif key not in ENVELOPE_KEYS:
            problems.append(ValidationProblem((key,), f"unexpected key {key!r}", "unknown_key"))
    if "name" not in field:
        problems.append(ValidationProblem(("name",), "missing required key", "missing_key"))
    elif not isinstance(field["name"], str):
        problems.append(ValidationProblem(("name",), "expected a string name", "invalid_type"))
    elif (bad := name_problem(field["name"], ("name",))) is not None:
        problems.append(bad)
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
        for found in validate_json(item, ("configuration", key))
    )


def is_metadata_field_v3(value: object) -> TypeGuard[ZarrV3MetadataFieldJSON]:
    """Whether `value` is a v3 metadata field: a bare name as the spec names an extension, or a named config."""
    if isinstance(value, str):
        return well_named(value)
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
    "EXTENSION_NAME_SCHEMA_PATTERN",
    "ChunkGridField",
    "ChunkKeyEncodingField",
    "CodecField",
    "DataTypeField",
    "StaticCodecField",
    "StorageTransformerField",
    "ZarrV3MetadataFieldJSON",
    "envelope_problems",
    "is_metadata_field_v3",
    "name_problem",
    "parse_metadata_field_v3",
    "validate_metadata_field_v3",
    "well_named",
]
