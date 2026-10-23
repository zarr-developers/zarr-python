"""The v3 metadata field: its JSON type, the type aliases for members that hold a field, and the validators that check a single field.

This module is private. It sits below both code paths that read a field:
the model, which checks the fields of a document, and the definitions,
which read a field's configuration. Public consumers import
`ZarrV3MetadataFieldJSON` from `zarr_metadata.v3`, the aliases from
`zarr_metadata.v3.definition`, and the validators from
`zarr_metadata.model`.
"""

import functools
import re
from typing import Final, TypeAlias, TypeGuard, cast

from typing_extensions import TypeAliasType

from zarr_metadata._common import ZarrV3NamedConfigJSON
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    is_canonical_json,
    is_object,
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
    return with_input((*envelope, *_configuration_json_problems(value)), value)


ENVELOPE_KEYS: frozenset[str] = frozenset({"name", "configuration", "must_understand"})
"""The keys a metadata field's envelope declares. Any other key is an unknown key."""


_EXTENSION_NAME: Final = re.compile(r"[a-z][a-z0-9_.-]+")
"""A registered extension name: the spec's `^[a-z][a-z0-9-_.]+$`, with the hyphen moved last so no regex engine reads it as a range."""

_URI: Final = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")
"""A URI, to the extent this module checks names. It matches an RFC 3986 scheme, a colon, and one or more characters from the URI unreserved and reserved sets plus the `%` of percent-encoding (https://www.rfc-editor.org/rfc/rfc3986#section-2). Earlier versions of the spec required names of this form.

The characters are listed explicitly instead of written as `\\S`.
Python's `re` and ECMA-262 (the regex dialect of a JSON Schema
`pattern`) disagree on what `\\S` matches: `\\x1c`-`\\x1f` and `\\x85` are
whitespace in one, and `\\ufeff` is whitespace in the other. With `\\S`, a
schema validator could accept a name that this reader rejects, or the
reverse.
"""

EXTENSION_NAME_SCHEMA_PATTERN: Final = rf"^({_EXTENSION_NAME.pattern}|{_URI.pattern})(?![\s\S])"
r"""The names `well_named` accepts, as a JSON Schema `pattern`. It matches either pattern against the whole string. It uses `(?![\s\S])` instead of `$`, because `$` also matches before a final newline. `_RAW_BYTES_SCHEMA_PATTERN` does the same."""


@functools.lru_cache(maxsize=4096)
def well_named(name: str) -> bool:
    """Whether `name` is a valid extension name under the spec. Results are cached per name, because a document uses few names and a reader checks each one many times.

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
    """The problem to report at `at` when `name` is not a valid extension name under the spec; None when it is valid, as decided by `well_named`."""
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
    """Every reason `value` is not a valid v3 metadata field envelope. The contents of the configuration are not checked.

    The envelope is the part around the configuration: a `name` that is a
    valid extension name under the spec, a `configuration` that is an
    object with string keys, a boolean `must_understand`, and no other
    keys. `resolve` calls this on a field it has already narrowed to JSON,
    so the configuration is walked only once.
    """
    if isinstance(value, str):
        bad = name_problem(value, ())
        return () if bad is None else (bad,)
    if not is_object(value):
        return (
            ValidationProblem(
                (),
                f"expected a metadata field (string or extension object), got {shown(value)}",
                "invalid_type",
            ),
        )
    field = value
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
        if not is_object(configuration):
            problems.append(
                ValidationProblem(("configuration",), "expected an object", "invalid_type")
            )
        elif not all(isinstance(k, str) for k in configuration):
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
    """A located problem for each member of `value`'s configuration that is not JSON. Returns nothing if there is no configuration object with string keys to walk."""
    if not is_object(value):
        return ()
    configuration = value.get("configuration")
    if not is_object(configuration):
        return ()
    members = configuration
    if not all(isinstance(key, str) for key in members):
        return ()
    return tuple(
        found
        for key, item in members.items()
        if isinstance(key, str)
        for found in validate_json(item, ("configuration", key))
    )


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
    "ENVELOPE_KEYS",
    "EXTENSION_NAME_SCHEMA_PATTERN",
    "ZarrV3ChunkGridField",
    "ZarrV3ChunkKeyEncodingField",
    "ZarrV3CodecField",
    "ZarrV3DataTypeField",
    "ZarrV3MetadataFieldJSON",
    "ZarrV3StaticCodecField",
    "ZarrV3StorageTransformerField",
    "envelope_problems",
    "is_metadata_field_v3",
    "name_problem",
    "parse_metadata_field_v3",
    "validate_metadata_field_v3",
    "well_named",
]
