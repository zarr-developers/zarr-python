"""The v3 metadata field: its JSON, and the validators that judge one on its own.

Private, and below both readers of a field: the model, which judges the
fields of a document, and the definitions, which read a field's
configuration. Public consumers import `ZarrV3MetadataFieldJSON` from
`zarr_metadata.v3`, and the validators from `zarr_metadata.model`.
"""

from collections.abc import Mapping
from typing import TypeGuard, cast

from zarr_metadata._common import ZarrV3NamedConfigJSON
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    is_canonical_json,
    prefixed,
    validate_json,
)

ZarrV3MetadataFieldJSON = str | ZarrV3NamedConfigJSON
"""The JSON shape of any v3 metadata extension-point entry: either a bare
short-hand name string or a `{name, configuration, must_understand}` envelope.

Used for `data_type`, `chunk_grid`, `chunk_key_encoding`, individual
codec entries, and `storage_transformers` in v3 array metadata, and for
the inner `codecs` / `index_codecs` lists of the `sharding_indexed`
codec.
"""


def validate_metadata_field_v3(
    value: object, *, allow_must_understand_false: bool = True
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a v3 metadata field.

    A metadata field is a bare name string or a mapping containing `name` and
    optional `configuration` and `must_understand` members.
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
    allowed_keys = frozenset({"name", "configuration", "must_understand"})
    for key in field:
        if not isinstance(key, str):
            problems.append(
                ValidationProblem((), f"non-string metadata field key {key!r}", "invalid_type")
            )
        elif key not in allowed_keys:
            problems.append(
                ValidationProblem((key,), "unexpected metadata field member", "invalid_value")
            )
    if not isinstance(field.get("name"), str):
        problems.append(ValidationProblem(("name",), "expected a string name", "invalid_type"))
    if "configuration" in field:
        configuration = field["configuration"]
        if not isinstance(configuration, Mapping):
            problems.append(
                ValidationProblem(("configuration",), "expected a mapping", "invalid_type")
            )
        elif not all(isinstance(k, str) for k in cast("Mapping[object, object]", configuration)):
            problems.append(
                ValidationProblem(("configuration",), "expected string keys", "invalid_type")
            )
        else:
            for key, item in cast("Mapping[str, object]", configuration).items():
                problems.extend(prefixed("configuration", prefixed(key, validate_json(item))))
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
    normalized = arrays_to_tuples(value)
    problems = validate_metadata_field_v3(normalized)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast(ZarrV3MetadataFieldJSON, normalized)


__all__ = [
    "ZarrV3MetadataFieldJSON",
    "is_metadata_field_v3",
    "parse_metadata_field_v3",
    "validate_metadata_field_v3",
]
