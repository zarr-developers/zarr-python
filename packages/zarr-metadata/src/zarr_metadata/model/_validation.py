"""Structural validation for Zarr metadata documents.

Validators check JSON structure (key presence, value shapes, and fixed
literals like `zarr_format`), not domain validity. Each concept gets a
`validate_*` function returning every problem found, an `is_*` type guard,
and a `parse_*` function that narrows or raises `MetadataValidationError`.
The guards are `TypeGuard`s, not `TypeIs`: True narrows a value to its
document type, and False says nothing about its type, since a value can be
well typed and still not a valid document.

Every `ValidationProblem` carries a machine-readable `kind` alongside its
human-readable `message`, so consumers can dispatch on the failure mode
(`missing_key`, `invalid_type`, `invalid_value`, `invalid_json`,
`unknown_key`) without string-matching messages.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Final, TypeGuard, TypeVar, cast

from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    refine_user_data,
    validate_json,
)
from zarr_metadata._json import is_canonical_json as _is_canonical_json
from zarr_metadata._json import prefixed as _prefix
from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON
from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON

# The standard top-level keys of a v3 array metadata document. Anything outside
# this set is an extension field. Built from the TypedDict's required/optional
# key sets (which resolve inherited keys, unlike `__annotations__`).
ARRAY_METADATA_REQUIRED_KEYS_V3: Final[frozenset[str]] = frozenset(
    ZarrV3ArrayMetadataJSON.__required_keys__
)
ARRAY_METADATA_OPTIONAL_KEYS_V3: Final[frozenset[str]] = frozenset(
    ZarrV3ArrayMetadataJSON.__optional_keys__
)
ARRAY_METADATA_STANDARD_KEYS_V3: Final[frozenset[str]] = (
    ARRAY_METADATA_REQUIRED_KEYS_V3 | ARRAY_METADATA_OPTIONAL_KEYS_V3
)

ARRAY_METADATA_REQUIRED_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2ArrayMetadataJSON.__required_keys__
)
ARRAY_METADATA_OPTIONAL_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2ArrayMetadataJSON.__optional_keys__
)
ARRAY_METADATA_STANDARD_KEYS_V2: Final[frozenset[str]] = (
    ARRAY_METADATA_REQUIRED_KEYS_V2 | ARRAY_METADATA_OPTIONAL_KEYS_V2
)

# The standard top-level keys of a v3 group metadata document. Anything outside
# this set is an extension field.
GROUP_METADATA_REQUIRED_KEYS_V3: Final[frozenset[str]] = frozenset(
    ZarrV3GroupMetadataJSON.__required_keys__
)
GROUP_METADATA_OPTIONAL_KEYS_V3: Final[frozenset[str]] = frozenset(
    ZarrV3GroupMetadataJSON.__optional_keys__
)
GROUP_METADATA_STANDARD_KEYS_V3: Final[frozenset[str]] = (
    GROUP_METADATA_REQUIRED_KEYS_V3 | GROUP_METADATA_OPTIONAL_KEYS_V3
)

GROUP_METADATA_REQUIRED_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2GroupMetadataJSON.__required_keys__
)
GROUP_METADATA_OPTIONAL_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2GroupMetadataJSON.__optional_keys__
)
GROUP_METADATA_STANDARD_KEYS_V2: Final[frozenset[str]] = (
    GROUP_METADATA_REQUIRED_KEYS_V2 | GROUP_METADATA_OPTIONAL_KEYS_V2
)


def _missing_keys(
    required: frozenset[str], doc: Mapping[object, object]
) -> tuple[ValidationProblem, ...]:
    """One `missing_key` problem per required key absent from `doc`."""
    return tuple(
        ValidationProblem((key,), "missing required key", "missing_key")
        for key in sorted(required - doc.keys())
    )


def _unexpected_keys(
    allowed: frozenset[str], doc: Mapping[object, object]
) -> tuple[ValidationProblem, ...]:
    """One problem per member outside a closed document's declared shape."""
    problems: list[ValidationProblem] = []
    for key in doc:
        if not isinstance(key, str):
            problems.append(
                ValidationProblem((), f"non-string document key {key!r}", "invalid_type")
            )
        elif key not in allowed:
            problems.append(
                ValidationProblem((key,), "unexpected document member", "invalid_value")
            )
    return tuple(problems)


def _check_literal(
    doc: Mapping[object, object], key: str, expected: object
) -> tuple[ValidationProblem, ...]:
    """One `invalid_value` problem if `doc[key]` is present but not `expected`."""
    if key in doc and (type(doc[key]) is not type(expected) or doc[key] != expected):
        return (
            ValidationProblem((key,), f"expected {expected!r}, got {doc[key]!r}", "invalid_value"),
        )
    return ()


def _validate_other_members(
    doc: Mapping[object, object],
    standard_keys: frozenset[str],
    *,
    additional_reserved_keys: frozenset[str] = frozenset(),
) -> tuple[ValidationProblem, ...]:
    """Every key a string, and every member outside `standard_keys` a JSON value.

    For a document open to other members: v3 extension fields, and the
    members a v2 array's readers ignore.
    """
    problems: list[ValidationProblem] = []
    reserved_keys = standard_keys | additional_reserved_keys
    for key, value in doc.items():
        if not isinstance(key, str):
            problems.append(
                ValidationProblem((), f"non-string top-level key {key!r}", "invalid_type")
            )
            continue
        if key in reserved_keys:
            continue
        problems.extend(_prefix(key, validate_json(value)))
    return tuple(problems)


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
                problems.extend(_prefix("configuration", _prefix(key, validate_json(item))))
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
    return _is_canonical_json(field) and not validate_metadata_field_v3(field)


def parse_metadata_field_v3(value: object) -> ZarrV3MetadataFieldJSON:
    """Return `value` narrowed to `ZarrV3MetadataFieldJSON`, or raise `MetadataValidationError`."""
    normalized = arrays_to_tuples(value)
    problems = validate_metadata_field_v3(normalized)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast(ZarrV3MetadataFieldJSON, normalized)


def _is_array(value: object) -> TypeGuard[Sequence[object]]:
    """Whether `value` reads as a JSON array: a sequence that is not a string or bytes.

    `str`, `bytes` and `bytearray` are sequences to Python, and none of them
    is an array to JSON. A `TypeGuard`, not a `TypeIs`: a `str` is a
    `Sequence[object]` this says no to.
    """
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_int_sequence(value: object) -> TypeGuard[Sequence[int]]:
    """Whether `value` is a JSON array of integers.

    JSON booleans decode to `bool`, which is an `int` subclass in Python but
    is not an integer in a metadata document, so booleans are excluded. A
    `TypeGuard`, not a `TypeIs`: `bytes` is a `Sequence[int]` this says no to.
    """
    return _is_array(value) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def _validate_dim_sequence(doc: Mapping[object, object], key: str) -> tuple[ValidationProblem, ...]:
    """Validate a dimension sequence (`shape` / `chunks`) if present in `doc`.

    Dimension lengths are non-negative integers.
    """
    if key not in doc:
        return ()
    value = doc[key]
    if not _is_int_sequence(value):
        return (ValidationProblem((key,), "expected a sequence of int", "invalid_type"),)
    if any(item < 0 for item in value):
        return (ValidationProblem((key,), "expected non-negative integers", "invalid_value"),)
    return ()


def _is_dtype_v2(value: object) -> bool:
    """Whether `value` is shaped like a v2 dtype: a string or field records.

    A field record is a `(name, dtype)` or `(name, dtype, shape)` sequence,
    where `dtype` is itself a string or nested field records and `shape` is a
    sequence of int. The string content is NOT interpreted — whether the
    string names a real dtype is domain validity, not structure.
    """
    if isinstance(value, str):
        return True
    if not _is_array(value):
        return False
    for record in value:
        if not _is_array(record) or len(record) not in (2, 3):
            return False
        if not isinstance(record[0], str):
            return False
        if not _is_dtype_v2(record[1]):
            return False
        if len(record) == 3 and not _is_int_sequence(record[2]):
            return False
    return True


def _is_canonical_dtype_v2(value: object) -> bool:
    """Whether a validated v2 dtype uses the tuple-backed public representation."""
    if isinstance(value, str):
        return True
    if not isinstance(value, tuple):
        return False
    for record in cast("tuple[object, ...]", value):
        if not isinstance(record, tuple):
            return False
        fields = cast("tuple[object, ...]", record)
        if not _is_canonical_dtype_v2(fields[1]):
            return False
        if len(fields) == 3 and not isinstance(fields[2], tuple):
            return False
    return True


def _is_canonical_metadata_field_v3(value: object) -> bool:
    """Whether a validated v3 metadata field has its declared runtime container type."""
    return isinstance(value, (str, dict))


def _is_canonical_array_metadata_v3(value: object) -> bool:
    """Whether a validated v3 array document matches `ZarrV3ArrayMetadataJSON` at runtime."""
    if not isinstance(value, dict):
        return False
    doc = cast("dict[str, object]", value)
    if not isinstance(doc["shape"], tuple) or not isinstance(doc["codecs"], tuple):
        return False
    if "storage_transformers" in doc and not isinstance(doc["storage_transformers"], tuple):
        return False
    if "dimension_names" in doc and not isinstance(doc["dimension_names"], tuple):
        return False
    if not all(
        _is_canonical_metadata_field_v3(doc[key])
        for key in ("data_type", "chunk_grid", "chunk_key_encoding")
    ):
        return False
    if not all(
        _is_canonical_metadata_field_v3(item) for item in cast("tuple[object, ...]", doc["codecs"])
    ):
        return False
    return "storage_transformers" not in doc or all(
        _is_canonical_metadata_field_v3(item)
        for item in cast("tuple[object, ...]", doc["storage_transformers"])
    )


def _is_canonical_array_metadata_v2(value: object) -> bool:
    """Whether a validated v2 array document matches `ZarrV2ArrayMetadataJSON` at runtime."""
    if not isinstance(value, dict):
        return False
    doc = cast("dict[str, object]", value)
    if not isinstance(doc["shape"], tuple) or not isinstance(doc["chunks"], tuple):
        return False
    if not _is_canonical_dtype_v2(doc["dtype"]):
        return False
    compressor = doc["compressor"]
    if compressor is not None and not isinstance(compressor, dict):
        return False
    filters = doc["filters"]
    return filters is None or (
        isinstance(filters, tuple)
        and all(isinstance(item, dict) for item in cast("tuple[object, ...]", filters))
    )


def _is_codec_v2(value: object) -> bool:
    """Whether `value` is shaped like a v2 codec config: a mapping with a string `id`."""
    return isinstance(value, Mapping) and isinstance(
        cast("Mapping[object, object]", value).get("id"), str
    )


def _validate_codec_v2(value: object) -> tuple[ValidationProblem, ...]:
    """Validate a v2 codec's required shape and JSON-valued configuration."""
    if not _is_codec_v2(value):
        return (
            ValidationProblem(
                (), "expected a codec configuration with a string 'id'", "invalid_type"
            ),
        )
    return validate_json(value)


def _validate_attributes(value: object) -> tuple[ValidationProblem, ...]:
    """Validate an `attributes` value: a mapping with string keys.

    Returns a problem at `("attributes",)` if it is not, else `[]`. Shared by the
    v2 and v3 validators. Unlike the other `validate_*` functions (which
    return value-relative locs for the caller to `_prefix`), this emits the
    already-parent-relative `("attributes",)` loc, since it is only ever called
    with a document's `attributes` value.
    """
    if not isinstance(value, Mapping) or not all(
        isinstance(k, str) for k in cast("Mapping[object, object]", value)
    ):
        return (
            ValidationProblem(
                ("attributes",), "expected a mapping with string keys", "invalid_type"
            ),
        )
    problems: list[ValidationProblem] = []
    for key, item in cast("Mapping[str, object]", value).items():
        problems.extend(refine_user_data(item, ("attributes", key))[1])
    return tuple(problems)


def validate_array_metadata_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a structurally-valid v3 array doc.

    Checks structure, not domain validity. Unknown top-level keys are allowed
    (they map to `extra_fields`).
    """
    if not isinstance(value, Mapping):
        return (ValidationProblem((), "expected a mapping", "invalid_type"),)
    doc = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = list(_missing_keys(ARRAY_METADATA_REQUIRED_KEYS_V3, doc))
    problems.extend(_validate_other_members(doc, ARRAY_METADATA_STANDARD_KEYS_V3))
    problems.extend(_check_literal(doc, "zarr_format", 3))
    problems.extend(_check_literal(doc, "node_type", "array"))
    problems.extend(_validate_dim_sequence(doc, "shape"))
    if "fill_value" in doc:
        problems.extend(_prefix("fill_value", validate_json(doc["fill_value"])))
    for key in ("data_type", "chunk_grid", "chunk_key_encoding"):
        if key in doc:
            problems.extend(
                _prefix(
                    key,
                    validate_metadata_field_v3(doc[key], allow_must_understand_false=False),
                )
            )
    for key in ("codecs", "storage_transformers"):
        if key in doc:
            entries = doc[key]
            if not _is_array(entries):
                problems.append(ValidationProblem((key,), "expected a sequence", "invalid_type"))
            else:
                if key == "codecs" and len(entries) == 0:
                    problems.append(
                        ValidationProblem(
                            ("codecs",), "expected at least one codec", "invalid_value"
                        )
                    )
                for index, entry in enumerate(entries):
                    problems.extend(_prefix(key, _prefix(index, validate_metadata_field_v3(entry))))
    if "attributes" in doc:
        problems.extend(_validate_attributes(doc["attributes"]))
    if "dimension_names" in doc:
        # Simple typed sequences (dimension_names, shape, chunks) report a single
        # field-level loc, not per-bad-item locs; per-index locs are reserved for
        # the metadata-field lists (codecs, storage_transformers).
        names = doc["dimension_names"]
        shape = doc.get("shape")
        if not _is_array(names):
            problems.append(
                ValidationProblem(("dimension_names",), "expected a sequence", "invalid_type")
            )
        elif not all(item is None or isinstance(item, str) for item in names):
            problems.append(
                ValidationProblem(
                    ("dimension_names",), "expected items of str or None", "invalid_type"
                )
            )
        elif _is_int_sequence(shape) and len(names) != len(shape):
            problems.append(
                ValidationProblem(
                    ("dimension_names",),
                    "expected one name per dimension of shape",
                    "invalid_value",
                )
            )
    return tuple(problems)


def is_array_metadata_v3(value: object) -> TypeGuard[ZarrV3ArrayMetadataJSON]:
    """Whether `value` is a structurally-valid v3 array metadata document."""
    return (
        _is_canonical_json(value, finite=False)
        and not validate_array_metadata_v3(value)
        and _is_canonical_array_metadata_v3(value)
    )


def parse_array_metadata_v3(value: object) -> ZarrV3ArrayMetadataJSON:
    """Return `value` as `ZarrV3ArrayMetadataJSON`, or raise `MetadataValidationError`."""
    normalized = arrays_to_tuples(value)
    problems = validate_array_metadata_v3(normalized)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3ArrayMetadataJSON", normalized)


def validate_array_metadata_v2(value: object) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a structurally-valid v2 array doc.

    Checks structure, not domain validity: `dtype` must be a string or field
    records, but the string content is not interpreted; `compressor` and
    `filters` are required keys that may be `None`, and otherwise must be
    codec configurations (mappings with a string `id`).
    """
    if not isinstance(value, Mapping):
        return (ValidationProblem((), "expected a mapping", "invalid_type"),)
    doc = cast("Mapping[object, object]", value)
    # Unlike the group document ("Other keys MUST NOT be present",
    # https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L313), the v2 array document is open: other keys "SHOULD NOT be
    # present within the metadata object and SHOULD be ignored by
    # implementations" (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L91-L92), so a member outside
    # ARRAY_METADATA_STANDARD_KEYS_V2 is not a problem for being there. Ignored
    # is not unchecked: it is JSON, and its key a string, as in v3.
    problems: list[ValidationProblem] = list(_missing_keys(ARRAY_METADATA_REQUIRED_KEYS_V2, doc))
    problems.extend(_validate_other_members(doc, ARRAY_METADATA_STANDARD_KEYS_V2))
    problems.extend(_check_literal(doc, "zarr_format", 2))
    shape_problems = _validate_dim_sequence(doc, "shape")
    chunks_problems = _validate_dim_sequence(doc, "chunks")
    problems.extend(shape_problems)
    problems.extend(chunks_problems)
    shape = doc.get("shape")
    chunks = doc.get("chunks")
    if (
        len(shape_problems) == 0
        and len(chunks_problems) == 0
        and _is_int_sequence(shape)
        and _is_int_sequence(chunks)
        and len(shape) != len(chunks)
    ):
        problems.append(
            ValidationProblem(
                ("chunks",),
                "expected the same number of dimensions as shape",
                "invalid_value",
            )
        )
    if "dtype" in doc and not _is_dtype_v2(doc["dtype"]):
        problems.append(
            ValidationProblem(
                ("dtype",),
                "expected a v2 dtype string or a sequence of field records",
                "invalid_type",
            )
        )
    if "order" in doc and doc["order"] not in ("C", "F"):
        problems.append(
            ValidationProblem(
                ("order",), f"expected 'C' or 'F', got {doc['order']!r}", "invalid_value"
            )
        )
    if "compressor" in doc:
        compressor = doc["compressor"]
        if compressor is not None:
            problems.extend(_prefix("compressor", _validate_codec_v2(compressor)))
    if "filters" in doc:
        filters = doc["filters"]
        if filters is not None and (
            not _is_array(filters) or not all(_is_codec_v2(item) for item in filters)
        ):
            problems.append(
                ValidationProblem(
                    ("filters",),
                    "expected null or a sequence of codec configurations with string 'id's",
                    "invalid_type",
                )
            )
        elif _is_array(filters):
            # "A list of JSON objects providing codec configurations, or
            # null" (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L76-L79): an empty list is a list.
            for index, item in enumerate(filters):
                problems.extend(_prefix("filters", _prefix(index, validate_json(item))))
    if "dimension_separator" in doc and doc["dimension_separator"] not in (".", "/"):
        problems.append(
            ValidationProblem(
                ("dimension_separator",),
                f"expected '.' or '/', got {doc['dimension_separator']!r}",
                "invalid_value",
            )
        )
    if "fill_value" in doc:
        problems.extend(_prefix("fill_value", validate_json(doc["fill_value"])))
    if "attributes" in doc:
        problems.extend(_validate_attributes(doc["attributes"]))
    return tuple(problems)


def is_array_metadata_v2(value: object) -> TypeGuard[ZarrV2ArrayMetadataJSON]:
    """Whether `value` is a structurally-valid v2 array metadata document."""
    return (
        _is_canonical_json(value, finite=False)
        and not validate_array_metadata_v2(value)
        and _is_canonical_array_metadata_v2(value)
    )


def parse_array_metadata_v2(value: object) -> ZarrV2ArrayMetadataJSON:
    """Return `value` as `ZarrV2ArrayMetadataJSON`, or raise `MetadataValidationError`."""
    normalized = arrays_to_tuples(value)
    problems = validate_array_metadata_v2(normalized)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV2ArrayMetadataJSON", normalized)


def validate_consolidated_metadata_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a valid inline consolidated envelope.

    Locs are value-relative (the caller prefixes with `consolidated_metadata`
    where appropriate). Entries recurse into the array and group document
    validators, so a validator verdict always agrees with what
    `ZarrV3ConsolidatedMetadata.from_json` accepts.
    """
    if not isinstance(value, Mapping):
        return (ValidationProblem((), "expected a mapping", "invalid_type"),)
    env = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = [
        ValidationProblem((key,), "missing required key", "missing_key")
        for key in ("kind", "must_understand", "metadata")
        if key not in env
    ]
    problems.extend(_unexpected_keys(frozenset({"kind", "must_understand", "metadata"}), env))
    problems.extend(_check_literal(env, "kind", "inline"))
    if "must_understand" in env and env["must_understand"] is not False:
        problems.append(ValidationProblem(("must_understand",), "expected False", "invalid_value"))
    if "metadata" in env:
        entries = env["metadata"]
        if not isinstance(entries, Mapping):
            problems.append(ValidationProblem(("metadata",), "expected a mapping", "invalid_type"))
        else:
            for key, entry in cast("Mapping[object, object]", entries).items():
                if not isinstance(key, str):
                    problems.append(
                        ValidationProblem(("metadata",), f"non-string key {key!r}", "invalid_type")
                    )
                    continue
                entry_obj: object = entry
                node_type: object = None
                if isinstance(entry, Mapping):
                    node_type = cast("Mapping[object, object]", entry).get("node_type")
                if node_type == "array":
                    problems.extend(
                        _prefix("metadata", _prefix(key, validate_array_metadata_v3(entry_obj)))
                    )
                elif node_type == "group":
                    problems.extend(
                        _prefix("metadata", _prefix(key, validate_group_metadata_v3(entry_obj)))
                    )
                else:
                    problems.append(
                        ValidationProblem(
                            ("metadata", key, "node_type"),
                            "expected 'array' or 'group'",
                            "invalid_value",
                        )
                    )
    return tuple(problems)


def validate_group_metadata_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a structurally-valid v3 group doc.

    Checks structure, not domain validity. Unknown top-level keys are allowed
    (they map to `extra_fields`); a `consolidated_metadata` key, if present,
    is deep-validated (envelope and entries) via
    `validate_consolidated_metadata_v3`.
    """
    if not isinstance(value, Mapping):
        return (ValidationProblem((), "expected a mapping", "invalid_type"),)
    doc = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = list(_missing_keys(GROUP_METADATA_REQUIRED_KEYS_V3, doc))
    problems.extend(
        _validate_other_members(
            doc,
            GROUP_METADATA_STANDARD_KEYS_V3,
            additional_reserved_keys=frozenset({"consolidated_metadata"}),
        )
    )
    problems.extend(_check_literal(doc, "zarr_format", 3))
    problems.extend(_check_literal(doc, "node_type", "group"))
    if "attributes" in doc:
        problems.extend(_validate_attributes(doc["attributes"]))
    if "consolidated_metadata" in doc and doc["consolidated_metadata"] is not None:
        # consolidated_metadata: null (a historical zarr-python bug) is
        # structurally accepted so those stores remain readable, but the model
        # repairs it to absence on read and never writes it back.
        problems.extend(
            _prefix(
                "consolidated_metadata",
                validate_consolidated_metadata_v3(doc["consolidated_metadata"]),
            )
        )
    return tuple(problems)


def is_group_metadata_v3(value: object) -> TypeGuard[ZarrV3GroupMetadataJSON]:
    """Whether `value` is a structurally-valid v3 group metadata document."""
    return _is_canonical_json(value, finite=False) and not validate_group_metadata_v3(value)


def parse_group_metadata_v3(value: object) -> ZarrV3GroupMetadataJSON:
    """Return `value` narrowed to `ZarrV3GroupMetadataJSON`, or raise `MetadataValidationError`."""
    normalized = arrays_to_tuples(value)
    problems = validate_group_metadata_v3(normalized)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast(ZarrV3GroupMetadataJSON, normalized)


def validate_group_metadata_v2(value: object) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a structurally-valid v2 group doc.

    Validates the in-memory merged form: the `.zgroup` fields plus an
    optional `attributes` mapping folded in from `.zattrs`.
    """
    if not isinstance(value, Mapping):
        return (ValidationProblem((), "expected a mapping", "invalid_type"),)
    doc = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = list(_missing_keys(GROUP_METADATA_REQUIRED_KEYS_V2, doc))
    problems.extend(_unexpected_keys(GROUP_METADATA_STANDARD_KEYS_V2, doc))
    problems.extend(_check_literal(doc, "zarr_format", 2))
    if "attributes" in doc:
        problems.extend(_validate_attributes(doc["attributes"]))
    return tuple(problems)


def is_group_metadata_v2(value: object) -> TypeGuard[ZarrV2GroupMetadataJSON]:
    """Whether `value` is a structurally-valid v2 group metadata document."""
    return _is_canonical_json(value, finite=False) and not validate_group_metadata_v2(value)


def parse_group_metadata_v2(value: object) -> ZarrV2GroupMetadataJSON:
    """Return `value` narrowed to `ZarrV2GroupMetadataJSON`, or raise `MetadataValidationError`."""
    normalized = arrays_to_tuples(value)
    problems = validate_group_metadata_v2(normalized)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast(ZarrV2GroupMetadataJSON, normalized)


StoreKey = TypeVar("StoreKey", bound=str)
"""The key type of a mapping of store keys to bytes: `str`, or the literal keys one document names."""


def load_store_json(mapping: Mapping[StoreKey, bytes], key: str) -> object:
    """Decode the JSON document stored at `key` in `mapping`.

    Returns `object`, not `Any`: what a store holds is unknown until a
    validator says otherwise, and `Any` would let unchecked values flow
    into typed positions silently. Narrow the result with a `parse_*`.

    Decoding is Python's, so `NaN`, `Infinity` and `-Infinity` are read as
    the floats they spell, as zarr-python writes attributes; where one may
    be is the document's validator's to say. Every ingestion failure here
    surfaces as `MetadataValidationError`: a missing store key is a
    `missing_key` problem, a value that is not `bytes` an `invalid_type`
    problem, and undecodable bytes an `invalid_json` problem, rather than
    leaking `KeyError`, `TypeError` or `json.JSONDecodeError` to
    callers.
    """
    # Read by a `str` key whatever narrower key type the mapping declares:
    # a key it does not hold is only absent.
    stored = cast("Mapping[str, bytes]", mapping)
    if key not in stored:
        raise MetadataValidationError(
            [ValidationProblem((key,), "missing store key", "missing_key")]
        )
    # The runtime half of the annotation: `json.loads` decodes a `str` and
    # raises `TypeError` on most else.
    raw = cast("object", stored[key])
    if not isinstance(raw, bytes):
        raise MetadataValidationError(
            [ValidationProblem((key,), f"expected bytes, got {type(raw).__name__}", "invalid_type")]
        )
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise MetadataValidationError(
            [ValidationProblem((key,), f"invalid JSON: {exc}", "invalid_json")]
        ) from exc


def dump_store_json(value: object, *, indent: int | str | None = None) -> bytes:
    """Encode a document its validator has passed as JSON bytes.

    A non-finite number is written as Python's `json` writes it (`NaN`,
    `Infinity`, `-Infinity`), as zarr-python writes attributes; the
    validator is what keeps one out of anywhere else.
    """
    return json.dumps(value, indent=indent, allow_nan=True).encode("utf-8")
