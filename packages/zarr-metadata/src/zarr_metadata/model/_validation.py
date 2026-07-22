"""Structural validation for Zarr metadata documents.

Validators check JSON structure (key presence, value shapes, and fixed
literals like `zarr_format`), not domain validity. Each concept gets a
`validate_*` function returning every problem found, an `is_*` type guard,
and a `parse_*` function that narrows or raises `MetadataValidationError`.

Every `ValidationProblem` carries a machine-readable `kind` alongside its
human-readable `message`, so consumers can dispatch on the failure mode
(`missing_key`, `invalid_type`, `invalid_value`, `invalid_json`) without
string-matching messages.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

from typing_extensions import TypeIs

from zarr_metadata._common import JSONValue
from zarr_metadata.v2.array import ArrayMetadataV2
from zarr_metadata.v2.group import GroupMetadataV2
from zarr_metadata.v3._common import MetadataV3
from zarr_metadata.v3.array import ArrayMetadataV3
from zarr_metadata.v3.group import GroupMetadataV3

ProblemKind = Literal["missing_key", "invalid_type", "invalid_value", "invalid_json"]
"""Machine-readable classification of a `ValidationProblem`.

- `missing_key`: a required key (document key or store key) is absent.
- `invalid_type`: a value has the wrong structural type (e.g. a string where
  a mapping is required, a non-JSON-serializable object).
- `invalid_value`: a value has an acceptable type but an invalid content
  (e.g. `zarr_format: 2` in a v3 document, `order: "Q"`).
- `invalid_json`: bytes that do not decode as JSON.
"""


@dataclass(frozen=True, slots=True)
class ValidationProblem:
    """A single structural problem found while validating a metadata document.

    `loc` is the path from the document root to the offending value, e.g.
    `("codecs", 0, "name")`. An empty `loc` refers to the document as a whole.
    `kind` classifies the failure mode for programmatic dispatch; `message`
    is the human-readable description.
    """

    loc: tuple[str | int, ...]
    message: str
    kind: ProblemKind

    def __str__(self) -> str:
        location = ".".join(str(part) for part in self.loc) if self.loc else "<root>"
        return f"{location}: {self.message}"


class MetadataValidationError(ValueError):
    """Raised when a value fails structural metadata validation.

    Carries every problem found (not just the first) in `.problems`.
    """

    def __init__(self, problems: list[ValidationProblem]) -> None:
        self.problems = problems
        super().__init__("\n".join(str(problem) for problem in problems))


def _prefix(loc_head: str | int, problems: list[ValidationProblem]) -> list[ValidationProblem]:
    """Prepend `loc_head` to the `loc` of every problem (for nested validators)."""
    return [ValidationProblem((loc_head, *p.loc), p.message, p.kind) for p in problems]


def validate_json(value: object) -> list[ValidationProblem]:
    """Return every reason `value` is not JSON-serializable (recursively)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return []
    problems: list[ValidationProblem] = []
    if isinstance(value, Mapping):
        for key, item in cast("Mapping[object, object]", value).items():
            if not isinstance(key, str):
                problems.append(
                    ValidationProblem((), f"non-string key {key!r} in JSON object", "invalid_type")
                )
                continue
            problems.extend(_prefix(key, validate_json(item)))
        return problems
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(cast("Sequence[object]", value)):
            problems.extend(_prefix(index, validate_json(item)))
        return problems
    return [ValidationProblem((), f"not a JSON-serializable value: {value!r}", "invalid_type")]


def is_json(value: object) -> TypeIs[JSONValue]:
    """Whether `value` is a JSON-serializable structure (recursively)."""
    return not validate_json(value)


def parse_json(value: object) -> JSONValue:
    """Return `value` narrowed to `JSONValue`, or raise `MetadataValidationError`."""
    problems = validate_json(value)
    if problems:
        raise MetadataValidationError(problems)
    return cast(JSONValue, value)


# The standard top-level keys of a v3 array metadata document. Anything outside
# this set is an extension field. Built from the TypedDict's required/optional
# key sets (which resolve inherited keys, unlike `__annotations__`).
ARRAY_METADATA_REQUIRED_KEYS_V3: Final[frozenset[str]] = frozenset(
    ArrayMetadataV3.__required_keys__
)
ARRAY_METADATA_OPTIONAL_KEYS_V3: Final[frozenset[str]] = frozenset(
    ArrayMetadataV3.__optional_keys__
)
ARRAY_METADATA_STANDARD_KEYS_V3: Final[frozenset[str]] = (
    ARRAY_METADATA_REQUIRED_KEYS_V3 | ARRAY_METADATA_OPTIONAL_KEYS_V3
)

ARRAY_METADATA_REQUIRED_KEYS_V2: Final[frozenset[str]] = frozenset(
    ArrayMetadataV2.__required_keys__
)

# The standard top-level keys of a v3 group metadata document. Anything outside
# this set is an extension field.
GROUP_METADATA_REQUIRED_KEYS_V3: Final[frozenset[str]] = frozenset(
    GroupMetadataV3.__required_keys__
)
GROUP_METADATA_OPTIONAL_KEYS_V3: Final[frozenset[str]] = frozenset(
    GroupMetadataV3.__optional_keys__
)
GROUP_METADATA_STANDARD_KEYS_V3: Final[frozenset[str]] = (
    GROUP_METADATA_REQUIRED_KEYS_V3 | GROUP_METADATA_OPTIONAL_KEYS_V3
)

GROUP_METADATA_REQUIRED_KEYS_V2: Final[frozenset[str]] = frozenset(
    GroupMetadataV2.__required_keys__
)


def _missing_keys(required: frozenset[str], doc: Mapping[str, object]) -> list[ValidationProblem]:
    """One `missing_key` problem per required key absent from `doc`."""
    return [
        ValidationProblem((key,), "missing required key", "missing_key")
        for key in sorted(required - doc.keys())
    ]


def _check_literal(
    doc: Mapping[str, object], key: str, expected: object
) -> list[ValidationProblem]:
    """One `invalid_value` problem if `doc[key]` is present but not `expected`."""
    if key in doc and doc[key] != expected:
        return [
            ValidationProblem((key,), f"expected {expected!r}, got {doc[key]!r}", "invalid_value")
        ]
    return []


def _validate_extension_fields_v3(
    doc: Mapping[object, object],
    standard_keys: frozenset[str],
    *,
    additional_reserved_keys: frozenset[str] = frozenset(),
) -> list[ValidationProblem]:
    """Validate v3 top-level key types and unknown-field JSON payloads."""
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
    return problems


def validate_metadata_field_v3(value: object) -> list[ValidationProblem]:
    """Return every reason `value` is not a v3 metadata field.

    A metadata field is a bare name string or a `{name, configuration}` mapping.
    """
    if isinstance(value, str):
        return []
    if not isinstance(value, Mapping):
        return [
            ValidationProblem(
                (),
                "expected a metadata field (string or {name, configuration})",
                "invalid_type",
            )
        ]
    field = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = []
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
    return problems


def is_metadata_field_v3(value: object) -> TypeIs[MetadataV3]:
    """Whether `value` is a v3 metadata field: a bare name or a named config."""
    return not validate_metadata_field_v3(value)


def parse_metadata_field_v3(value: object) -> MetadataV3:
    """Return `value` narrowed to `MetadataV3`, or raise `MetadataValidationError`."""
    problems = validate_metadata_field_v3(value)
    if problems:
        raise MetadataValidationError(problems)
    return cast(MetadataV3, value)


def _is_int_sequence(value: object) -> bool:
    """Whether `value` is a non-string sequence of integers.

    JSON booleans decode to `bool`, which is an `int` subclass in Python but
    is not an integer in a metadata document, so booleans are excluded.
    """
    return (
        not isinstance(value, (str, bytes, bytearray))
        and isinstance(value, Sequence)
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in cast("Sequence[object]", value)
        )
    )


def _validate_dim_sequence(doc: Mapping[str, object], key: str) -> list[ValidationProblem]:
    """Validate a dimension sequence (`shape` / `chunks`) if present in `doc`.

    Dimension lengths are non-negative integers.
    """
    if key not in doc:
        return []
    value = doc[key]
    if not _is_int_sequence(value):
        return [ValidationProblem((key,), "expected a sequence of int", "invalid_type")]
    if any(item < 0 for item in cast("Sequence[int]", value)):
        return [ValidationProblem((key,), "expected non-negative integers", "invalid_value")]
    return []


def _is_dtype_v2(value: object) -> bool:
    """Whether `value` is shaped like a v2 dtype: a string or field records.

    A field record is a `(name, dtype)` or `(name, dtype, shape)` sequence,
    where `dtype` is itself a string or nested field records and `shape` is a
    sequence of int. The string content is NOT interpreted — whether the
    string names a real dtype is domain validity, not structure.
    """
    if isinstance(value, str):
        return True
    if not isinstance(value, Sequence):
        return False
    for record in cast("Sequence[object]", value):
        if isinstance(record, str) or not isinstance(record, Sequence):
            return False
        fields = cast("Sequence[object]", record)
        if len(fields) not in (2, 3):
            return False
        if not isinstance(fields[0], str):
            return False
        if not _is_dtype_v2(fields[1]):
            return False
        if len(fields) == 3 and not _is_int_sequence(fields[2]):
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
    """Whether a validated v3 array document matches `ArrayMetadataV3` at runtime."""
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
    if "storage_transformers" in doc and not all(
        _is_canonical_metadata_field_v3(item)
        for item in cast("tuple[object, ...]", doc["storage_transformers"])
    ):
        return False
    return all(
        key in ARRAY_METADATA_STANDARD_KEYS_V3 or isinstance(item, dict)
        for key, item in doc.items()
    )


def _is_canonical_array_metadata_v2(value: object) -> bool:
    """Whether a validated v2 array document matches `ArrayMetadataV2` at runtime."""
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


def _validate_codec_v2(value: object) -> list[ValidationProblem]:
    """Validate a v2 codec's required shape and JSON-valued configuration."""
    if not _is_codec_v2(value):
        return [
            ValidationProblem(
                (), "expected a codec configuration with a string 'id'", "invalid_type"
            )
        ]
    return validate_json(value)


def _validate_attributes(value: object) -> list[ValidationProblem]:
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
        return [
            ValidationProblem(
                ("attributes",), "expected a mapping with string keys", "invalid_type"
            )
        ]
    problems: list[ValidationProblem] = []
    for key, item in cast("Mapping[str, object]", value).items():
        problems.extend(_prefix("attributes", _prefix(key, validate_json(item))))
    return problems


def validate_array_metadata_v3(value: object) -> list[ValidationProblem]:
    """Return every reason `value` is not a structurally-valid v3 array doc.

    Checks structure, not domain validity. Unknown top-level keys are allowed
    (they map to `extra_fields`).
    """
    if not isinstance(value, Mapping):
        return [ValidationProblem((), "expected a mapping", "invalid_type")]
    doc = cast("Mapping[str, object]", value)
    problems: list[ValidationProblem] = _missing_keys(ARRAY_METADATA_REQUIRED_KEYS_V3, doc)
    problems.extend(
        _validate_extension_fields_v3(
            cast("Mapping[object, object]", value), ARRAY_METADATA_STANDARD_KEYS_V3
        )
    )
    problems.extend(_check_literal(doc, "zarr_format", 3))
    problems.extend(_check_literal(doc, "node_type", "array"))
    problems.extend(_validate_dim_sequence(doc, "shape"))
    if "fill_value" in doc:
        problems.extend(_prefix("fill_value", validate_json(doc["fill_value"])))
    for key in ("data_type", "chunk_grid", "chunk_key_encoding"):
        if key in doc:
            problems.extend(_prefix(key, validate_metadata_field_v3(doc[key])))
    for key in ("codecs", "storage_transformers"):
        if key in doc:
            entries = doc[key]
            if isinstance(entries, str) or not isinstance(entries, Sequence):
                problems.append(ValidationProblem((key,), "expected a sequence", "invalid_type"))
            else:
                for index, entry in enumerate(cast("Sequence[object]", entries)):
                    problems.extend(_prefix(key, _prefix(index, validate_metadata_field_v3(entry))))
    if "attributes" in doc:
        problems.extend(_validate_attributes(doc["attributes"]))
    if "dimension_names" in doc:
        # Simple typed sequences (dimension_names, shape, chunks) report a single
        # field-level loc, not per-bad-item locs; per-index locs are reserved for
        # the metadata-field lists (codecs, storage_transformers).
        names = doc["dimension_names"]
        if isinstance(names, str) or not isinstance(names, Sequence):
            problems.append(
                ValidationProblem(("dimension_names",), "expected a sequence", "invalid_type")
            )
        elif not all(
            item is None or isinstance(item, str) for item in cast("Sequence[object]", names)
        ):
            problems.append(
                ValidationProblem(
                    ("dimension_names",), "expected items of str or None", "invalid_type"
                )
            )
        elif _is_int_sequence(doc.get("shape")) and len(cast("Sequence[object]", names)) != len(
            cast("Sequence[int]", doc["shape"])
        ):
            problems.append(
                ValidationProblem(
                    ("dimension_names",),
                    "expected one name per dimension of shape",
                    "invalid_value",
                )
            )
    return problems


def is_array_metadata_v3(value: object) -> TypeIs[ArrayMetadataV3]:
    """Whether `value` is a structurally-valid v3 array metadata document."""
    return not validate_array_metadata_v3(value) and _is_canonical_array_metadata_v3(value)


def parse_array_metadata_v3(value: object) -> ArrayMetadataV3:
    """Return `value` narrowed to `ArrayMetadataV3`, or raise `MetadataValidationError`."""
    normalized = arrays_to_tuples(value)
    problems = validate_array_metadata_v3(normalized)
    if problems:
        raise MetadataValidationError(problems)
    return cast(ArrayMetadataV3, normalized)


def validate_array_metadata_v2(value: object) -> list[ValidationProblem]:
    """Return every reason `value` is not a structurally-valid v2 array doc.

    Checks structure, not domain validity: `dtype` must be a string or field
    records, but the string content is not interpreted; `compressor` and
    `filters` are required keys that may be `None`, and otherwise must be
    codec configurations (mappings with a string `id`).
    """
    if not isinstance(value, Mapping):
        return [ValidationProblem((), "expected a mapping", "invalid_type")]
    doc = cast("Mapping[str, object]", value)
    problems: list[ValidationProblem] = _missing_keys(ARRAY_METADATA_REQUIRED_KEYS_V2, doc)
    problems.extend(_check_literal(doc, "zarr_format", 2))
    problems.extend(_validate_dim_sequence(doc, "shape"))
    problems.extend(_validate_dim_sequence(doc, "chunks"))
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
            isinstance(filters, str)
            or not isinstance(filters, Sequence)
            or not all(_is_codec_v2(item) for item in cast("Sequence[object]", filters))
        ):
            problems.append(
                ValidationProblem(
                    ("filters",),
                    "expected null or a sequence of codec configurations with string 'id's",
                    "invalid_type",
                )
            )
        elif filters is not None:
            for index, item in enumerate(cast("Sequence[object]", filters)):
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
    return problems


def is_array_metadata_v2(value: object) -> TypeIs[ArrayMetadataV2]:
    """Whether `value` is a structurally-valid v2 array metadata document."""
    return not validate_array_metadata_v2(value) and _is_canonical_array_metadata_v2(value)


def parse_array_metadata_v2(value: object) -> ArrayMetadataV2:
    """Return `value` narrowed to `ArrayMetadataV2`, or raise `MetadataValidationError`."""
    normalized = arrays_to_tuples(value)
    problems = validate_array_metadata_v2(normalized)
    if problems:
        raise MetadataValidationError(problems)
    return cast(ArrayMetadataV2, normalized)


def validate_consolidated_metadata_v3(value: object) -> list[ValidationProblem]:
    """Return every reason `value` is not a valid inline consolidated envelope.

    Locs are value-relative (the caller prefixes with `consolidated_metadata`
    where appropriate). Entries recurse into the array and group document
    validators, so a validator verdict always agrees with what
    `ConsolidatedMetadataModelV3.from_json` accepts.
    """
    if not isinstance(value, Mapping):
        return [ValidationProblem((), "expected a mapping", "invalid_type")]
    env = cast("Mapping[str, object]", value)
    problems: list[ValidationProblem] = [
        ValidationProblem((key,), "missing required key", "missing_key")
        for key in ("kind", "must_understand", "metadata")
        if key not in env
    ]
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
                    node_type = cast("Mapping[str, object]", entry).get("node_type")
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
    return problems


def validate_group_metadata_v3(value: object) -> list[ValidationProblem]:
    """Return every reason `value` is not a structurally-valid v3 group doc.

    Checks structure, not domain validity. Unknown top-level keys are allowed
    (they map to `extra_fields`); a `consolidated_metadata` key, if present,
    is deep-validated (envelope and entries) via
    `validate_consolidated_metadata_v3`.
    """
    if not isinstance(value, Mapping):
        return [ValidationProblem((), "expected a mapping", "invalid_type")]
    doc = cast("Mapping[str, object]", value)
    problems: list[ValidationProblem] = _missing_keys(GROUP_METADATA_REQUIRED_KEYS_V3, doc)
    problems.extend(
        _validate_extension_fields_v3(
            cast("Mapping[object, object]", value),
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
    return problems


def is_group_metadata_v3(value: object) -> TypeIs[GroupMetadataV3]:
    """Whether `value` is a structurally-valid v3 group metadata document."""
    return not validate_group_metadata_v3(value)


def parse_group_metadata_v3(value: object) -> GroupMetadataV3:
    """Return `value` narrowed to `GroupMetadataV3`, or raise `MetadataValidationError`."""
    problems = validate_group_metadata_v3(value)
    if problems:
        raise MetadataValidationError(problems)
    return cast(GroupMetadataV3, value)


def validate_group_metadata_v2(value: object) -> list[ValidationProblem]:
    """Return every reason `value` is not a structurally-valid v2 group doc.

    Validates the in-memory merged form: the `.zgroup` fields plus an
    optional `attributes` mapping folded in from `.zattrs`.
    """
    if not isinstance(value, Mapping):
        return [ValidationProblem((), "expected a mapping", "invalid_type")]
    doc = cast("Mapping[str, object]", value)
    problems: list[ValidationProblem] = _missing_keys(GROUP_METADATA_REQUIRED_KEYS_V2, doc)
    problems.extend(_check_literal(doc, "zarr_format", 2))
    if "attributes" in doc:
        problems.extend(_validate_attributes(doc["attributes"]))
    return problems


def is_group_metadata_v2(value: object) -> TypeIs[GroupMetadataV2]:
    """Whether `value` is a structurally-valid v2 group metadata document."""
    return not validate_group_metadata_v2(value)


def parse_group_metadata_v2(value: object) -> GroupMetadataV2:
    """Return `value` narrowed to `GroupMetadataV2`, or raise `MetadataValidationError`."""
    problems = validate_group_metadata_v2(value)
    if problems:
        raise MetadataValidationError(problems)
    return cast(GroupMetadataV2, value)


def load_store_json(mapping: Mapping[str, bytes], key: str) -> Any:
    """Decode the JSON document stored at `key` in `mapping`.

    Every ingestion failure surfaces as `MetadataValidationError`: a missing
    store key is a `missing_key` problem and undecodable bytes are an
    `invalid_json` problem, rather than leaking `KeyError` /
    `json.JSONDecodeError` to callers.
    """
    if key not in mapping:
        raise MetadataValidationError(
            [ValidationProblem((key,), "missing store key", "missing_key")]
        )
    try:
        return json.loads(mapping[key])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataValidationError(
            [ValidationProblem((key,), f"invalid JSON: {exc}", "invalid_json")]
        ) from exc


def arrays_to_tuples(obj: object) -> object:
    """Recursively convert every list in a JSON-decoded structure to a tuple."""
    if isinstance(obj, list):
        return tuple(arrays_to_tuples(item) for item in cast("list[object]", obj))
    if isinstance(obj, dict):
        mapping = cast("dict[object, object]", obj)
        converted: dict[object, object] = {
            key: arrays_to_tuples(value) for key, value in mapping.items()
        }
        if all(converted[key] is value for key, value in mapping.items()):
            return cast("object", obj)
        return converted
    return obj
