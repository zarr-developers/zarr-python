"""Structural validation for Zarr metadata documents.

Validators check JSON structure (shapes, key presence, primitive kinds),
not domain validity. Each concept gets a `validate_*` function returning
every problem found, an `is_*` type guard, and a `parse_*` function that
narrows or raises `MetadataValidationError`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from typing_extensions import TypeIs

from zarr_metadata import ArrayMetadataV2, ArrayMetadataV3, MetadataV3
from zarr_metadata._common import JSONValue


@dataclass(frozen=True, slots=True)
class ValidationProblem:
    """A single structural problem found while validating a metadata document.

    `loc` is the path from the document root to the offending value, e.g.
    `("codecs", 0, "name")`. An empty `loc` refers to the document as a whole.
    """

    loc: tuple[str | int, ...]
    message: str

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
    return [ValidationProblem((loc_head, *p.loc), p.message) for p in problems]


def validate_json(value: object) -> list[ValidationProblem]:
    """Return every reason `value` is not JSON-serializable (recursively)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return []
    problems: list[ValidationProblem] = []
    if isinstance(value, Mapping):
        for key, item in cast("Mapping[object, object]", value).items():
            if not isinstance(key, str):
                problems.append(ValidationProblem((), f"non-string key {key!r} in JSON object"))
                continue
            problems.extend(_prefix(key, validate_json(item)))
        return problems
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(cast("Sequence[object]", value)):
            problems.extend(_prefix(index, validate_json(item)))
        return problems
    return [ValidationProblem((), f"not a JSON-serializable value: {value!r}")]


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


def validate_metadata_field_v3(value: object) -> list[ValidationProblem]:
    """Return every reason `value` is not a v3 metadata field.

    A metadata field is a bare name string or a `{name, configuration}` mapping.
    """
    if isinstance(value, str):
        return []
    if not isinstance(value, Mapping):
        return [
            ValidationProblem((), "expected a metadata field (string or {name, configuration})")
        ]
    field = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = []
    if not isinstance(field.get("name"), str):
        problems.append(ValidationProblem(("name",), "expected a string name"))
    if "configuration" in field:
        configuration = field["configuration"]
        if not isinstance(configuration, Mapping):
            problems.append(ValidationProblem(("configuration",), "expected a mapping"))
        elif not all(isinstance(k, str) for k in cast("Mapping[object, object]", configuration)):
            problems.append(ValidationProblem(("configuration",), "expected string keys"))
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
    """Whether `value` is a non-string sequence of integers."""
    return (
        not isinstance(value, str)
        and isinstance(value, Sequence)
        and all(isinstance(item, int) for item in cast("Sequence[object]", value))
    )


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
        return [ValidationProblem(("attributes",), "expected a mapping with string keys")]
    return []


def validate_array_metadata_v3(value: object) -> list[ValidationProblem]:
    """Return every reason `value` is not a structurally-valid v3 array doc.

    Checks structure, not domain validity. Unknown top-level keys are allowed
    (they map to `extra_fields`).
    """
    if not isinstance(value, Mapping):
        return [ValidationProblem((), "expected a mapping")]
    doc = cast("Mapping[str, object]", value)
    problems: list[ValidationProblem] = [
        ValidationProblem((key,), "missing required key")
        for key in sorted(ARRAY_METADATA_REQUIRED_KEYS_V3 - doc.keys())
    ]
    if "shape" in doc and not _is_int_sequence(doc["shape"]):
        problems.append(ValidationProblem(("shape",), "expected a sequence of int"))
    if "fill_value" in doc:
        problems.extend(_prefix("fill_value", validate_json(doc["fill_value"])))
    for key in ("data_type", "chunk_grid", "chunk_key_encoding"):
        if key in doc:
            problems.extend(_prefix(key, validate_metadata_field_v3(doc[key])))
    for key in ("codecs", "storage_transformers"):
        if key in doc:
            entries = doc[key]
            if isinstance(entries, str) or not isinstance(entries, Sequence):
                problems.append(ValidationProblem((key,), "expected a sequence"))
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
            problems.append(ValidationProblem(("dimension_names",), "expected a sequence"))
        elif not all(
            item is None or isinstance(item, str) for item in cast("Sequence[object]", names)
        ):
            problems.append(
                ValidationProblem(("dimension_names",), "expected items of str or None")
            )
    return problems


def is_array_metadata_v3(value: object) -> TypeIs[ArrayMetadataV3]:
    """Whether `value` is a structurally-valid v3 array metadata document."""
    return not validate_array_metadata_v3(value)


def parse_array_metadata_v3(value: object) -> ArrayMetadataV3:
    """Return `value` narrowed to `ArrayMetadataV3`, or raise `MetadataValidationError`."""
    problems = validate_array_metadata_v3(value)
    if problems:
        raise MetadataValidationError(problems)
    return cast(ArrayMetadataV3, value)


def validate_array_metadata_v2(value: object) -> list[ValidationProblem]:
    """Return every reason `value` is not a structurally-valid v2 array doc.

    Checks structure, not domain validity. `compressor`/`filters` are required
    keys but may be `None`.
    """
    if not isinstance(value, Mapping):
        return [ValidationProblem((), "expected a mapping")]
    doc = cast("Mapping[str, object]", value)
    problems: list[ValidationProblem] = [
        ValidationProblem((key,), "missing required key")
        for key in sorted(ARRAY_METADATA_REQUIRED_KEYS_V2 - doc.keys())
    ]
    problems.extend(
        ValidationProblem((key,), "expected a sequence of int")
        for key in ("shape", "chunks")
        if key in doc and not _is_int_sequence(doc[key])
    )
    if "fill_value" in doc:
        problems.extend(_prefix("fill_value", validate_json(doc["fill_value"])))
    if "attributes" in doc:
        problems.extend(_validate_attributes(doc["attributes"]))
    return problems


def is_array_metadata_v2(value: object) -> TypeIs[ArrayMetadataV2]:
    """Whether `value` is a structurally-valid v2 array metadata document."""
    return not validate_array_metadata_v2(value)


def parse_array_metadata_v2(value: object) -> ArrayMetadataV2:
    """Return `value` narrowed to `ArrayMetadataV2`, or raise `MetadataValidationError`."""
    problems = validate_array_metadata_v2(value)
    if problems:
        raise MetadataValidationError(problems)
    return cast(ArrayMetadataV2, value)


def _arrays_to_tuples(obj: object) -> object:
    """Recursively convert every list in a JSON-decoded structure to a tuple."""
    if isinstance(obj, list):
        return tuple(_arrays_to_tuples(item) for item in cast("list[object]", obj))
    if isinstance(obj, dict):
        return {
            key: _arrays_to_tuples(value)
            for key, value in cast("dict[object, object]", obj).items()
        }
    return obj
