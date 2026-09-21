"""Whole-document validation and canonicalization: the door.

These `validate_*` and `parse_*` functions mirror the model API but apply
both validation layers, and `canonicalize_array_metadata_v3` answers with
`Canonical[T] | Invalid`: the document in its simplest equivalent
spelling, or every reason it is not valid. The work is done by the
documents themselves -- `zarr_metadata.v3._document` and
`zarr_metadata.v2._document` -- and what this module decides is which
entities are in scope while it asks.

There is deliberately no `is_*` counterpart: composition validity is
stricter than TypedDict membership, so a guard here could not narrow
honestly. Use `zarr_metadata.model.is_*` for type narrowing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, cast

from zarr_metadata.model._array import ZarrV3ArrayMetadata
from zarr_metadata.model._validation import (
    MetadataValidationError,
    arrays_to_tuples,
)
from zarr_metadata.model._validation import (
    validate_array_metadata_v2 as _validate_structure_v2,
)
from zarr_metadata.model._validation import (
    validate_array_metadata_v3 as _validate_structure_v3,
)
from zarr_metadata.model._validation import (
    validate_group_metadata_v2 as _validate_group_structure_v2,
)
from zarr_metadata.model._validation import (
    validate_group_metadata_v3 as _validate_group_structure_v3,
)
from zarr_metadata.v2._document import array_problems_v2
from zarr_metadata.v3._document import array_problems_v3, group_problems_v3, read_array_v3
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr_metadata.model._validation import ValidationProblem
    from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON
    from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
    from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON

    _StructuralValidator = Callable[[object], tuple[ValidationProblem, ...]]
    _SemanticValidator = Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]]

DocumentT = TypeVar("DocumentT")


@dataclass(frozen=True, slots=True)
class Canonical(Generic[DocumentT]):
    """A semantically valid document, in its simplest equivalent spelling."""

    document: DocumentT
    valid: Literal[True] = True


@dataclass(frozen=True, slots=True)
class Invalid:
    """Every reason a document is not semantically valid; never empty."""

    problems: tuple[ValidationProblem, ...]
    valid: Literal[False] = False

    def __post_init__(self) -> None:
        if len(self.problems) == 0:
            msg = "Invalid requires at least one validation problem"
            raise ValueError(msg)


def _no_semantics(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """v2 group documents carry no cross-field constraints."""
    return ()


def _array_semantics_v3(context: Context) -> _SemanticValidator:
    """The v3 array semantics, asked in `context`.

    The work is `zarr_metadata.v3` asking each entity about itself and
    about the parts of the document it meets; what this layer decides is
    which entities are in scope while it asks.
    """

    def judge(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
        return array_problems_v3(document, context)

    return judge


def _group_semantics_v3(context: Context) -> _SemanticValidator:
    """The v3 group semantics, asked in `context`.

    A group's only semantic content is its inline consolidated children,
    and those are array and group documents judged in the same scope.
    """

    def judge(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
        return group_problems_v3(document, context)

    return judge


def _judged(
    normalized: object, structure: _StructuralValidator, semantics: _SemanticValidator
) -> tuple[ValidationProblem, ...]:
    """Structural and semantic problems in an already-normalized document.

    Takes the normalized value rather than the caller's input so that
    `validate_*` and `parse_*` each walk the document once.
    """
    problems = structure(normalized)
    if isinstance(normalized, Mapping):
        problems = problems + semantics(cast("Mapping[str, object]", normalized))
    return tuple(problems)


def validate_array_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
) -> tuple[ValidationProblem, ...]:
    """Why `value` is not a valid v3 array document.

    Every structural problem, and every semantic problem that can be
    determined. One member that cannot be read costs the *composition*
    judgments about the entity holding it -- whether a shard's inner
    shape divides the array it is handed cannot be answered by a shard
    that could not be built -- so a document with two defects in one
    configuration may need a second pass. The verdict is never affected.

    Structural problems (from the model layer) and semantic problems
    (from the entities themselves) are reported together. JSON arrays are
    normalized to tuples before judgment, so list-spelled documents
    (e.g. fresh `json.loads` output) are judged at the canonical data
    level rather than rejected for their spelling.
    """
    return _judged(arrays_to_tuples(value), _validate_structure_v3, _array_semantics_v3(context))


def parse_array_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
) -> ZarrV3ArrayMetadataJSON:
    """Return `value` as a valid `ZarrV3ArrayMetadataJSON`, or raise.

    Normalizes JSON arrays to tuples, then raises a single
    `MetadataValidationError` carrying every structural and composition
    problem found.
    """
    normalized = arrays_to_tuples(value)
    problems = _judged(normalized, _validate_structure_v3, _array_semantics_v3(context))
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3ArrayMetadataJSON", normalized)


def validate_array_metadata_v2(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v2 array document (merged form).

    JSON arrays are normalized to tuples before judgment, as in
    `validate_array_metadata_v3`.
    """
    return _judged(arrays_to_tuples(value), _validate_structure_v2, array_problems_v2)


def parse_array_metadata_v2(value: object) -> ZarrV2ArrayMetadataJSON:
    """Return `value` as a valid `ZarrV2ArrayMetadataJSON`, or raise.

    Normalizes JSON arrays to tuples, then raises a single
    `MetadataValidationError` carrying every structural and composition
    problem found.
    """
    normalized = arrays_to_tuples(value)
    problems = _judged(normalized, _validate_structure_v2, array_problems_v2)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV2ArrayMetadataJSON", normalized)


def validate_group_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v3 group document.

    Composition rules recurse into inline consolidated metadata, so a
    consolidated child document invalid under its own rules is reported
    here, at its path.
    """
    return _judged(
        arrays_to_tuples(value), _validate_group_structure_v3, _group_semantics_v3(context)
    )


def parse_group_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
) -> ZarrV3GroupMetadataJSON:
    """Return `value` as a valid `ZarrV3GroupMetadataJSON`, or raise."""
    normalized = arrays_to_tuples(value)
    problems = _judged(normalized, _validate_group_structure_v3, _group_semantics_v3(context))
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3GroupMetadataJSON", normalized)


def validate_group_metadata_v2(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v2 group document (merged form).

    v2 group documents carry no composition constraints today, so this is
    the structural judgment, offered here for a uniform read-side API.
    """
    return _judged(arrays_to_tuples(value), _validate_group_structure_v2, _no_semantics)


def parse_group_metadata_v2(value: object) -> ZarrV2GroupMetadataJSON:
    """Return `value` as a valid `ZarrV2GroupMetadataJSON`, or raise."""
    normalized = arrays_to_tuples(value)
    problems = _judged(normalized, _validate_group_structure_v2, _no_semantics)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV2GroupMetadataJSON", normalized)


def canonicalize_array_metadata_v3(
    document: object, *, context: Context = CORE_AND_EXTENSIONS
) -> Canonical[ZarrV3ArrayMetadataJSON] | Invalid:
    """`document` in canonical form, or every reason it is not valid.

    Canonical means the simplest spelling with the same meaning, and the
    document decides that for itself in `ArrayDocumentV3.canonical`: each
    entity in its own canonical form, and `dimension_names` of nothing
    but nulls omitted. Two properties are worth holding on to, and
    `tests/rules/test_canonical.py` asserts both: canonicalizing twice
    changes nothing further, and canonicalizing never changes a verdict.

    Takes any value, like `validate_array_metadata_v3`: a document the
    model layer has not accepted is not an error -- the structural
    problems come back with the semantic ones, and the result is
    `Invalid` rather than a canonical document. The document is read
    once: the entities that judge it are the entities that are rewritten.

    Tell the two apart with `result.valid is True` or
    `isinstance(result, Invalid)`; pyright narrows the literal on a
    comparison, not on `if result.valid`.

    An entity whose canonical form breaks its own rules raises
    `MetadataValidationError` from here, as its constructor does: that
    is a bug in the entity, not a verdict on the document, which was
    valid.
    """
    normalized = arrays_to_tuples(document)
    problems = _validate_structure_v3(normalized)
    if not isinstance(normalized, Mapping):
        return Invalid(problems)
    array, found = read_array_v3(cast("Mapping[str, object]", normalized), context)
    problems = (*problems, *found, *array.problems())
    if len(problems) != 0:
        return Invalid(problems)
    return Canonical(ZarrV3ArrayMetadata.from_json(array.canonical().to_json()).to_json())


__all__ = [
    "Canonical",
    "Invalid",
    "canonicalize_array_metadata_v3",
    "parse_array_metadata_v2",
    "parse_array_metadata_v3",
    "parse_group_metadata_v2",
    "parse_group_metadata_v3",
    "validate_array_metadata_v2",
    "validate_array_metadata_v3",
    "validate_group_metadata_v2",
    "validate_group_metadata_v3",
]
