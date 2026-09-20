"""Whole-document structural and composition validation.

These `validate_*` and `parse_*` functions mirror the model API but apply
both validation layers. There is deliberately no `is_*` counterpart:
composition validity is stricter than TypedDict membership, so a guard
here could not narrow honestly. Use `zarr_metadata.model.is_*` for type
narrowing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

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
from zarr_metadata.rules._v2_array import array_problems_v2
from zarr_metadata.rules._v3_group import group_problems_v3
from zarr_metadata.v3._document import array_problems_v3
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr_metadata.model._validation import ValidationProblem
    from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON
    from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
    from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON

    _StructuralValidator = Callable[[object], tuple[ValidationProblem, ...]]
    _SemanticValidator = Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]]


def _no_semantics(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """v2 group documents carry no cross-field constraints."""
    return ()


def _array_semantics_v3(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """The v3 array semantics, in the scope this layer chooses.

    The work is `zarr_metadata.v3` asking each entity about itself and
    about the parts of the document it meets; what this layer decides is
    which entities are in scope while it asks.
    """
    return array_problems_v3(document, CORE_AND_EXTENSIONS)


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


def validate_array_metadata_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v3 array document.

    Structural problems (from the model layer) and semantic problems
    (from the entities themselves) are reported together. JSON arrays are
    normalized to tuples before judgment, so list-spelled documents
    (e.g. fresh `json.loads` output) are judged at the canonical data
    level rather than rejected for their spelling.
    """
    return _judged(arrays_to_tuples(value), _validate_structure_v3, _array_semantics_v3)


def parse_array_metadata_v3(value: object) -> ZarrV3ArrayMetadataJSON:
    """Return `value` as a valid `ZarrV3ArrayMetadataJSON`, or raise.

    Normalizes JSON arrays to tuples, then raises a single
    `MetadataValidationError` carrying every structural and composition
    problem found.
    """
    normalized = arrays_to_tuples(value)
    problems = _judged(normalized, _validate_structure_v3, _array_semantics_v3)
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


def validate_group_metadata_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v3 group document.

    Composition rules recurse into inline consolidated metadata, so a
    consolidated child document invalid under its own rules is reported
    here, at its path.
    """
    return _judged(arrays_to_tuples(value), _validate_group_structure_v3, group_problems_v3)


def parse_group_metadata_v3(value: object) -> ZarrV3GroupMetadataJSON:
    """Return `value` as a valid `ZarrV3GroupMetadataJSON`, or raise."""
    normalized = arrays_to_tuples(value)
    problems = _judged(normalized, _validate_group_structure_v3, group_problems_v3)
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


__all__ = [
    "parse_array_metadata_v2",
    "parse_array_metadata_v3",
    "parse_group_metadata_v2",
    "parse_group_metadata_v3",
    "validate_array_metadata_v2",
    "validate_array_metadata_v3",
    "validate_group_metadata_v2",
    "validate_group_metadata_v3",
]
