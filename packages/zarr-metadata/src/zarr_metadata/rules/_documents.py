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
from zarr_metadata.rules._engine import run_rules
from zarr_metadata.rules._v2_array import ZARR_V2_ARRAY_RULES
from zarr_metadata.rules._v3_group import ZARR_V3_GROUP_RULES
from zarr_metadata.v3._document import array_problems_v3
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from zarr_metadata.model._validation import ValidationProblem
    from zarr_metadata.rules._engine import Rule
    from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON
    from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
    from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON

    _StructuralValidator = Callable[[object], tuple[ValidationProblem, ...]]


def _judged_array_v3(normalized: object) -> tuple[ValidationProblem, ...]:
    """Structural and semantic problems in a v3 array document.

    The semantic half is `zarr_metadata.v3` asking each entity about
    itself and about the parts of the document it meets; this layer
    chooses the scope those questions are asked in.
    """
    problems = _validate_structure_v3(normalized)
    if isinstance(normalized, Mapping):
        problems = problems + array_problems_v3(
            cast("Mapping[str, object]", normalized), CORE_AND_EXTENSIONS
        )
    return tuple(problems)


def _judged(
    normalized: object, structure: _StructuralValidator, rules: Sequence[Rule]
) -> tuple[ValidationProblem, ...]:
    """Structural and composition problems in an already-normalized document.

    Takes the normalized value rather than the caller's input so that
    `validate_*` and `parse_*` each walk the document once.
    """
    problems = structure(normalized)
    if isinstance(normalized, Mapping):
        problems = problems + run_rules(rules, cast("Mapping[str, object]", normalized))
    return tuple(problems)


def validate_array_metadata_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v3 array document.

    Structural problems (from the model layer) and semantic problems
    (from the entities themselves) are reported together. JSON arrays are
    normalized to tuples before judgment, so list-spelled documents
    (e.g. fresh `json.loads` output) are judged at the canonical data
    level rather than rejected for their spelling.
    """
    return _judged_array_v3(arrays_to_tuples(value))


def parse_array_metadata_v3(value: object) -> ZarrV3ArrayMetadataJSON:
    """Return `value` as a valid `ZarrV3ArrayMetadataJSON`, or raise.

    Normalizes JSON arrays to tuples, then raises a single
    `MetadataValidationError` carrying every structural and composition
    problem found.
    """
    normalized = arrays_to_tuples(value)
    problems = _judged_array_v3(normalized)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3ArrayMetadataJSON", normalized)


def validate_array_metadata_v2(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v2 array document (merged form).

    JSON arrays are normalized to tuples before judgment, as in
    `validate_array_metadata_v3`.
    """
    return _judged(arrays_to_tuples(value), _validate_structure_v2, ZARR_V2_ARRAY_RULES)


def parse_array_metadata_v2(value: object) -> ZarrV2ArrayMetadataJSON:
    """Return `value` as a valid `ZarrV2ArrayMetadataJSON`, or raise.

    Normalizes JSON arrays to tuples, then raises a single
    `MetadataValidationError` carrying every structural and composition
    problem found.
    """
    normalized = arrays_to_tuples(value)
    problems = _judged(normalized, _validate_structure_v2, ZARR_V2_ARRAY_RULES)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV2ArrayMetadataJSON", normalized)


def validate_group_metadata_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v3 group document.

    Composition rules recurse into inline consolidated metadata, so a
    consolidated child document invalid under its own rules is reported
    here, at its path.
    """
    return _judged(arrays_to_tuples(value), _validate_group_structure_v3, ZARR_V3_GROUP_RULES)


def parse_group_metadata_v3(value: object) -> ZarrV3GroupMetadataJSON:
    """Return `value` as a valid `ZarrV3GroupMetadataJSON`, or raise."""
    normalized = arrays_to_tuples(value)
    problems = _judged(normalized, _validate_group_structure_v3, ZARR_V3_GROUP_RULES)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3GroupMetadataJSON", normalized)


def validate_group_metadata_v2(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v2 group document (merged form).

    v2 group documents carry no composition constraints today, so this is
    the structural judgment, offered here for a uniform read-side API.
    """
    return _judged(arrays_to_tuples(value), _validate_group_structure_v2, ())


def parse_group_metadata_v2(value: object) -> ZarrV2GroupMetadataJSON:
    """Return `value` as a valid `ZarrV2GroupMetadataJSON`, or raise."""
    normalized = arrays_to_tuples(value)
    problems = _judged(normalized, _validate_group_structure_v2, ())
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
