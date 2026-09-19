"""Whole-document structural and composition validation.

These `validate_*`, `is_*`, and `parse_*` functions mirror the model API
but apply both validation layers. The `is_*` functions return `bool`, not
`TypeIs`: composition validity is stricter than TypedDict membership.
Use `zarr_metadata.model.is_*` for type narrowing.
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
from zarr_metadata.rules._v3_array import ZARR_V3_ARRAY_RULES
from zarr_metadata.rules._v3_group import ZARR_V3_GROUP_RULES

if TYPE_CHECKING:
    from zarr_metadata.model._validation import ValidationProblem
    from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON
    from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
    from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON


def validate_array_metadata_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v3 array document.

    Structural problems (from the model layer) and composition problems
    (from `ZARR_V3_ARRAY_RULES`) are reported together. JSON arrays are
    normalized to tuples before judgment, so list-spelled documents
    (e.g. fresh `json.loads` output) are judged at the canonical data
    level rather than rejected for their spelling.
    """
    normalized = arrays_to_tuples(value)
    problems = _validate_structure_v3(normalized)
    if isinstance(normalized, Mapping):
        document = cast("Mapping[str, object]", normalized)
        problems = problems + run_rules(ZARR_V3_ARRAY_RULES, document)
    return tuple(problems)


def is_array_metadata_v3(value: object) -> bool:
    """Whether `value` is a structurally and compositionally valid v3 array doc.

    Deliberately not a `TypeIs` guard — see the module docstring. Use
    `zarr_metadata.model.is_array_metadata_v3` to narrow.
    """
    return len(validate_array_metadata_v3(value)) == 0


def parse_array_metadata_v3(value: object) -> ZarrV3ArrayMetadataJSON:
    """Return `value` as a valid `ZarrV3ArrayMetadataJSON`, or raise.

    Normalizes JSON arrays to tuples, then raises a single
    `MetadataValidationError` carrying every structural and composition
    problem found.
    """
    normalized = arrays_to_tuples(value)
    problems = validate_array_metadata_v3(normalized)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3ArrayMetadataJSON", normalized)


def validate_array_metadata_v2(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v2 array document (merged form).

    JSON arrays are normalized to tuples before judgment, as in
    `validate_array_metadata_v3`.
    """
    normalized = arrays_to_tuples(value)
    problems = _validate_structure_v2(normalized)
    if isinstance(normalized, Mapping):
        document = cast("Mapping[str, object]", normalized)
        problems = problems + run_rules(ZARR_V2_ARRAY_RULES, document)
    return tuple(problems)


def is_array_metadata_v2(value: object) -> bool:
    """Whether `value` is a structurally and compositionally valid v2 array doc.

    Deliberately not a `TypeIs` guard — see the module docstring.
    """
    return len(validate_array_metadata_v2(value)) == 0


def parse_array_metadata_v2(value: object) -> ZarrV2ArrayMetadataJSON:
    """Return `value` as a valid `ZarrV2ArrayMetadataJSON`, or raise.

    Normalizes JSON arrays to tuples, then raises a single
    `MetadataValidationError` carrying every structural and composition
    problem found.
    """
    normalized = arrays_to_tuples(value)
    problems = validate_array_metadata_v2(normalized)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV2ArrayMetadataJSON", normalized)


def validate_group_metadata_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v3 group document.

    Composition rules recurse into inline consolidated metadata, so a
    consolidated child document invalid under its own rules is reported
    here, at its path.
    """
    normalized = arrays_to_tuples(value)
    problems = _validate_group_structure_v3(normalized)
    if isinstance(normalized, Mapping):
        document = cast("Mapping[str, object]", normalized)
        problems = problems + run_rules(ZARR_V3_GROUP_RULES, document)
    return tuple(problems)


def is_group_metadata_v3(value: object) -> bool:
    """Whether `value` is a structurally and compositionally valid v3 group doc.

    Deliberately not a `TypeIs` guard — see the module docstring.
    """
    return len(validate_group_metadata_v3(value)) == 0


def parse_group_metadata_v3(value: object) -> ZarrV3GroupMetadataJSON:
    """Return `value` as a valid `ZarrV3GroupMetadataJSON`, or raise."""
    normalized = arrays_to_tuples(value)
    problems = validate_group_metadata_v3(normalized)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3GroupMetadataJSON", normalized)


def validate_group_metadata_v2(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v2 group document (merged form).

    v2 group documents carry no composition constraints today, so this is
    the structural judgment, offered here for a uniform read-side API.
    """
    return _validate_group_structure_v2(arrays_to_tuples(value))


def is_group_metadata_v2(value: object) -> bool:
    """Whether `value` is a valid v2 group document (merged form)."""
    return len(validate_group_metadata_v2(value)) == 0


def parse_group_metadata_v2(value: object) -> ZarrV2GroupMetadataJSON:
    """Return `value` as a valid `ZarrV2GroupMetadataJSON`, or raise."""
    normalized = arrays_to_tuples(value)
    problems = validate_group_metadata_v2(normalized)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV2GroupMetadataJSON", normalized)


__all__ = [
    "is_array_metadata_v2",
    "is_array_metadata_v3",
    "is_group_metadata_v2",
    "is_group_metadata_v3",
    "parse_array_metadata_v2",
    "parse_array_metadata_v3",
    "parse_group_metadata_v2",
    "parse_group_metadata_v3",
    "validate_array_metadata_v2",
    "validate_array_metadata_v3",
    "validate_group_metadata_v2",
    "validate_group_metadata_v3",
]
