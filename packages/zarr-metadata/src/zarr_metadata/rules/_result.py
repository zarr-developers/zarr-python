"""Tagged validation results.

`check_*` returns `Valid[T] | Invalid`. Testing the literal `valid`
field narrows to either the normalized document or a nonempty problem
tuple. Use `validate_*` to collect problems and `parse_*` to raise on
invalid input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from zarr_metadata.model._validation import ValidationProblem, arrays_to_tuples
from zarr_metadata.rules._documents import (
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_group_metadata_v2,
    validate_group_metadata_v3,
)
from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON  # noqa: TC001
from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON  # noqa: TC001
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON  # noqa: TC001
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON  # noqa: TC001

DocumentT = TypeVar("DocumentT")


@dataclass(frozen=True, slots=True)
class Valid(Generic[DocumentT]):
    """A document that passed structural and composition validation."""

    document: DocumentT
    valid: Literal[True] = True


@dataclass(frozen=True, slots=True)
class Invalid:
    """Every reason a document failed validation.

    `problems` is never empty: an empty report is a `Valid`.
    """

    problems: tuple[ValidationProblem, ...]
    valid: Literal[False] = False

    def __post_init__(self) -> None:
        if len(self.problems) == 0:
            msg = "Invalid requires at least one validation problem"
            raise ValueError(msg)


ValidationResult: TypeAlias = Valid[DocumentT] | Invalid
"""Either a validated document or the problems that disqualified it."""


def check_array_metadata_v3(value: object) -> ValidationResult[ZarrV3ArrayMetadataJSON]:
    """`value` as a valid v3 array document, or the problems disqualifying it.

    A `Valid` carries the normalized document (JSON arrays as tuples),
    exactly as `parse_array_metadata_v3` returns it.
    """
    problems = validate_array_metadata_v3(value)
    if len(problems) != 0:
        return Invalid(problems)
    return Valid(cast("ZarrV3ArrayMetadataJSON", arrays_to_tuples(value)))


def check_array_metadata_v2(value: object) -> ValidationResult[ZarrV2ArrayMetadataJSON]:
    """`value` as a valid v2 array document, or the problems disqualifying it."""
    problems = validate_array_metadata_v2(value)
    if len(problems) != 0:
        return Invalid(problems)
    return Valid(cast("ZarrV2ArrayMetadataJSON", arrays_to_tuples(value)))


def check_group_metadata_v3(value: object) -> ValidationResult[ZarrV3GroupMetadataJSON]:
    """`value` as a valid v3 group document, or the problems disqualifying it."""
    problems = validate_group_metadata_v3(value)
    if len(problems) != 0:
        return Invalid(problems)
    return Valid(cast("ZarrV3GroupMetadataJSON", arrays_to_tuples(value)))


def check_group_metadata_v2(value: object) -> ValidationResult[ZarrV2GroupMetadataJSON]:
    """`value` as a valid v2 group document, or the problems disqualifying it."""
    problems = validate_group_metadata_v2(value)
    if len(problems) != 0:
        return Invalid(problems)
    return Valid(cast("ZarrV2GroupMetadataJSON", arrays_to_tuples(value)))


__all__ = [
    "Invalid",
    "Valid",
    "ValidationResult",
    "check_array_metadata_v2",
    "check_array_metadata_v3",
    "check_group_metadata_v2",
    "check_group_metadata_v3",
]
