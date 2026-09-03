"""Validate structure and composition of Zarr metadata documents.

`zarr_metadata.model` checks JSON structure. This module also checks
cross-field constraints such as fill-value compatibility, codec
ordering, and dimension counts. Its `validate_*`, `is_*`, and `parse_*`
functions mirror the model API; `check_*` returns `Valid[T] | Invalid`.

Rules target canonical metadata and may be stricter than readers that
coerce inputs. Unknown entity names are left unjudged. Known entities
must match their modeled shape; extra configuration keys produce an
`unknown_key` problem without suppressing other checks. Model
round-trips preserve those unmodeled members.
"""

from zarr_metadata.rules._documents import (
    is_array_metadata_v2,
    is_array_metadata_v3,
    is_group_metadata_v2,
    is_group_metadata_v3,
    parse_array_metadata_v2,
    parse_array_metadata_v3,
    parse_group_metadata_v2,
    parse_group_metadata_v3,
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_group_metadata_v2,
    validate_group_metadata_v3,
)
from zarr_metadata.rules._engine import Rule, RuleCheck, applicable, run_rules
from zarr_metadata.rules._result import (
    Invalid,
    Valid,
    ValidationResult,
    check_array_metadata_v2,
    check_array_metadata_v3,
    check_group_metadata_v2,
    check_group_metadata_v3,
)
from zarr_metadata.rules._v2_array import ZARR_V2_ARRAY, ZARR_V2_ARRAY_RULES
from zarr_metadata.rules._v3_array import ZARR_V3_ARRAY, ZARR_V3_ARRAY_RULES
from zarr_metadata.rules._v3_group import ZARR_V3_GROUP, ZARR_V3_GROUP_RULES

__all__ = [
    "ZARR_V2_ARRAY",
    "ZARR_V2_ARRAY_RULES",
    "ZARR_V3_ARRAY",
    "ZARR_V3_ARRAY_RULES",
    "ZARR_V3_GROUP",
    "ZARR_V3_GROUP_RULES",
    "Invalid",
    "Rule",
    "RuleCheck",
    "Valid",
    "ValidationResult",
    "applicable",
    "check_array_metadata_v2",
    "check_array_metadata_v3",
    "check_group_metadata_v2",
    "check_group_metadata_v3",
    "is_array_metadata_v2",
    "is_array_metadata_v3",
    "is_group_metadata_v2",
    "is_group_metadata_v3",
    "parse_array_metadata_v2",
    "parse_array_metadata_v3",
    "parse_group_metadata_v2",
    "parse_group_metadata_v3",
    "run_rules",
    "validate_array_metadata_v2",
    "validate_array_metadata_v3",
    "validate_group_metadata_v2",
    "validate_group_metadata_v3",
]
