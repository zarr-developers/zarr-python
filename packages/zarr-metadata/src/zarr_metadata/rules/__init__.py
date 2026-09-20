"""Validate structure and composition of Zarr metadata documents.

`zarr_metadata.model` checks JSON structure. This module also checks
cross-field constraints such as fill-value compatibility, codec
ordering, and dimension counts. Its `validate_*` and `parse_*` functions
mirror the model API, and `canonicalize_array_metadata_v3` answers with
either the document in its simplest equivalent spelling or every reason
it is not valid.

Rules target canonical metadata and may be stricter than readers that
coerce inputs. Unknown entity names are left unjudged. Known entities
must match their modeled shape; extra configuration keys produce an
`unknown_key` problem without suppressing other checks. Model
round-trips preserve those unmodeled members.
"""

from zarr_metadata.rules._canonical import (
    Canonical,
    Invalid,
    canonicalize_array_metadata_v3,
)
from zarr_metadata.rules._documents import (
    parse_array_metadata_v2,
    parse_array_metadata_v3,
    parse_group_metadata_v2,
    parse_group_metadata_v3,
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_group_metadata_v2,
    validate_group_metadata_v3,
)

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
