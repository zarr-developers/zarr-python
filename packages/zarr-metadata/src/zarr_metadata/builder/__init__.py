"""Validated construction of Zarr metadata documents.

`create_*` factories provide typed one-shot construction for every
document TypedDict: required keys and value types are checked statically
at literal-keyword call sites, and the runtime pass normalizes the input
and applies structural and composition validation, raising one
`MetadataValidationError` carrying every problem.

Use `zarr_metadata.rules` to validate documents read from storage.
"""

from zarr_metadata.builder._create import (
    create_zarr_v2_array_metadata_json,
    create_zarr_v2_consolidated_metadata_json,
    create_zarr_v2_group_metadata_json,
    create_zarr_v2_zarray_json,
    create_zarr_v2_zgroup_json,
    create_zarr_v3_array_metadata_json,
    create_zarr_v3_consolidated_metadata_json,
    create_zarr_v3_group_metadata_json,
)

__all__ = [
    "create_zarr_v2_array_metadata_json",
    "create_zarr_v2_consolidated_metadata_json",
    "create_zarr_v2_group_metadata_json",
    "create_zarr_v2_zarray_json",
    "create_zarr_v2_zgroup_json",
    "create_zarr_v3_array_metadata_json",
    "create_zarr_v3_consolidated_metadata_json",
    "create_zarr_v3_group_metadata_json",
]
