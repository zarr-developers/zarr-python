"""In-memory models for Zarr metadata documents.

Models are frozen dataclasses that hold a canonical, semantically lossless
representation of the JSON documents; they never interpret extension points
(codecs, chunk grids, data types). Validators check JSON structure, not domain validity.
Each document concept gets a `validate_*` function returning every problem
found (a `list[ValidationProblem]`, each with a machine-readable `kind`), an
`is_*` type guard, and a `parse_*` function that narrows or raises
`MetadataValidationError`. Model `from_json` / `from_key_value` constructors
raise `MetadataValidationError` for every ingestion failure, including
missing store keys and undecodable bytes.
"""

from zarr_metadata.model._array import (
    ZarrV2ArrayMetadata,
    ZarrV2ArrayMetadataPartial,
    ZarrV3ArrayMetadata,
    ZarrV3ArrayMetadataPartial,
    ZarrV3MetadataField,
    ZarrV3NamedConfig,
)
from zarr_metadata.model._group import (
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV2GroupMetadataPartial,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
    ZarrV3GroupMetadataPartial,
)
from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import (
    ARRAY_METADATA_OPTIONAL_KEYS_V3,
    ARRAY_METADATA_REQUIRED_KEYS_V2,
    ARRAY_METADATA_REQUIRED_KEYS_V3,
    ARRAY_METADATA_STANDARD_KEYS_V3,
    GROUP_METADATA_OPTIONAL_KEYS_V3,
    GROUP_METADATA_REQUIRED_KEYS_V2,
    GROUP_METADATA_REQUIRED_KEYS_V3,
    GROUP_METADATA_STANDARD_KEYS_V3,
    MetadataValidationError,
    ProblemKind,
    ValidationProblem,
    is_array_metadata_v2,
    is_array_metadata_v3,
    is_group_metadata_v2,
    is_group_metadata_v3,
    is_json,
    is_metadata_field_v3,
    parse_array_metadata_v2,
    parse_array_metadata_v3,
    parse_group_metadata_v2,
    parse_group_metadata_v3,
    parse_json,
    parse_metadata_field_v3,
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_group_metadata_v2,
    validate_group_metadata_v3,
    validate_json,
    validate_metadata_field_v3,
)

# Store keys are facts about the on-disk specs, so they are defined in the
# `v2`/`v3` modules that describe those documents. They are re-exported here
# because the model layer is where consumers reach for them.
from zarr_metadata.v2.array import (
    ZARR_V2_ARRAY_METADATA_STORE_KEY,
    ZarrV2ArrayMetadataStoreKey,
)
from zarr_metadata.v2.attributes import (
    ZARR_V2_ATTRIBUTES_STORE_KEY,
    ZarrV2AttributesStoreKey,
)
from zarr_metadata.v2.consolidated import (
    ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY,
    ZarrV2ConsolidatedMetadataStoreKey,
)
from zarr_metadata.v2.group import (
    ZARR_V2_GROUP_METADATA_STORE_KEY,
    ZarrV2GroupMetadataStoreKey,
)
from zarr_metadata.v3.array import (
    ZARR_V3_ARRAY_METADATA_STORE_KEY,
    ZarrV3ArrayMetadataStoreKey,
)
from zarr_metadata.v3.consolidated import ZARR_V3_CONSOLIDATED_METADATA_KEY
from zarr_metadata.v3.group import (
    ZARR_V3_GROUP_METADATA_STORE_KEY,
    ZarrV3GroupMetadataStoreKey,
)

__all__ = [
    "ARRAY_METADATA_OPTIONAL_KEYS_V3",
    "ARRAY_METADATA_REQUIRED_KEYS_V2",
    "ARRAY_METADATA_REQUIRED_KEYS_V3",
    "ARRAY_METADATA_STANDARD_KEYS_V3",
    "GROUP_METADATA_OPTIONAL_KEYS_V3",
    "GROUP_METADATA_REQUIRED_KEYS_V2",
    "GROUP_METADATA_REQUIRED_KEYS_V3",
    "GROUP_METADATA_STANDARD_KEYS_V3",
    "UNSET",
    "ZARR_V2_ARRAY_METADATA_STORE_KEY",
    "ZARR_V2_ATTRIBUTES_STORE_KEY",
    "ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY",
    "ZARR_V2_GROUP_METADATA_STORE_KEY",
    "ZARR_V3_ARRAY_METADATA_STORE_KEY",
    "ZARR_V3_CONSOLIDATED_METADATA_KEY",
    "ZARR_V3_GROUP_METADATA_STORE_KEY",
    "MetadataValidationError",
    "ProblemKind",
    "ValidationProblem",
    "ZarrV2ArrayMetadata",
    "ZarrV2ArrayMetadataPartial",
    "ZarrV2ArrayMetadataStoreKey",
    "ZarrV2AttributesStoreKey",
    "ZarrV2ConsolidatedMetadata",
    "ZarrV2ConsolidatedMetadataStoreKey",
    "ZarrV2GroupMetadata",
    "ZarrV2GroupMetadataPartial",
    "ZarrV2GroupMetadataStoreKey",
    "ZarrV3ArrayMetadata",
    "ZarrV3ArrayMetadataPartial",
    "ZarrV3ArrayMetadataStoreKey",
    "ZarrV3ConsolidatedMetadata",
    "ZarrV3GroupMetadata",
    "ZarrV3GroupMetadataPartial",
    "ZarrV3GroupMetadataStoreKey",
    "ZarrV3MetadataField",
    "ZarrV3NamedConfig",
    "is_array_metadata_v2",
    "is_array_metadata_v3",
    "is_group_metadata_v2",
    "is_group_metadata_v3",
    "is_json",
    "is_metadata_field_v3",
    "parse_array_metadata_v2",
    "parse_array_metadata_v3",
    "parse_group_metadata_v2",
    "parse_group_metadata_v3",
    "parse_json",
    "parse_metadata_field_v3",
    "validate_array_metadata_v2",
    "validate_array_metadata_v3",
    "validate_group_metadata_v2",
    "validate_group_metadata_v3",
    "validate_json",
    "validate_metadata_field_v3",
]
