"""In-memory models for Zarr metadata documents.

A model is a metadata document, as written and refined, and the scope it
was read in; what it hands out is read-only. Validators check a document's JSON
structure and, in a v3 document, read each extension point (codecs, chunk
grids, data types, ...) through the definition that claims its name in a
scope, `CORE_AND_EXTENSIONS` unless a `context` is passed, and judge the
fill value against the data type it names, the chunk grid against the
shape, and the codecs as a pipeline, each against the chunk it is
handed. Each document concept gets a `validate_*` function returning
every problem found (a tuple of `ValidationProblem`, each with a
machine-readable `kind`), an `is_*` type guard, and a `parse_*` function
that narrows or raises `MetadataValidationError`; a v3 array or group
document also gets `read_array_metadata_v3`, `read_group_metadata_v3` or
`read_array_metadata_v2`,
one read that returns what it read, the problems, and the model when
there are none. A store another writer made, holding a known writer
bug, is read by `read_repaired_node_metadata_v3`, which undoes each one
with `repair_node_metadata_v3` before the strict read and says what it
changed. `node_metadata_json_schema_v3` writes what the v3
validators read as a JSON Schema, but for the rules. Model `from_json` /
`from_key_value` constructors raise
`MetadataValidationError` for every ingestion failure, including missing
store keys and undecodable bytes, and the v3 ones take the same
`context`. A v3 model is its document and the scope it was read in:
`ZarrV3ArrayMetadata(document, context=None)` reads the document in the
scope and raises `MetadataValidationError` with every problem, so no
model is built invalid; `to_json` writes the document as written;
`update` reads new members in the model's own scope; `with_context` and
`refined_in` read the document in another; `to_key_value` writes a model
as it is. A group's `consolidated_metadata` takes node models as entries,
each accepted when its claims refine into the group's scope and refused
at its path otherwise. Every reader takes `context=None` for the default
scope.
"""

from zarr_metadata._json import (
    MetadataValidationError,
    ProblemKind,
    ValidationProblem,
)
from zarr_metadata._sentinel import UNSET
from zarr_metadata.model._array import (
    ZarrV2ArrayMetadata,
    ZarrV2ArrayMetadataUpdate,
    ZarrV3ArrayMetadata,
    ZarrV3ArrayMetadataUpdate,
    read_array_metadata_v2,
    read_array_metadata_v3,
)
from zarr_metadata.model._group import (
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV2GroupMetadataUpdate,
    ZarrV2NodeMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3ConsolidatedMetadataInput,
    ZarrV3GroupMetadata,
    ZarrV3GroupMetadataReading,
    ZarrV3GroupMetadataUpdate,
    ZarrV3NodeMetadata,
    ZarrV3NodeMetadataInput,
    ZarrV3NodeMetadataReading,
    ZarrV3UnknownNodeReading,
    is_group_metadata_v3,
    node_metadata_from_json_v3,
    node_metadata_from_key_value_v3,
    parse_group_metadata_v3,
    read_group_metadata_v3,
    read_node_metadata_v3,
    validate_group_metadata_v3,
    validate_node_metadata_v3,
)
from zarr_metadata.model._json_schema import node_metadata_json_schema_v3
from zarr_metadata.model._repair import (
    Repair,
    RepairKind,
    ZarrV2RepairedConsolidatedMetadataReading,
    ZarrV3RepairedNodeMetadataReading,
    read_repaired_consolidated_metadata_v2,
    read_repaired_node_metadata_v3,
    repair_consolidated_metadata_v2,
    repair_node_metadata_v3,
)
from zarr_metadata.model._validation import (
    ZarrV2ArrayMetadataReading,
    ZarrV3ArrayMetadataReading,
    is_array_metadata_v2,
    is_array_metadata_v3,
    is_group_metadata_v2,
    parse_array_metadata_v2,
    parse_array_metadata_v3,
    parse_group_metadata_v2,
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_group_metadata_v2,
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
from zarr_metadata.v3._common import (
    is_metadata_field_v3,
    parse_metadata_field_v3,
    validate_metadata_field_v3,
)
from zarr_metadata.v3._hierarchy import (
    is_node_name_v3,
    is_node_path_v3,
    parse_node_name_v3,
    parse_node_path_v3,
    validate_node_name_v3,
    validate_node_path_v3,
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
    "Repair",
    "RepairKind",
    "ValidationProblem",
    "ZarrV2ArrayMetadata",
    "ZarrV2ArrayMetadataReading",
    "ZarrV2ArrayMetadataStoreKey",
    "ZarrV2ArrayMetadataUpdate",
    "ZarrV2AttributesStoreKey",
    "ZarrV2ConsolidatedMetadata",
    "ZarrV2ConsolidatedMetadataStoreKey",
    "ZarrV2GroupMetadata",
    "ZarrV2GroupMetadataStoreKey",
    "ZarrV2GroupMetadataUpdate",
    "ZarrV2NodeMetadata",
    "ZarrV2RepairedConsolidatedMetadataReading",
    "ZarrV3ArrayMetadata",
    "ZarrV3ArrayMetadataReading",
    "ZarrV3ArrayMetadataStoreKey",
    "ZarrV3ArrayMetadataUpdate",
    "ZarrV3ConsolidatedMetadata",
    "ZarrV3ConsolidatedMetadataInput",
    "ZarrV3GroupMetadata",
    "ZarrV3GroupMetadataReading",
    "ZarrV3GroupMetadataStoreKey",
    "ZarrV3GroupMetadataUpdate",
    "ZarrV3NodeMetadata",
    "ZarrV3NodeMetadataInput",
    "ZarrV3NodeMetadataReading",
    "ZarrV3RepairedNodeMetadataReading",
    "ZarrV3UnknownNodeReading",
    "is_array_metadata_v2",
    "is_array_metadata_v3",
    "is_group_metadata_v2",
    "is_group_metadata_v3",
    "is_metadata_field_v3",
    "is_node_name_v3",
    "is_node_path_v3",
    "node_metadata_from_json_v3",
    "node_metadata_from_key_value_v3",
    "node_metadata_json_schema_v3",
    "parse_array_metadata_v2",
    "parse_array_metadata_v3",
    "parse_group_metadata_v2",
    "parse_group_metadata_v3",
    "parse_metadata_field_v3",
    "parse_node_name_v3",
    "parse_node_path_v3",
    "read_array_metadata_v2",
    "read_array_metadata_v3",
    "read_group_metadata_v3",
    "read_node_metadata_v3",
    "read_repaired_consolidated_metadata_v2",
    "read_repaired_node_metadata_v3",
    "repair_consolidated_metadata_v2",
    "repair_node_metadata_v3",
    "validate_array_metadata_v2",
    "validate_array_metadata_v3",
    "validate_group_metadata_v2",
    "validate_group_metadata_v3",
    "validate_metadata_field_v3",
    "validate_node_metadata_v3",
    "validate_node_name_v3",
    "validate_node_path_v3",
]
