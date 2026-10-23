"""In-memory models for Zarr metadata documents.

A model holds a metadata document (as written, then refined) and the
scope the document was read in. Everything a model returns is read-only.
Validators check a document's JSON structure. In a v3 document they
also read each extension point (codecs, chunk grids, data types, ...)
through the definition that claims its name in a scope. The scope is
`CORE_AND_EXTENSIONS` unless a `context` is passed. The validators check
the fill value against the data type the document names, the chunk grid
against the shape, and the codecs as a pipeline, each codec against the
chunk it receives. Each document concept gets a `validate_*` function
that returns every problem found (a tuple of `ValidationProblem`, each
with a machine-readable `kind`), an `is_*` type guard, and a `parse_*`
function that narrows the value or raises `MetadataValidationError`. A
v3 array or group document and a v2 array document also get
`read_array_metadata_v3`, `read_group_metadata_v3` or
`read_array_metadata_v2`. Each of these does one read and returns what
it read, the problems, and the model when there are no problems. To read
a store that another writer made with a known writer bug, use
`read_repaired_node_metadata_v3`. It undoes each bug with
`repair_node_metadata_v3` before the strict read, and reports what it
changed. `zarr_metadata.json_schema` writes a JSON Schema of what the v3
validators read, except for the rules. The model constructors
`from_json` and `from_key_value` raise `MetadataValidationError` for
every ingestion failure, including missing store keys and undecodable
bytes. The v3 constructors take the same `context`. A v3 model is its
document plus the scope it was read in.
`ZarrV3ArrayMetadata(document, context=None)` reads the document in the
scope and raises `MetadataValidationError` with every problem, so an
invalid model is never built. `to_json` returns the document as written.
`update` reads new members in the model's own scope. `with_context` and
`refined_in` read the document in another scope. `to_key_value` writes
the model as it is. A group's `consolidated_metadata` accepts node
models as entries. An entry is accepted when its claims refine into the
group's scope, and refused with a problem at its path otherwise. Every
reader takes `context=None` for the default scope.
"""

from zarr_metadata._json import (
    MetadataValidationError,
    ProblemKind,
    ValidationProblem,
)
from zarr_metadata._sentinel import UNSET
from zarr_metadata.model._array_v2 import (
    ZarrV2ArrayMetadata,
    ZarrV2ArrayMetadataReading,
    ZarrV2ArrayMetadataUpdate,
    is_array_metadata_v2,
    parse_array_metadata_v2,
    read_array_metadata_v2,
    validate_array_metadata_v2,
)
from zarr_metadata.model._array_v3 import (
    ZarrV3ArrayMetadata,
    ZarrV3ArrayMetadataReading,
    ZarrV3ArrayMetadataUpdate,
    is_array_metadata_v3,
    parse_array_metadata_v3,
    read_array_metadata_v3,
    validate_array_metadata_v3,
)
from zarr_metadata.model._group_v2 import (
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV2GroupMetadataUpdate,
    ZarrV2NodeMetadata,
    is_group_metadata_v2,
    parse_group_metadata_v2,
    validate_group_metadata_v2,
)
from zarr_metadata.model._group_v3 import (
    ZarrV3ConsolidatedMetadata,
    ZarrV3ConsolidatedMetadataInput,
    ZarrV3GroupMetadata,
    ZarrV3GroupMetadataReading,
    ZarrV3GroupMetadataUpdate,
    ZarrV3NodeMetadata,
    ZarrV3NodeMetadataInput,
    ZarrV3NodeMetadataReading,
    ZarrV3UnknownNodeMetadataReading,
    is_group_metadata_v3,
    node_metadata_from_json_v3,
    node_metadata_from_key_value_v3,
    parse_group_metadata_v3,
    read_group_metadata_v3,
    read_node_metadata_v3,
    validate_group_metadata_v3,
    validate_node_metadata_v3,
)
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
    "ZarrV3UnknownNodeMetadataReading",
    "is_array_metadata_v2",
    "is_array_metadata_v3",
    "is_group_metadata_v2",
    "is_group_metadata_v3",
    "is_metadata_field_v3",
    "is_node_name_v3",
    "is_node_path_v3",
    "node_metadata_from_json_v3",
    "node_metadata_from_key_value_v3",
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
