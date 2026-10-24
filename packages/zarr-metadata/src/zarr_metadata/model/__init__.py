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
function that narrows the value or raises `MetadataValidationError`.
Every document type also gets a reader (`read_array_metadata_v3`,
`read_group_metadata_v3`, `read_consolidated_metadata_v3`,
`read_array_metadata_v2`, `read_group_metadata_v2`,
`read_consolidated_metadata_v2`). Each does
one read and returns what it read, the problems, and the model when there
are no problems. It is the form of the model's constructor that returns
problems as values instead of raising them. A node of either type is read by `read_node_metadata_v3`, from
a `zarr.json` and its `node_type`, or by `read_node_metadata_v2`, from
the node's decoded `.zarray` or `.zgroup` and `.zattrs`, which can come
from a store or from a `.zmetadata`. To read
a store that another writer made with a known writer bug, use
`read_repaired_node_metadata_v3` or `read_repaired_node_metadata_v2`.
They undo each bug with `repair_node_metadata_v3` or
`repair_node_metadata_v2` before the strict read, and report what they
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
the model as it is. A v3 group reads its `consolidated_metadata` as an
envelope and does not read the documents it lists, because consolidated
metadata is a convention of the reference implementation, not core
metadata. `ZarrV3ConsolidatedMetadata` and
`read_consolidated_metadata_v3` read those documents. The model also
accepts node models as entries: an entry is accepted when its claims
refine into the model's scope, and refused with a problem at its path
otherwise. Every reader takes `context=None` for the default scope.
"""

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
    ZarrV2ConsolidatedMetadataReading,
    ZarrV2GroupMetadata,
    ZarrV2GroupMetadataReading,
    ZarrV2GroupMetadataUpdate,
    ZarrV2NodeMetadata,
    ZarrV2NodeMetadataReading,
    ZarrV2UnknownNodeMetadataReading,
    is_group_metadata_v2,
    node_metadata_from_key_value_v2,
    parse_group_metadata_v2,
    read_consolidated_metadata_v2,
    read_group_metadata_v2,
    read_node_metadata_v2,
    validate_group_metadata_v2,
    validate_node_metadata_v2,
)
from zarr_metadata.model._group_v3 import (
    ZarrV3ConsolidatedMetadata,
    ZarrV3ConsolidatedMetadataInput,
    ZarrV3ConsolidatedMetadataReading,
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
    read_consolidated_metadata_v3,
    read_group_metadata_v3,
    read_node_metadata_v3,
    validate_group_metadata_v3,
    validate_node_metadata_v3,
)
from zarr_metadata.model._repair import (
    Repair,
    RepairKind,
    ZarrV2RepairedConsolidatedMetadataReading,
    ZarrV2RepairedNodeMetadataReading,
    ZarrV3RepairedNodeMetadataReading,
    read_repaired_consolidated_metadata_v2,
    read_repaired_node_metadata_v2,
    read_repaired_node_metadata_v3,
    repair_consolidated_metadata_v2,
    repair_node_metadata_v2,
    repair_node_metadata_v3,
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

__all__ = [
    "UNSET",
    "Repair",
    "RepairKind",
    "ZarrV2ArrayMetadata",
    "ZarrV2ArrayMetadataReading",
    "ZarrV2ArrayMetadataUpdate",
    "ZarrV2ConsolidatedMetadata",
    "ZarrV2ConsolidatedMetadataReading",
    "ZarrV2GroupMetadata",
    "ZarrV2GroupMetadataReading",
    "ZarrV2GroupMetadataUpdate",
    "ZarrV2NodeMetadata",
    "ZarrV2NodeMetadataReading",
    "ZarrV2RepairedConsolidatedMetadataReading",
    "ZarrV2RepairedNodeMetadataReading",
    "ZarrV2UnknownNodeMetadataReading",
    "ZarrV3ArrayMetadata",
    "ZarrV3ArrayMetadataReading",
    "ZarrV3ArrayMetadataUpdate",
    "ZarrV3ConsolidatedMetadata",
    "ZarrV3ConsolidatedMetadataInput",
    "ZarrV3ConsolidatedMetadataReading",
    "ZarrV3GroupMetadata",
    "ZarrV3GroupMetadataReading",
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
    "node_metadata_from_key_value_v2",
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
    "read_consolidated_metadata_v2",
    "read_consolidated_metadata_v3",
    "read_group_metadata_v2",
    "read_group_metadata_v3",
    "read_node_metadata_v2",
    "read_node_metadata_v3",
    "read_repaired_consolidated_metadata_v2",
    "read_repaired_node_metadata_v2",
    "read_repaired_node_metadata_v3",
    "repair_consolidated_metadata_v2",
    "repair_node_metadata_v2",
    "repair_node_metadata_v3",
    "validate_array_metadata_v2",
    "validate_array_metadata_v3",
    "validate_group_metadata_v2",
    "validate_group_metadata_v3",
    "validate_metadata_field_v3",
    "validate_node_metadata_v2",
    "validate_node_metadata_v3",
    "validate_node_name_v3",
    "validate_node_path_v3",
]
