"""
Backend-agnostic low-level functional CRUD API for zarr hierarchies.

The public functions delegate byte- and metadata-level work to a `CrudBackend`.
Two backends ship: a pure-Python reference backend (the default) and a
zarrs-accelerated backend (`zarr.zarrs`, requires the `zarrs-bindings`
extension). Select one with the `crud.backend` config key or a per-call
`backend=` argument.

Array routines take an explicit metadata document (a `dict` matching the
`zarr.json` / `.zarray` document) rather than reading it from the store, which
makes read-only and virtual views possible.
"""

from zarr.crud._api import (
    CrudOptions,
    create_new_array,
    create_new_group,
    create_overwrite_array,
    create_overwrite_group,
    delete_chunk,
    delete_node,
    list_children,
    read_chunk,
    read_encoded_chunk,
    read_metadata,
    read_region,
    write_chunk,
)
from zarr.crud._backend import CrudBackend, NodeExistsError
from zarr.crud._reference import ReferenceBackend
from zarr.crud._registry import get_backend, register_backend

register_backend("reference", ReferenceBackend())

__all__ = [
    "CrudBackend",
    "CrudOptions",
    "NodeExistsError",
    "ReferenceBackend",
    "create_new_array",
    "create_new_group",
    "create_overwrite_array",
    "create_overwrite_group",
    "delete_chunk",
    "delete_node",
    "get_backend",
    "list_children",
    "read_chunk",
    "read_encoded_chunk",
    "read_metadata",
    "read_region",
    "register_backend",
    "write_chunk",
]
