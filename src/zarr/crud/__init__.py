"""
Backend-agnostic low-level functional CRUD API for zarr hierarchies.

The public functions delegate byte- and metadata-level work to a `CrudBackend`
implementation. Ship with a pure-Python reference backend (the default), a
zarrs-accelerated backend (`zarr.zarrs`, requires the `zarrs-bindings`
extension), and a zarrista backend (`zarr.zarrista`).

Naming: you select an execution path by **engine** — a name passed as `engine=`
(here and on the top-level `create_array` / `open_array`) or via the
`array.engine` config key. The engine namespace is `{"zarr"}` (the native path)
plus every registered **backend**, where "backend" is the implementation object
(`CrudBackend`) that a non-`"zarr"` engine resolves to. So: engine = the name
you pick; backend = the object it resolves to. Use `list_backends()` to
enumerate the non-native engine names.

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
    write_region,
)
from zarr.crud._backend import CrudBackend, NodeExistsError
from zarr.crud._reference import ReferenceBackend
from zarr.crud._registry import get_backend, list_backends, register_backend

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
    "list_backends",
    "list_children",
    "read_chunk",
    "read_encoded_chunk",
    "read_metadata",
    "read_region",
    "register_backend",
    "write_chunk",
    "write_region",
]
