"""
Low-level functional API for zarr hierarchies, backed by the Rust
[`zarrs`](https://zarrs.dev) crate.

This subpackage is experimental. It requires the `zarrs-bindings` package
(in-repo Rust crate; install for development with `uv sync --group zarrs`).

All array routines take an explicit metadata document (a `dict` matching the
`zarr.json` / `.zarray` document) rather than reading metadata from the store,
which makes read-only and virtual views possible.
"""

try:
    import _zarrs_bindings
except ImportError as e:
    raise ImportError(
        "zarr.zarrs requires the `zarrs-bindings` package, which is not installed. "
        "It is built from the zarr-python repository: run `uv sync --group zarrs`."
    ) from e

__version__: str = _zarrs_bindings.version()

from zarr.zarrs._api import (
    NodeExistsError,
    ZarrsOptions,
    create_new_array,
    create_new_group,
    create_overwrite_array,
    create_overwrite_group,
    delete_node,
    list_children,
    read_metadata,
)

__all__ = [
    "NodeExistsError",
    "ZarrsOptions",
    "__version__",
    "create_new_array",
    "create_new_group",
    "create_overwrite_array",
    "create_overwrite_group",
    "delete_node",
    "list_children",
    "read_metadata",
]
