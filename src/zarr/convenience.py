import warnings

from zarr.api.synchronous import (
    consolidate_metadata,
    copy,
    copy_all,
    copy_store,
    load,
    open,
    open_consolidated,
    save,
    save_array,
    save_group,
    tree,
)

__all__ = [
    "open",
    "save",
    "load",
    "save_array",
    "save_group",
    "copy",
    "copy_all",
    "copy_store",
    "tree",
    "consolidate_metadata",
    "open_consolidated",
]

warnings.warn(
    "zarr.convenience is deprecated, use zarr.api.synchronous",
    DeprecationWarning,
    stacklevel=2,
)
