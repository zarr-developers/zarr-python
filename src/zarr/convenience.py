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
    "consolidate_metadata",
    "copy",
    "copy_all",
    "copy_store",
    "load",
    "open",
    "open_consolidated",
    "save",
    "save_array",
    "save_group",
    "tree",
]

warnings.warn(
    "zarr.convenience is deprecated, use zarr.api.synchronous",
    DeprecationWarning,
    stacklevel=2,
)
