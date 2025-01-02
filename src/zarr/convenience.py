"""
Convenience helpers.

.. warning::

    This sub-module is deprecated. All functions here are defined
    in the top level zarr namespace instead.
"""

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
    "zarr.convenience is deprecated. "
    "Import these functions from the top level zarr. namespace instead.",
    DeprecationWarning,
    stacklevel=2,
)
