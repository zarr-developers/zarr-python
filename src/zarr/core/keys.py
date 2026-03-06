"""Utilities for determining the set of valid store keys for zarr nodes.

A zarr node (array or group) implicitly defines a subset of keys in the
underlying store.  For an **array** the valid keys are:

* metadata documents (``zarr.json`` for v3, ``.zarray`` / ``.zattrs`` for v2)
* chunk (or shard) keys whose decoded coordinates fall within the storage grid

For a **group** the valid keys are:

* its own metadata documents
* any path ``<child>/<subkey>`` where ``<child>`` is a direct member and
  ``<subkey>`` is recursively valid for that child
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, ZMETADATA_V2_JSON

if TYPE_CHECKING:
    from zarr.core.array import Array
    from zarr.core.group import Group

_METADATA_KEYS_V3 = frozenset({ZARR_JSON})
_METADATA_KEYS_V2 = frozenset({ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, ZMETADATA_V2_JSON})


def metadata_keys(zarr_format: int) -> frozenset[str]:
    """Return the set of metadata key basenames for a given zarr format version.

    Parameters
    ----------
    zarr_format : int
        The zarr format version (2 or 3).

    Returns
    -------
    frozenset of str
    """
    if zarr_format == 3:
        return _METADATA_KEYS_V3
    return _METADATA_KEYS_V2


def decode_chunk_key(array: Array[Any], key: str) -> tuple[int, ...] | None:
    """Try to decode *key* into chunk coordinates for *array*.

    Parameters
    ----------
    array : Array
        The array whose chunk key encoding should be used.
    key : str
        The candidate chunk key string.

    Returns
    -------
    tuple of int, or None
        The decoded coordinates, or ``None`` if *key* is not a valid chunk key.
    """
    from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding, V2ChunkKeyEncoding
    from zarr.core.metadata.v2 import ArrayV2Metadata

    try:
        if isinstance(array.metadata, ArrayV2Metadata):
            parts = key.split(array.metadata.dimension_separator)
            return tuple(int(p) for p in parts)

        encoding = array.metadata.chunk_key_encoding
        if isinstance(encoding, DefaultChunkKeyEncoding):
            # Default v3 keys have the form "c<sep>0<sep>1<sep>2".
            prefix = "c" + encoding.separator
            if key == "c":
                return ()
            if not key.startswith(prefix):
                return None
            return tuple(int(p) for p in key[len(prefix) :].split(encoding.separator))
        if isinstance(encoding, V2ChunkKeyEncoding):
            return tuple(int(p) for p in key.split(encoding.separator))

        # Unknown encoding — fall back to the encoding's own decode.
        return encoding.decode_chunk_key(key)
    except (ValueError, TypeError, NotImplementedError):
        return None


def is_valid_chunk_key(array: Array[Any], key: str) -> bool:
    """Check whether *key* is a valid chunk key for *array*.

    Tries to decode the key and checks that the resulting coordinates fall
    within the storage grid (shard grid if sharding is used, chunk grid
    otherwise).

    Parameters
    ----------
    array : Array
        The array to validate against.
    key : str
        The candidate chunk key string.

    Returns
    -------
    bool
    """
    coords = decode_chunk_key(array, key)
    if coords is None:
        return False
    grid = array._shard_grid_shape
    if len(coords) != len(grid):
        return False
    return all(0 <= c < g for c, g in zip(coords, grid, strict=True))


def is_valid_array_key(array: Array[Any], key: str) -> bool:
    """Check whether *key* is a valid store key for *array*.

    Valid keys are metadata documents and chunk keys.

    Parameters
    ----------
    array : Array
        The array to validate against.
    key : str
        The candidate key, relative to the array's root.

    Returns
    -------
    bool
    """
    if key in metadata_keys(array.metadata.zarr_format):
        return True
    return is_valid_chunk_key(array, key)


def is_valid_node_key(node: Array[Any] | Group, key: str) -> bool:
    """Check whether *key* is a valid store key relative to *node*.

    For an ``Array``, valid keys are metadata documents and chunk keys.

    For a ``Group``, valid keys are the group's own metadata documents, or
    a path of the form ``<child>/<subkey>`` where ``<child>`` is a direct
    member and ``<subkey>`` is recursively valid for that child.

    Parameters
    ----------
    node : Array or Group
        The zarr node to validate against.
    key : str
        The candidate key, relative to the node's root.

    Returns
    -------
    bool
    """
    from zarr.core.array import Array

    if isinstance(node, Array):
        return is_valid_array_key(node, key)

    # Group
    if key in metadata_keys(node.metadata.zarr_format):
        return True

    # Try to match the first path component against a child member.
    if "/" in key:
        child_name, remainder = key.split("/", 1)
    else:
        # A bare name with no slash can't be a valid group-level key —
        # groups contain children (which have subkeys), not bare keys.
        return False

    try:
        child = node[child_name]
    except KeyError:
        return False

    return is_valid_node_key(child, remainder)
