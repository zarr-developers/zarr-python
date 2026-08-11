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

ZARR_JSON = "zarr.json"
ZARRAY_JSON = ".zarray"
ZGROUP_JSON = ".zgroup"
ZATTRS_JSON = ".zattrs"
ZMETADATA_V2_JSON = ".zmetadata"

if TYPE_CHECKING:
    from zarr import Array, Group

_ARRAY_METADATA_KEYS_V3 = frozenset({ZARR_JSON})
_ARRAY_METADATA_KEYS_V2 = frozenset({ZARRAY_JSON, ZATTRS_JSON})
_GROUP_METADATA_KEYS_V3 = frozenset({ZARR_JSON})
_GROUP_METADATA_KEYS_V2 = frozenset({ZGROUP_JSON, ZATTRS_JSON, ZMETADATA_V2_JSON})


def array_metadata_keys(zarr_format: int) -> frozenset[str]:
    """Return the metadata key basenames an array owns, for a zarr format.

    Parameters
    ----------
    zarr_format : int
        The zarr format version (2 or 3).

    Returns
    -------
    frozenset of str
    """
    if zarr_format == 3:
        return _ARRAY_METADATA_KEYS_V3
    return _ARRAY_METADATA_KEYS_V2


def group_metadata_keys(zarr_format: int) -> frozenset[str]:
    """Return the metadata key basenames a group owns, for a zarr format.

    Parameters
    ----------
    zarr_format : int
        The zarr format version (2 or 3).

    Returns
    -------
    frozenset of str
    """
    if zarr_format == 3:
        return _GROUP_METADATA_KEYS_V3
    return _GROUP_METADATA_KEYS_V2


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
    try:
        if array.metadata.zarr_format == 2:
            coords = tuple(int(p) for p in key.split(array.metadata.dimension_separator))
            # A 0-d v2 array holds its single chunk under "0", which decodes
            # to a 1-tuple that no 0-d grid could match.
            if len(array.shape) == 0:
                return () if coords == (0,) else None
            return coords

        # Ask zarr rather than predicting it: the encoding owns its own
        # grammar, so a new or third-party chunk key encoding decodes here
        # without this package knowing anything about it.
        return array.metadata.chunk_key_encoding.decode_chunk_key(key)
    except (ValueError, TypeError, NotImplementedError):
        return None


def _shard_grid_shape(array: Array[Any]) -> tuple[int, ...]:
    """Shape of the shard grid, falling back to the chunk grid when unsharded."""
    shard_shape = array.shards if array.shards is not None else array.chunks
    return tuple(-(-s // c) for s, c in zip(array.shape, shard_shape, strict=True))


def is_valid_chunk_key(array: Array[Any], key: str) -> bool:
    """Check whether *key* is a valid chunk key for *array*.

    Decodes the key, checks that the resulting coordinates fall within the
    storage grid (shard grid if sharding is used, chunk grid otherwise), and
    requires the key to be spelled exactly as zarr itself would spell it.

    That last check is what makes the accepted key set equal to the set of
    keys zarr can actually read. Decoding alone is lenient -- `int` accepts
    leading zeros, a leading `+`/`-`, surrounding whitespace, underscore
    separators, and non-ASCII decimal digits -- so `c/00/00` and `c/0/0`
    decode to the same coordinates while naming *different* store keys. A
    write to the non-canonical spelling would be stored under a key no reader
    ever looks up: the client sees success and the data is invisible.

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
    grid = _shard_grid_shape(array)
    if len(coords) != len(grid):
        return False
    if not all(0 <= c < g for c, g in zip(coords, grid, strict=True)):
        return False
    return array.metadata.encode_chunk_key(coords) == key


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
    if key in array_metadata_keys(array.metadata.zarr_format):
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
    from zarr import Array

    if isinstance(node, Array):
        return is_valid_array_key(node, key)

    # Group
    if key in group_metadata_keys(node.metadata.zarr_format):
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
        # There is no such member, so no key beneath it can be valid.
        #
        # Only a missing name is caught here. Anything else -- an I/O error
        # reading the child's metadata, unparsable JSON, a codec from a
        # plugin this process lacks -- means the key could not be *judged*,
        # which is not the same as judging it absent. Reporting those as 404
        # would be a lie with teeth: under the v3 spec an absent chunk is an
        # uninitialized one, so a correct reader answers a 404 by silently
        # substituting the array's fill value over data that exists. Letting
        # them propagate surfaces a 500, which is the honest answer and the
        # one a client cannot mistake for data.
        return False

    return is_valid_node_key(child, remainder)
