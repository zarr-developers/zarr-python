from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Literal, overload

from zarr.core.buffer import default_buffer_prototype
from zarr.core.common import JSON, ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, ZarrFormat
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.group import GroupMetadata
from zarr.errors import (
    ArrayNotFoundError,
    GroupNotFoundError,
    MetadataValidationError,
    NodeNotFoundError,
)
from zarr.storage._utils import _join_paths, _set_return_key

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from zarr.abc.store import Store


@overload
async def _read_array_metadata(
    store: Store, path: str, zarr_format: Literal[2]
) -> ArrayV2Metadata: ...


@overload
async def _read_array_metadata(
    store: Store, path: str, zarr_format: Literal[3]
) -> ArrayV3Metadata: ...


@overload
async def _read_array_metadata(
    store: Store, path: str, zarr_format: None
) -> ArrayV2Metadata | ArrayV3Metadata: ...


async def _read_array_metadata(
    store: Store, path: str, zarr_format: ZarrFormat | None
) -> ArrayV2Metadata | ArrayV3Metadata:
    """
    Read array metadata from storage for Zarr formats 2 or 3. If zarr_format is ``None``, then
    Zarr V3 will be tried first, followed by Zarr V2.
    """
    if zarr_format == 2:
        return await _read_array_metadata_v2(store=store, path=path)
    elif zarr_format == 3:
        return await _read_array_metadata_v3(store=store, path=path)
    elif zarr_format is None:
        try:
            return await _read_array_metadata_v3(store=store, path=path)
        except ArrayNotFoundError:
            try:
                return await _read_array_metadata_v2(store=store, path=path)
            except ArrayNotFoundError as e:
                msg = (
                    "Neither Zarr V2 nor Zarr V3 array metadata documents were found in store "
                    f"{store!r} at path {path!r}."
                )
                raise ArrayNotFoundError(msg) from e
    else:
        msg = f"Invalid value for zarr_format. Expected one of 2, 3, or None. Got {zarr_format}."
        raise ValueError(msg)


def _build_metadata_v2(
    zarr_json: dict[str, JSON], attrs_json: dict[str, JSON]
) -> ArrayV2Metadata | GroupMetadata:
    """
    Convert a dict representation of Zarr V2 metadata into the corresponding metadata class.
    """
    match zarr_json:
        case {"shape": _}:
            return ArrayV2Metadata.from_dict(zarr_json | {"attributes": attrs_json})
        case _:  # pragma: no cover
            return GroupMetadata.from_dict(zarr_json | {"attributes": attrs_json})


async def _read_metadata_v2(store: Store, path: str) -> ArrayV2Metadata | GroupMetadata:
    """
    Given a store_path, return ArrayV2Metadata or GroupMetadata defined by the metadata
    document stored at store_path.path / (.zgroup | .zarray). If no metadata document is found,
    this routine raises a ``NodeNotFoundError``.
    """
    # TODO: consider first fetching array metadata, and only fetching group metadata when we don't
    # find an array
    zarray_bytes, zgroup_bytes, zattrs_bytes = await asyncio.gather(
        store.get(_join_paths([path, ZARRAY_JSON]), prototype=default_buffer_prototype()),
        store.get(_join_paths([path, ZGROUP_JSON]), prototype=default_buffer_prototype()),
        store.get(_join_paths([path, ZATTRS_JSON]), prototype=default_buffer_prototype()),
    )

    if zattrs_bytes is None:
        zattrs = {}
    else:
        zattrs = json.loads(zattrs_bytes.to_bytes())

    # TODO: decide how to handle finding both array and group metadata. The spec does not seem to
    # consider this situation. A practical approach would be to ignore that combination, and only
    # return the array metadata.
    if zarray_bytes is not None:
        zmeta = json.loads(zarray_bytes.to_bytes())
    else:
        if zgroup_bytes is not None:
            zmeta = json.loads(zgroup_bytes.to_bytes())
        else:
            # neither .zarray or .zgroup were found results in NodeNotFoundError
            msg = (
                f"Neither array nor group metadata were found in store {store!r} at path {path!r}."
            )
            raise NodeNotFoundError(msg)

    return _build_metadata_v2(zmeta, zattrs)


async def _read_group_metadata_v2(store: Store, path: str) -> GroupMetadata:
    """
    Read Zarr V2 group metadata.
    """
    try:
        meta = await _read_metadata_v2(store=store, path=path)
    except NodeNotFoundError as e:
        # NodeNotFoundError is raised when neither array nor group metadata were found,
        # but since this function is concerned with group metadata,
        # it returns a more specific exception here.
        msg = f"A group metadata document was not found in store {store!r} at path {path!r}."
        raise GroupNotFoundError(msg) from e
    if not isinstance(meta, GroupMetadata):
        # TODO: test this exception
        msg = (
            f"Group metadata was not found in store {store!r} at path {path!r}. "
            "An array metadata document was found there instead."
        )
        raise GroupNotFoundError(msg)
    return meta


async def _read_group_metadata_v3(store: Store, path: str) -> GroupMetadata:
    """
    Read Zarr V3 group metadata.
    """
    try:
        meta = await _read_metadata_v3(store=store, path=path)
    except NodeNotFoundError as e:
        # NodeNotFoundError is raised when neither array nor group metadata were found,
        # but since this function is concerned with group metadata,
        # it returns a more specific exception here.
        msg = f"A group metadata document was not found in store {store!r} at path {path!r}."
        raise GroupNotFoundError(msg) from e
    if not isinstance(meta, GroupMetadata):
        # TODO: test this exception
        msg = (
            f"Group metadata was not found in store {store!r} at path {path!r}. "
            "An array metadata document was found there instead."
        )
        raise GroupNotFoundError(msg)
    return meta


async def _read_group_metadata(
    store: Store, path: str, *, zarr_format: ZarrFormat
) -> GroupMetadata:
    if zarr_format == 2:
        return await _read_group_metadata_v2(store=store, path=path)
    return await _read_group_metadata_v3(store=store, path=path)


def _build_metadata_v3(zarr_json: dict[str, JSON]) -> ArrayV3Metadata | GroupMetadata:
    """
    Convert a dict representation of Zarr V3 metadata into the corresponding metadata class.
    """
    if "node_type" not in zarr_json:
        msg = (
            "Invalid value for node_type. "
            "Expected 'array' or 'group'. Got nothing (the key is missing)."
        )
        raise MetadataValidationError(msg)
    match zarr_json:
        case {"node_type": "array"}:
            return ArrayV3Metadata.from_dict(zarr_json)
        case {"node_type": "group"}:
            return GroupMetadata.from_dict(zarr_json)
        case _:  # pragma: no cover
            raise ValueError(
                "invalid value for `node_type` key in metadata document"
            )  # pragma: no cover


async def _read_metadata_v3(store: Store, path: str) -> ArrayV3Metadata | GroupMetadata:
    """
    Given a store_path, return ArrayV3Metadata or GroupMetadata defined by the metadata
    document stored at store_path.path / zarr.json. If no such document is found, raise a
    FileNotFoundError.
    """
    zarr_json_bytes = await store.get(
        _join_paths([path, ZARR_JSON]), prototype=default_buffer_prototype()
    )
    if zarr_json_bytes is None:
        msg = f"Neither array nor group metadata were found in store {store!r} at path {path!r}."
        raise NodeNotFoundError(msg)
    else:
        zarr_json = json.loads(zarr_json_bytes.to_bytes())
        return _build_metadata_v3(zarr_json)


async def _read_array_metadata_v3(store: Store, path: str) -> ArrayV3Metadata:
    """
    Read Zarr V3 array metadata from a store at a path. Raises ``ArrayNotFoundError`` if Zarr V3
    metadata was not found, or if Zarr V3 metadata was found, but it described a group instead of
    an array.
    """
    try:
        maybe_array_meta = await _read_metadata_v3(store, path)
    except NodeNotFoundError as e:
        msg = f"Zarr V3 array metadata was not found in store {store!r} at path {path!r}."
        raise ArrayNotFoundError(msg) from e
    if not isinstance(maybe_array_meta, ArrayV3Metadata):
        msg = (
            f"Zarr V3 array metadata was not found in store {store!r} at path {path!r}."
            "Zarr V3 group metadata was found there instead."
        )
        raise ArrayNotFoundError(msg)
    return maybe_array_meta


async def _read_array_metadata_v2(store: Store, path: str) -> ArrayV2Metadata:
    """
    Read Zarr V2 array metadata. Raises ``ArrayNotFoundError`` if .zarray is not found.

    Performance note: Only use this function to read from a storage location where an array is
    expected. If a storage location could contain an array, or a group, then the function
    ``_read_metadata_v2`` makes more efficient use of storage operations and should be used instead.
    """
    zarray_bytes, zattrs_bytes = await asyncio.gather(
        store.get(_join_paths([path, ZARRAY_JSON]), prototype=default_buffer_prototype()),
        store.get(_join_paths([path, ZATTRS_JSON]), prototype=default_buffer_prototype()),
    )

    if zattrs_bytes is None:
        attrs_json = {}
    else:
        attrs_json = json.loads(zattrs_bytes.to_bytes())

    if zarray_bytes is not None:
        zmeta = json.loads(zarray_bytes.to_bytes())
        return ArrayV2Metadata.from_dict(zmeta | {"attributes": attrs_json})
    else:
        raise ArrayNotFoundError(
            f"Zarr V2 array metadata was not found in store {store!r} at path {path!r}."
        )


def _persist_metadata(
    store: Store,
    path: str,
    metadata: ArrayV2Metadata | ArrayV3Metadata | GroupMetadata,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[Coroutine[None, None, str], ...]:
    """
    Prepare to save a metadata document to storage, returning a tuple of coroutines that must be awaited.
    """

    to_save = metadata.to_buffer_dict(default_buffer_prototype())
    return tuple(
        _set_return_key(store=store, key=_join_paths([path, key]), value=value, semaphore=semaphore)
        for key, value in to_save.items()
    )
