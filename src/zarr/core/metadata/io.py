from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from zarr.abc.store import set_or_delete
from zarr.core.buffer.core import default_buffer_prototype
from zarr.errors import ContainsArrayError
from zarr.storage._common import StorePath, ensure_no_existing_node

if TYPE_CHECKING:
    from zarr.core.buffer import Buffer
    from zarr.core.common import ZarrFormat
    from zarr.core.group import GroupMetadata
    from zarr.core.metadata import ArrayMetadata


def _build_parents(store_path: StorePath, zarr_format: ZarrFormat) -> dict[str, GroupMetadata]:
    from zarr.core.group import GroupMetadata

    path = store_path.path
    if not path:
        return {}

    required_parts = path.split("/")[:-1]

    # the root group
    parents = {"": GroupMetadata(zarr_format=zarr_format)}

    for i, part in enumerate(required_parts):
        parent_path = "/".join(required_parts[:i] + [part])
        parents[parent_path] = GroupMetadata(zarr_format=zarr_format)

    return parents


async def save_metadata(
    store_path: StorePath, metadata: ArrayMetadata | GroupMetadata, ensure_parents: bool = False
) -> None:
    """Asynchronously save the array or group metadata.

    Parameters
    ----------
    store_path : StorePath
        Location to save metadata.
    metadata : ArrayMetadata | GroupMetadata
        Metadata to save.
    ensure_parents : bool, optional
        Create any missing parent groups, and check no existing parents are arrays.

    Raises
    ------
    ValueError
    """
    to_save = metadata.to_buffer_dict(default_buffer_prototype())
    await _write_metadata(store_path, to_save, metadata.zarr_format, ensure_parents=ensure_parents)


async def save_new_metadata(
    store_path: StorePath,
    metadata: ArrayMetadata | GroupMetadata,
    *,
    overwrite: bool,
    ensure_parents: bool = False,
) -> None:
    """Save the metadata of a new array or group, replacing any existing node if requested.

    The metadata is encoded before the store is modified, so metadata that cannot be
    encoded raises without deleting an existing node.

    Parameters
    ----------
    store_path : StorePath
        Location of the new node.
    metadata : ArrayMetadata | GroupMetadata
        Metadata of the new node.
    overwrite : bool
        If true and the store supports deletes, delete any existing node at `store_path`.
        Otherwise, raise if a node already exists at `store_path`.
    ensure_parents : bool, optional
        Create any missing parent groups, and check no existing parents are arrays.
    """
    to_save = metadata.to_buffer_dict(default_buffer_prototype())
    if overwrite and store_path.store.supports_deletes:
        await store_path.delete_dir()
    else:
        await ensure_no_existing_node(store_path, zarr_format=metadata.zarr_format)
    await _write_metadata(store_path, to_save, metadata.zarr_format, ensure_parents=ensure_parents)


async def _write_metadata(
    store_path: StorePath,
    to_save: dict[str, Buffer],
    zarr_format: ZarrFormat,
    *,
    ensure_parents: bool,
) -> None:
    set_awaitables = [set_or_delete(store_path / key, value) for key, value in to_save.items()]

    if ensure_parents:
        # To enable zarr.create(store, path="a/b/c"), we need to create all the intermediate groups.
        parents = _build_parents(store_path, zarr_format)
        ensure_array_awaitables = []

        for parent_path, parent_metadata in parents.items():
            parent_store_path = StorePath(store_path.store, parent_path)

            # Error if an array already exists at any parent location. Only groups can have child nodes.
            ensure_array_awaitables.append(
                ensure_no_existing_node(
                    parent_store_path, parent_metadata.zarr_format, node_type="array"
                )
            )
            set_awaitables.extend(
                [
                    (parent_store_path / key).set_if_not_exists(value)
                    for key, value in parent_metadata.to_buffer_dict(
                        default_buffer_prototype()
                    ).items()
                ]
            )

        # Checks for parent arrays must happen first, before any metadata is modified
        try:
            await asyncio.gather(*ensure_array_awaitables)
        except ContainsArrayError as e:
            # clear awaitables to avoid RuntimeWarning: coroutine was never awaited
            for awaitable in set_awaitables:
                awaitable.close()

            raise ValueError(
                f"A parent of {store_path} is an array - only groups may have child nodes."
            ) from e

    await asyncio.gather(*set_awaitables)
