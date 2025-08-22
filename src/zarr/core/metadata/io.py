from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from zarr.abc.store import set_or_delete
from zarr.core.buffer.core import default_buffer_prototype
from zarr.errors import ContainsArrayError
from zarr.storage._common import StorePath, ensure_no_existing_node

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat
    from zarr.core.group import AsyncGroup, GroupMetadata
    from zarr.core.metadata import ArrayMetadata


def _build_parents(store_path: StorePath, zarr_format: ZarrFormat) -> list[AsyncGroup]:
    from zarr.core.group import AsyncGroup, GroupMetadata

    store = store_path.store
    path = store_path.path
    if not path:
        return []

    required_parts = path.split("/")[:-1]
    parents = [
        # the root group
        AsyncGroup(
            metadata=GroupMetadata(zarr_format=zarr_format),
            store_path=StorePath(store=store, path=""),
        )
    ]

    for i, part in enumerate(required_parts):
        p = "/".join(required_parts[:i] + [part])
        parents.append(
            AsyncGroup(
                metadata=GroupMetadata(zarr_format=zarr_format),
                store_path=StorePath(store=store, path=p),
            )
        )

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
        Whether to create any missing parent groups

    Raises
    ------
    ValueError
    """
    to_save = metadata.to_buffer_dict(default_buffer_prototype())
    set_awaitables = [set_or_delete(store_path / key, value) for key, value in to_save.items()]

    if ensure_parents:
        # To enable zarr.create(store, path="a/b/c"), we need to create all the intermediate groups.
        parents = _build_parents(store_path, metadata.zarr_format)
        ensure_array_awaitables = []

        for parent in parents:
            # Error if an array already exists at any parent location. Only groups can have child nodes.
            ensure_array_awaitables.append(
                ensure_no_existing_node(parent.store_path, metadata.zarr_format, node_type="array")
            )
            set_awaitables.extend(
                [
                    (parent.store_path / key).set_if_not_exists(value)
                    for key, value in parent.metadata.to_buffer_dict(
                        default_buffer_prototype()
                    ).items()
                ]
            )

        # Checks for parent arrays must happen first, before any metadata is modified
        try:
            await asyncio.gather(*ensure_array_awaitables)
        except ContainsArrayError as e:
            set_awaitables = []  # clear awaitables to avoid printed RuntimeWarning: coroutine was never awaited
            raise ValueError(
                f"A parent of {store_path} is an array - only groups may have child nodes."
            ) from e

    await asyncio.gather(*set_awaitables)
