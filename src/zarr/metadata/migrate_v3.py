import asyncio
import logging
from typing import TYPE_CHECKING, Literal, cast

import zarr
from zarr import Array, Group
from zarr.abc.store import Store
from zarr.core.array import v2_to_v3_codecs
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from zarr.core.codec_pipeline import codecs_from_list
from zarr.core.common import (
    ZARR_JSON,
    ZARRAY_JSON,
    ZATTRS_JSON,
    ZGROUP_JSON,
    ZMETADATA_V2_JSON,
    ZarrFormat,
)
from zarr.core.group import GroupMetadata
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.storage import StorePath

if TYPE_CHECKING:
    from zarr.abc.codec import Codec

_logger = logging.getLogger(__name__)


def migrate_v2_to_v3(
    *,
    input_store: Store,
    output_store: Store | None = None,
    dry_run: bool = False,
) -> None:
    """Migrate all v2 metadata in a Zarr store to v3.

    This will create a zarr.json file at each level of a Zarr hierarchy (for every group / array).
    v2 files (.zarray, .zattrs etc.) will be left as-is.

    Parameters
    ----------
    input_store : Store
        Input Zarr to migrate.
    output_store : Store, optional
        Output location to write v3 metadata (no array data will be copied). If not provided, v3 metadata will be
        written to input_store.
    dry_run : bool, optional
        Enable a 'dry run' - files that would be created are logged, but no files are created or changed.
    """

    zarr_v2 = zarr.open(store=input_store, mode="r+")

    if output_store is not None:
        # w- access to not allow overwrite of existing data
        output_path = sync(StorePath.open(output_store, path="", mode="w-"))
    else:
        output_path = zarr_v2.store_path

    migrate_to_v3(zarr_v2, output_path, dry_run=dry_run)


def migrate_to_v3(zarr_v2: Array | Group, output_path: StorePath, dry_run: bool = False) -> None:
    """Migrate all v2 metadata in a Zarr array/group to v3.

    Note - if a group is provided, then all arrays / groups within this group will also be converted.
    A zarr.json file will be created for each level and written to output_path, with any v2 files
    (.zarray, .zattrs etc.) left as-is.

    Parameters
    ----------
    zarr_v2 : Array | Group
        An array or group with zarr_format = 2
    output_path : StorePath
        The store path to write generated v3 metadata to.
    dry_run : bool, optional
        Enable a 'dry run' - files that would be created are logged, but no files are created or changed.
    """
    if not zarr_v2.metadata.zarr_format == 2:
        raise TypeError("Only arrays / groups with zarr v2 metadata can be converted")

    if isinstance(zarr_v2.metadata, GroupMetadata):
        _convert_group(zarr_v2, output_path, dry_run)
    else:
        _convert_array(zarr_v2, output_path, dry_run)


async def remove_metadata(
    store: Store,
    zarr_format: ZarrFormat,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Remove all v2 (.zarray, .zattrs, .zgroup, .zmetadata) or v3 (zarr.json) metadata files from the given Zarr.

    Note - this will remove metadata files at all levels of the hierarchy (every group and array).

    Parameters
    ----------
    store : Store
        Zarr to remove metadata from.
    zarr_format : ZarrFormat
        Which format's metadata to remove - 2 or 3.
    force : bool, optional
        When False, metadata can only be removed if a valid alternative exists e.g. deletion of v2 metadata will
        only be allowed when v3 metadata is also present. When True, metadata can be removed when there is no
        alternative.
    dry_run : bool, optional
        Enable a 'dry run' - files that would be deleted are logged, but no files are removed or changed.
    """

    if not store.supports_deletes:
        raise ValueError("Store must support deletes to remove metadata")
    store_path = await StorePath.open(store, path="", mode="r+")

    metadata_files_all = {
        2: [ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, ZMETADATA_V2_JSON],
        3: [ZARR_JSON],
    }

    if zarr_format == 2:
        alternative_metadata = 3
    else:
        alternative_metadata = 2

    awaitables = []
    async for file_path in store.list():
        parent_path, _, file_name = file_path.rpartition("/")

        if file_name not in metadata_files_all[zarr_format]:
            continue

        if force or await _metadata_exists(
            cast(Literal[2, 3], alternative_metadata), store_path / parent_path
        ):
            _logger.info("Deleting metadata at %s", store_path / file_path)
            if not dry_run:
                awaitables.append((store_path / file_path).delete())
        else:
            raise ValueError(
                f"Cannot remove v{zarr_format} metadata at {store_path / file_path} - no v{alternative_metadata} "
                "metadata exists. To delete anyway, use the 'force' option."
            )

    await asyncio.gather(*awaitables)


def _convert_group(zarr_v2: Group, output_path: StorePath, dry_run: bool) -> None:
    if zarr_v2.metadata.consolidated_metadata is not None:
        raise NotImplementedError("Migration of consolidated metadata isn't supported.")

    # process members of the group
    for key in zarr_v2:
        migrate_to_v3(zarr_v2[key], output_path=output_path / key, dry_run=dry_run)

    # write group's converted metadata
    group_metadata_v3 = GroupMetadata(
        attributes=zarr_v2.metadata.attributes, zarr_format=3, consolidated_metadata=None
    )
    sync(_save_v3_metadata(group_metadata_v3, output_path, dry_run=dry_run))


def _convert_array(zarr_v2: Array, output_path: StorePath, dry_run: bool) -> None:
    array_metadata_v3 = _convert_array_metadata(cast(ArrayV2Metadata, zarr_v2.metadata))
    sync(_save_v3_metadata(array_metadata_v3, output_path, dry_run=dry_run))


async def _metadata_exists(zarr_format: ZarrFormat, store_path: StorePath) -> bool:
    metadata_files_required = {2: [ZARRAY_JSON, ZGROUP_JSON], 3: [ZARR_JSON]}

    for metadata_file in metadata_files_required[zarr_format]:
        if await (store_path / metadata_file).exists():
            return True

    return False


def _convert_array_metadata(metadata_v2: ArrayV2Metadata) -> ArrayV3Metadata:
    chunk_key_encoding = V2ChunkKeyEncoding(separator=metadata_v2.dimension_separator)

    codecs: tuple[Codec, ...] = ()
    # We first generate a sequence of V3 codecs, then we ensure that this sequence is valid
    aa, ab, bb = codecs_from_list(v2_to_v3_codecs(metadata_v2))
    codecs = (*aa, ab, *bb)
    return ArrayV3Metadata(
        shape=metadata_v2.shape,
        data_type=metadata_v2.dtype,
        chunk_grid=metadata_v2.chunk_grid,
        chunk_key_encoding=chunk_key_encoding,
        fill_value=metadata_v2.fill_value,
        codecs=codecs,
        attributes=metadata_v2.attributes,
        dimension_names=None,
        storage_transformers=None,
    )


async def _save_v3_metadata(
    metadata_v3: ArrayV3Metadata | GroupMetadata, output_path: StorePath, dry_run: bool = False
) -> None:
    zarr_json_path = output_path / ZARR_JSON
    if await zarr_json_path.exists():
        raise ValueError(f"{ZARR_JSON} already exists at {zarr_json_path}")

    _logger.info("Saving metadata to %s", zarr_json_path)
    to_save = metadata_v3.to_buffer_dict(default_buffer_prototype())

    if not dry_run:
        await zarr_json_path.set_if_not_exists(to_save[ZARR_JSON])
