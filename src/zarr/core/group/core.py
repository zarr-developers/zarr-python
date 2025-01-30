from __future__ import annotations
from zarr.abc.store import Store, set_or_delete
from zarr.api.asynchronous import _CREATE_MODES, _READ_MODES, _handle_zarr_version_or_format, _infer_overwrite
from zarr.core._info import GroupInfo
from zarr.core.array import AsyncArray, _build_parents, _parse_deprecated_compressor, create_array
from zarr.core.array_spec import ArrayConfig, ArrayConfigLike
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.core.chunk_key_encodings import ChunkKeyEncoding
from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, ZMETADATA_V2_JSON, ChunkCoords, _default_zarr_format, parse_shapelike
from zarr.core.config import config
from zarr.core.group import AsyncGroup, logger
from zarr.core.group.metadata import GroupMetadata, _build_metadata_v2, _build_metadata_v3
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.errors import MetadataValidationError
from zarr.storage import StorePath
from zarr.storage._common import ensure_no_existing_node, make_store_path


import numpy as np


import asyncio
import json
import warnings
from collections import defaultdict
from collections.abc import AsyncGenerator, Generator, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast
from typing_extensions import deprecated

DefaultT = TypeVar("DefaultT")
logger = logging.getLogger("zarr.group")

async def _iter_members_deep(
    group: AsyncGroup,
    *,
    max_depth: int | None,
    skip_keys: tuple[str, ...],
    semaphore: asyncio.Semaphore | None = None,
) -> AsyncGenerator[
    tuple[str, AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup], None
]:
    """
    Iterate over the arrays and groups contained in a group, and optionally the
    arrays and groups contained in those groups.

    Parameters
    ----------
    group : AsyncGroup
        The group to traverse.
    max_depth : int | None
        The maximum depth of recursion.
    skip_keys : tuple[str, ...]
        A tuple of keys to skip when iterating over the possible members of the group.
    semaphore : asyncio.Semaphore | None
        An optional semaphore to use for concurrency control.

    Yields
    ------
    tuple[str, AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup]
    """

    to_recurse = {}
    do_recursion = max_depth is None or max_depth > 0

    if max_depth is None:
        new_depth = None
    else:
        new_depth = max_depth - 1
    async for name, node in _iter_members(group, skip_keys=skip_keys, semaphore=semaphore):
        yield name, node
        if isinstance(node, AsyncGroup) and do_recursion:
            to_recurse[name] = _iter_members_deep(
                node, max_depth=new_depth, skip_keys=skip_keys, semaphore=semaphore
            )

    for prefix, subgroup_iter in to_recurse.items():
        async for name, node in subgroup_iter:
            key = f"{prefix}/{name}".lstrip("/")
            yield key, node


def _build_node_v3(
    metadata: ArrayV3Metadata | GroupMetadata, store_path: StorePath
) -> AsyncArray[ArrayV3Metadata] | AsyncGroup:
    """
    Take a metadata object and return a node (AsyncArray or AsyncGroup).
    """
    match metadata:
        case ArrayV3Metadata():
            return AsyncArray(metadata, store_path=store_path)
        case GroupMetadata():
            return AsyncGroup(metadata, store_path=store_path)
        case _:
            raise ValueError(f"Unexpected metadata type: {type(metadata)}")


def _build_node_v2(
    metadata: ArrayV2Metadata | GroupMetadata, store_path: StorePath
) -> AsyncArray[ArrayV2Metadata] | AsyncGroup:
    """
    Take a metadata object and return a node (AsyncArray or AsyncGroup).
    """

    match metadata:
        case ArrayV2Metadata():
            return AsyncArray(metadata, store_path=store_path)
        case GroupMetadata():
            return AsyncGroup(metadata, store_path=store_path)
        case _:
            raise ValueError(f"Unexpected metadata type: {type(metadata)}")


@dataclass(frozen=True)
class AsyncGroup:
    """
    Asynchronous Group object.
    """

    metadata: GroupMetadata
    store_path: StorePath

    @classmethod
    async def from_store(
        cls,
        store: StoreLike,
        *,
        attributes: dict[str, Any] | None = None,
        overwrite: bool = False,
        zarr_format: ZarrFormat = 3,
    ) -> AsyncGroup:
        store_path = await make_store_path(store)

        if overwrite:
            if store_path.store.supports_deletes:
                await store_path.delete_dir()
            else:
                await ensure_no_existing_node(store_path, zarr_format=zarr_format)
        else:
            await ensure_no_existing_node(store_path, zarr_format=zarr_format)
        attributes = attributes or {}
        group = cls(
            metadata=GroupMetadata(attributes=attributes, zarr_format=zarr_format),
            store_path=store_path,
        )
        await group._save_metadata(ensure_parents=True)
        return group

    @classmethod
    async def open(
        cls,
        store: StoreLike,
        zarr_format: ZarrFormat | None = 3,
        use_consolidated: bool | str | None = None,
    ) -> AsyncGroup:
        """Open a new AsyncGroup

        Parameters
        ----------
        store : StoreLike
        zarr_format : {2, 3}, optional
        use_consolidated : bool or str, default None
            Whether to use consolidated metadata.

            By default, consolidated metadata is used if it's present in the
            store (in the ``zarr.json`` for Zarr format 3 and in the ``.zmetadata`` file
            for Zarr format 2).

            To explicitly require consolidated metadata, set ``use_consolidated=True``,
            which will raise an exception if consolidated metadata is not found.

            To explicitly *not* use consolidated metadata, set ``use_consolidated=False``,
            which will fall back to using the regular, non consolidated metadata.

            Zarr format 2 allowed configuring the key storing the consolidated metadata
            (``.zmetadata`` by default). Specify the custom key as ``use_consolidated``
            to load consolidated metadata from a non-default key.
        """
        store_path = await make_store_path(store)

        consolidated_key = ZMETADATA_V2_JSON

        if (zarr_format == 2 or zarr_format is None) and isinstance(use_consolidated, str):
            consolidated_key = use_consolidated

        if zarr_format == 2:
            paths = [store_path / ZGROUP_JSON, store_path / ZATTRS_JSON]
            if use_consolidated or use_consolidated is None:
                paths.append(store_path / consolidated_key)

            zgroup_bytes, zattrs_bytes, *rest = await asyncio.gather(
                *[path.get() for path in paths]
            )
            if zgroup_bytes is None:
                raise FileNotFoundError(store_path)

            if use_consolidated or use_consolidated is None:
                maybe_consolidated_metadata_bytes = rest[0]

            else:
                maybe_consolidated_metadata_bytes = None

        elif zarr_format == 3:
            zarr_json_bytes = await (store_path / ZARR_JSON).get()
            if zarr_json_bytes is None:
                raise FileNotFoundError(store_path)
        elif zarr_format is None:
            (
                zarr_json_bytes,
                zgroup_bytes,
                zattrs_bytes,
                maybe_consolidated_metadata_bytes,
            ) = await asyncio.gather(
                (store_path / ZARR_JSON).get(),
                (store_path / ZGROUP_JSON).get(),
                (store_path / ZATTRS_JSON).get(),
                (store_path / str(consolidated_key)).get(),
            )
            if zarr_json_bytes is not None and zgroup_bytes is not None:
                # warn and favor v3
                msg = f"Both zarr.json (Zarr format 3) and .zgroup (Zarr format 2) metadata objects exist at {store_path}. Zarr format 3 will be used."
                warnings.warn(msg, stacklevel=1)
            if zarr_json_bytes is None and zgroup_bytes is None:
                raise FileNotFoundError(
                    f"could not find zarr.json or .zgroup objects in {store_path}"
                )
            # set zarr_format based on which keys were found
            if zarr_json_bytes is not None:
                zarr_format = 3
            else:
                zarr_format = 2
        else:
            raise MetadataValidationError("zarr_format", "2, 3, or None", zarr_format)

        if zarr_format == 2:
            # this is checked above, asserting here for mypy
            assert zgroup_bytes is not None

            if use_consolidated and maybe_consolidated_metadata_bytes is None:
                # the user requested consolidated metadata, but it was missing
                raise ValueError(consolidated_key)

            elif use_consolidated is False:
                # the user explicitly opted out of consolidated_metadata.
                # Discard anything we might have read.
                maybe_consolidated_metadata_bytes = None

            return cls._from_bytes_v2(
                store_path, zgroup_bytes, zattrs_bytes, maybe_consolidated_metadata_bytes
            )
        else:
            # V3 groups are comprised of a zarr.json object
            assert zarr_json_bytes is not None
            if not isinstance(use_consolidated, bool | None):
                raise TypeError("use_consolidated must be a bool or None for Zarr format 3.")

            return cls._from_bytes_v3(
                store_path,
                zarr_json_bytes,
                use_consolidated=use_consolidated,
            )

    @classmethod
    def _from_bytes_v2(
        cls,
        store_path: StorePath,
        zgroup_bytes: Buffer,
        zattrs_bytes: Buffer | None,
        consolidated_metadata_bytes: Buffer | None,
    ) -> AsyncGroup:
        # V2 groups are comprised of a .zgroup and .zattrs objects
        zgroup = json.loads(zgroup_bytes.to_bytes())
        zattrs = json.loads(zattrs_bytes.to_bytes()) if zattrs_bytes is not None else {}
        group_metadata = {**zgroup, "attributes": zattrs}

        if consolidated_metadata_bytes is not None:
            v2_consolidated_metadata = json.loads(consolidated_metadata_bytes.to_bytes())
            v2_consolidated_metadata = v2_consolidated_metadata["metadata"]
            # We already read zattrs and zgroup. Should we ignore these?
            v2_consolidated_metadata.pop(".zattrs", None)
            v2_consolidated_metadata.pop(".zgroup", None)

            consolidated_metadata: defaultdict[str, dict[str, Any]] = defaultdict(dict)

            # keys like air/.zarray, air/.zattrs
            for k, v in v2_consolidated_metadata.items():
                path, kind = k.rsplit("/.", 1)

                if kind == "zarray":
                    consolidated_metadata[path].update(v)
                elif kind == "zattrs":
                    consolidated_metadata[path]["attributes"] = v
                elif kind == "zgroup":
                    consolidated_metadata[path].update(v)
                else:
                    raise ValueError(f"Invalid file type '{kind}' at path '{path}")
            group_metadata["consolidated_metadata"] = {
                "metadata": dict(consolidated_metadata),
                "kind": "inline",
                "must_understand": False,
            }

        return cls.from_dict(store_path, group_metadata)

    @classmethod
    def _from_bytes_v3(
        cls,
        store_path: StorePath,
        zarr_json_bytes: Buffer,
        use_consolidated: bool | None,
    ) -> AsyncGroup:
        group_metadata = json.loads(zarr_json_bytes.to_bytes())
        if use_consolidated and group_metadata.get("consolidated_metadata") is None:
            msg = f"Consolidated metadata requested with 'use_consolidated=True' but not found in '{store_path.path}'."
            raise ValueError(msg)

        elif use_consolidated is False:
            # Drop consolidated metadata if it's there.
            group_metadata.pop("consolidated_metadata", None)

        return cls.from_dict(store_path, group_metadata)

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        data: dict[str, Any],
    ) -> AsyncGroup:
        return cls(
            metadata=GroupMetadata.from_dict(data),
            store_path=store_path,
        )

    async def setitem(self, key: str, value: Any) -> None:
        """
        Fastpath for creating a new array
        New arrays will be created with default array settings for the array type.

        Parameters
        ----------
        key : str
            Array name
        value : array-like
            Array data
        """
        path = self.store_path / key
        await save_array(
            store=path, arr=value, zarr_format=self.metadata.zarr_format, overwrite=True
        )

    async def getitem(
        self,
        key: str,
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup:
        """
        Get a subarray or subgroup from the group.

        Parameters
        ----------
        key : str
            Array or group name

        Returns
        -------
        AsyncArray or AsyncGroup
        """
        store_path = self.store_path / key
        logger.debug("key=%s, store_path=%s", key, store_path)
        metadata: ArrayV2Metadata | ArrayV3Metadata | GroupMetadata

        # Consolidated metadata lets us avoid some I/O operations so try that first.
        if self.metadata.consolidated_metadata is not None:
            return self._getitem_consolidated(store_path, key, prefix=self.name)

        # Note:
        # in zarr-python v2, we first check if `key` references an Array, else if `key` references
        # a group,using standalone `contains_array` and `contains_group` functions. These functions
        # are reusable, but for v3 they would perform redundant I/O operations.
        # Not clear how much of that strategy we want to keep here.
        elif self.metadata.zarr_format == 3:
            zarr_json_bytes = await (store_path / ZARR_JSON).get()
            if zarr_json_bytes is None:
                raise KeyError(key)
            else:
                zarr_json = json.loads(zarr_json_bytes.to_bytes())
                metadata = _build_metadata_v3(zarr_json)
                return _build_node_v3(metadata, store_path)

        elif self.metadata.zarr_format == 2:
            # Q: how do we like optimistically fetching .zgroup, .zarray, and .zattrs?
            # This guarantees that we will always make at least one extra request to the store
            zgroup_bytes, zarray_bytes, zattrs_bytes = await asyncio.gather(
                (store_path / ZGROUP_JSON).get(),
                (store_path / ZARRAY_JSON).get(),
                (store_path / ZATTRS_JSON).get(),
            )

            if zgroup_bytes is None and zarray_bytes is None:
                raise KeyError(key)

            # unpack the zarray, if this is None then we must be opening a group
            zarray = json.loads(zarray_bytes.to_bytes()) if zarray_bytes else None
            zgroup = json.loads(zgroup_bytes.to_bytes()) if zgroup_bytes else None
            # unpack the zattrs, this can be None if no attrs were written
            zattrs = json.loads(zattrs_bytes.to_bytes()) if zattrs_bytes is not None else {}

            if zarray is not None:
                metadata = _build_metadata_v2(zarray, zattrs)
                return _build_node_v2(metadata=metadata, store_path=store_path)
            else:
                # this is just for mypy
                if TYPE_CHECKING:
                    assert zgroup is not None
                metadata = _build_metadata_v2(zgroup, zattrs)
                return _build_node_v2(metadata=metadata, store_path=store_path)
        else:
            raise ValueError(f"unexpected zarr_format: {self.metadata.zarr_format}")

    def _getitem_consolidated(
        self, store_path: StorePath, key: str, prefix: str
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup:
        # getitem, in the special case where we have consolidated metadata.
        # Note that this is a regular def (non async) function.
        # This shouldn't do any additional I/O.

        # the caller needs to verify this!
        assert self.metadata.consolidated_metadata is not None

        # we support nested getitems like group/subgroup/array
        indexers = key.split("/")
        indexers.reverse()
        metadata: ArrayV2Metadata | ArrayV3Metadata | GroupMetadata = self.metadata

        while indexers:
            indexer = indexers.pop()
            if isinstance(metadata, ArrayV2Metadata | ArrayV3Metadata):
                # we've indexed into an array with group["array/subarray"]. Invalid.
                raise KeyError(key)
            if metadata.consolidated_metadata is None:
                # we've indexed into a group without consolidated metadata.
                # This isn't normal; typically, consolidated metadata
                # will include explicit markers for when there are no child
                # nodes as metadata={}.
                # We have some freedom in exactly how we interpret this case.
                # For now, we treat None as the same as {}, i.e. we don't
                # have any children.
                raise KeyError(key)
            try:
                metadata = metadata.consolidated_metadata.metadata[indexer]
            except KeyError as e:
                # The Group Metadata has consolidated metadata, but the key
                # isn't present. We trust this to mean that the key isn't in
                # the hierarchy, and *don't* fall back to checking the store.
                msg = f"'{key}' not found in consolidated metadata."
                raise KeyError(msg) from e

        # update store_path to ensure that AsyncArray/Group.name is correct
        if prefix != "/":
            key = "/".join([prefix.lstrip("/"), key])
        store_path = StorePath(store=store_path.store, path=key)

        if isinstance(metadata, GroupMetadata):
            return AsyncGroup(metadata=metadata, store_path=store_path)
        else:
            return AsyncArray(metadata=metadata, store_path=store_path)

    async def delitem(self, key: str) -> None:
        """Delete a group member.

        Parameters
        ----------
        key : str
            Array or group name
        """
        store_path = self.store_path / key

        await store_path.delete_dir()
        if self.metadata.consolidated_metadata:
            self.metadata.consolidated_metadata.metadata.pop(key, None)
            await self._save_metadata()

    async def get(
        self, key: str, default: DefaultT | None = None
    ) -> AsyncArray[Any] | AsyncGroup | DefaultT | None:
        """Obtain a group member, returning default if not found.

        Parameters
        ----------
        key : str
            Group member name.
        default : object
            Default value to return if key is not found (default: None).

        Returns
        -------
        object
            Group member (AsyncArray or AsyncGroup) or default if not found.
        """
        try:
            return await self.getitem(key)
        except KeyError:
            return default

    async def _save_metadata(self, ensure_parents: bool = False) -> None:
        to_save = self.metadata.to_buffer_dict(default_buffer_prototype())
        awaitables = [set_or_delete(self.store_path / key, value) for key, value in to_save.items()]

        if ensure_parents:
            parents = _build_parents(self)
            for parent in parents:
                awaitables.extend(
                    [
                        (parent.store_path / key).set_if_not_exists(value)
                        for key, value in parent.metadata.to_buffer_dict(
                            default_buffer_prototype()
                        ).items()
                    ]
                )

        await asyncio.gather(*awaitables)

    @property
    def path(self) -> str:
        """Storage path."""
        return self.store_path.path

    @property
    def name(self) -> str:
        """Group name following h5py convention."""
        if self.path:
            # follow h5py convention: add leading slash
            name = self.path
            if name[0] != "/":
                name = "/" + name
            return name
        return "/"

    @property
    def basename(self) -> str:
        """Final component of name."""
        return self.name.split("/")[-1]

    @property
    def attrs(self) -> dict[str, Any]:
        return self.metadata.attributes

    @property
    def info(self) -> Any:
        """
        Return a visual representation of the statically known information about a group.

        Note that this doesn't include dynamic information, like the number of child
        Groups or Arrays.

        Returns
        -------
        GroupInfo

        See Also
        --------
        AsyncGroup.info_complete
            All information about a group, including dynamic information
        """

        if self.metadata.consolidated_metadata:
            members = list(self.metadata.consolidated_metadata.flattened_metadata.values())
        else:
            members = None
        return self._info(members=members)

    async def info_complete(self) -> Any:
        """
        Return all the information for a group.

        This includes dynamic information like the number
        of child Groups or Arrays. If this group doesn't contain consolidated
        metadata then this will need to read from the backing Store.

        Returns
        -------
        GroupInfo

        See Also
        --------
        AsyncGroup.info
        """
        members = [x[1].metadata async for x in self.members(max_depth=None)]
        return self._info(members=members)

    def _info(
        self, members: list[ArrayV2Metadata | ArrayV3Metadata | GroupMetadata] | None = None
    ) -> Any:
        kwargs = {}
        if members is not None:
            kwargs["_count_members"] = len(members)
            count_arrays = 0
            count_groups = 0
            for member in members:
                if isinstance(member, GroupMetadata):
                    count_groups += 1
                else:
                    count_arrays += 1
            kwargs["_count_arrays"] = count_arrays
            kwargs["_count_groups"] = count_groups

        return GroupInfo(
            _name=self.store_path.path,
            _read_only=self.read_only,
            _store_type=type(self.store_path.store).__name__,
            _zarr_format=self.metadata.zarr_format,
            # maybe do a typeddict
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def store(self) -> Store:
        return self.store_path.store

    @property
    def read_only(self) -> bool:
        # Backwards compatibility for 2.x
        return self.store_path.read_only

    @property
    def synchronizer(self) -> None:
        # Backwards compatibility for 2.x
        # Not implemented in 3.x yet.
        return None

    async def create_group(
        self,
        name: str,
        *,
        overwrite: bool = False,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncGroup:
        """Create a sub-group.

        Parameters
        ----------
        name : str
            Group name.
        overwrite : bool, optional
            If True, do not raise an error if the group already exists.
        attributes : dict, optional
            Group attributes.

        Returns
        -------
        g : AsyncGroup
        """
        attributes = attributes or {}
        return await type(self).from_store(
            self.store_path / name,
            attributes=attributes,
            overwrite=overwrite,
            zarr_format=self.metadata.zarr_format,
        )

    async def require_group(self, name: str, overwrite: bool = False) -> AsyncGroup:
        """Obtain a sub-group, creating one if it doesn't exist.

        Parameters
        ----------
        name : str
            Group name.
        overwrite : bool, optional
            Overwrite any existing group with given `name` if present.

        Returns
        -------
        g : AsyncGroup
        """
        if overwrite:
            # TODO: check that overwrite=True errors if an array exists where the group is being created
            grp = await self.create_group(name, overwrite=True)
        else:
            try:
                item: (
                    AsyncGroup | AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]
                ) = await self.getitem(name)
                if not isinstance(item, AsyncGroup):
                    raise TypeError(
                        f"Incompatible object ({item.__class__.__name__}) already exists"
                    )
                assert isinstance(item, AsyncGroup)  # make mypy happy
                grp = item
            except KeyError:
                grp = await self.create_group(name)
        return grp

    async def require_groups(self, *names: str) -> tuple[AsyncGroup, ...]:
        """Convenience method to require multiple groups in a single call.

        Parameters
        ----------
        *names : str
            Group names.

        Returns
        -------
        Tuple[AsyncGroup, ...]
        """
        if not names:
            return ()
        return tuple(await asyncio.gather(*(self.require_group(name) for name in names)))

    async def create_array(
        self,
        name: str,
        *,
        shape: ShapeLike,
        dtype: npt.DTypeLike,
        chunks: ChunkCoords | Literal["auto"] = "auto",
        shards: ShardsLike | None = None,
        filters: FiltersLike = "auto",
        compressors: CompressorsLike = "auto",
        compressor: CompressorLike = "auto",
        serializer: SerializerLike = "auto",
        fill_value: Any | None = 0,
        order: MemoryOrder | None = None,
        attributes: dict[str, JSON] | None = None,
        chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None = None,
        dimension_names: Iterable[str] | None = None,
        storage_options: dict[str, Any] | None = None,
        overwrite: bool = False,
        config: ArrayConfig | ArrayConfigLike | None = None,
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an array within this group.

        This method lightly wraps :func:`zarr.core.array.create_array`.

        Parameters
        ----------
        name : str
            The name of the array relative to the group. If ``path`` is ``None``, the array will be located
            at the root of the store.
        shape : ChunkCoords
            Shape of the array.
        dtype : npt.DTypeLike
            Data type of the array.
        chunks : ChunkCoords, optional
            Chunk shape of the array.
            If not specified, default are guessed based on the shape and dtype.
        shards : ChunkCoords, optional
            Shard shape of the array. The default value of ``None`` results in no sharding at all.
        filters : Iterable[Codec], optional
            Iterable of filters to apply to each chunk of the array, in order, before serializing that
            chunk to bytes.

            For Zarr format 3, a "filter" is a codec that takes an array and returns an array,
            and these values must be instances of ``ArrayArrayCodec``, or dict representations
            of ``ArrayArrayCodec``.
            If no ``filters`` are provided, a default set of filters will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_filters``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default filters.

            For Zarr format 2, a "filter" can be any numcodecs codec; you should ensure that the
            the order if your filters is consistent with the behavior of each filter.
            If no ``filters`` are provided, a default set of filters will be used.
            These defaults can be changed by modifying the value of ``array.v2_default_filters``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default filters.
        compressors : Iterable[Codec], optional
            List of compressors to apply to the array. Compressors are applied in order, and after any
            filters are applied (if any are specified) and the data is serialized into bytes.

            For Zarr format 3, a "compressor" is a codec that takes a bytestream, and
            returns another bytestream. Multiple compressors my be provided for Zarr format 3.
            If no ``compressors`` are provided, a default set of compressors will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_compressors``
            in :mod:`zarr.core.config`.
            Use ``None`` to omit default compressors.

            For Zarr format 2, a "compressor" can be any numcodecs codec. Only a single compressor may
            be provided for Zarr format 2.
            If no ``compressor`` is provided, a default compressor will be used.
            in :mod:`zarr.core.config`.
            Use ``None`` to omit the default compressor.
        compressor : Codec, optional
            Deprecated in favor of ``compressors``.
        serializer : dict[str, JSON] | ArrayBytesCodec, optional
            Array-to-bytes codec to use for encoding the array data.
            Zarr format 3 only. Zarr format 2 arrays use implicit array-to-bytes conversion.
            If no ``serializer`` is provided, a default serializer will be used.
            These defaults can be changed by modifying the value of ``array.v3_default_serializer``
            in :mod:`zarr.core.config`.
        fill_value : Any, optional
            Fill value for the array.
        order : {"C", "F"}, optional
            The memory of the array (default is "C").
            For Zarr format 2, this parameter sets the memory order of the array.
            For Zarr format 3, this parameter is deprecated, because memory order
            is a runtime parameter for Zarr format 3 arrays. The recommended way to specify the memory
            order for Zarr format 3 arrays is via the ``config`` parameter, e.g. ``{'config': 'C'}``.
            If no ``order`` is provided, a default order will be used.
            This default can be changed by modifying the value of ``array.order`` in :mod:`zarr.core.config`.
        attributes : dict, optional
            Attributes for the array.
        chunk_key_encoding : ChunkKeyEncoding, optional
            A specification of how the chunk keys are represented in storage.
            For Zarr format 3, the default is ``{"name": "default", "separator": "/"}}``.
            For Zarr format 2, the default is ``{"name": "v2", "separator": "."}}``.
        dimension_names : Iterable[str], optional
            The names of the dimensions (default is None).
            Zarr format 3 only. Zarr format 2 arrays should not use this parameter.
        storage_options : dict, optional
            If using an fsspec URL to create the store, these will be passed to the backend implementation.
            Ignored otherwise.
        overwrite : bool, default False
            Whether to overwrite an array with the same name in the store, if one exists.
        config : ArrayConfig or ArrayConfigLike, optional
            Runtime configuration for the array.

        Returns
        -------
        AsyncArray

        """
        compressors = _parse_deprecated_compressor(
            compressor, compressors, zarr_format=self.metadata.zarr_format
        )
        return await create_array(
            store=self.store_path,
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            shards=shards,
            filters=filters,
            compressors=compressors,
            serializer=serializer,
            fill_value=fill_value,
            order=order,
            zarr_format=self.metadata.zarr_format,
            attributes=attributes,
            chunk_key_encoding=chunk_key_encoding,
            dimension_names=dimension_names,
            storage_options=storage_options,
            overwrite=overwrite,
            config=config,
        )

    @deprecated("Use AsyncGroup.create_array instead.")
    async def create_dataset(
        self, name: str, *, shape: ShapeLike, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an array.

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `AsyncGroup.create_array` instead.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.AsyncGroup.require_dataset` method.

        Parameters
        ----------
        name : str
            Array name.
        **kwargs : dict
            Additional arguments passed to :func:`zarr.AsyncGroup.create_array`.

        Returns
        -------
        a : AsyncArray
        """
        data = kwargs.pop("data", None)
        # create_dataset in zarr 2.x requires shape but not dtype if data is
        # provided. Allow this configuration by inferring dtype from data if
        # necessary and passing it to create_array
        if "dtype" not in kwargs and data is not None:
            kwargs["dtype"] = data.dtype
        array = await self.create_array(name, shape=shape, **kwargs)
        if data is not None:
            await array.setitem(slice(None), data)
        return array

    @deprecated("Use AsyncGroup.require_array instead.")
    async def require_dataset(
        self,
        name: str,
        *,
        shape: ChunkCoords,
        dtype: npt.DTypeLike = None,
        exact: bool = False,
        **kwargs: Any,
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Obtain an array, creating if it doesn't exist.

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `AsyncGroup.require_dataset` instead.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.AsyncGroup.create_dataset` method.

        Other `kwargs` are as per :func:`zarr.AsyncGroup.create_dataset`.

        Parameters
        ----------
        name : str
            Array name.
        shape : int or tuple of ints
            Array shape.
        dtype : str or dtype, optional
            NumPy dtype.
        exact : bool, optional
            If True, require `dtype` to match exactly. If false, require
            `dtype` can be cast from array dtype.

        Returns
        -------
        a : AsyncArray
        """
        return await self.require_array(name, shape=shape, dtype=dtype, exact=exact, **kwargs)

    async def require_array(
        self,
        name: str,
        *,
        shape: ShapeLike,
        dtype: npt.DTypeLike = None,
        exact: bool = False,
        **kwargs: Any,
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Obtain an array, creating if it doesn't exist.

        Other `kwargs` are as per :func:`zarr.AsyncGroup.create_dataset`.

        Parameters
        ----------
        name : str
            Array name.
        shape : int or tuple of ints
            Array shape.
        dtype : str or dtype, optional
            NumPy dtype.
        exact : bool, optional
            If True, require `dtype` to match exactly. If false, require
            `dtype` can be cast from array dtype.

        Returns
        -------
        a : AsyncArray
        """
        try:
            ds = await self.getitem(name)
            if not isinstance(ds, AsyncArray):
                raise TypeError(f"Incompatible object ({ds.__class__.__name__}) already exists")

            shape = parse_shapelike(shape)
            if shape != ds.shape:
                raise TypeError(f"Incompatible shape ({ds.shape} vs {shape})")

            dtype = np.dtype(dtype)
            if exact:
                if ds.dtype != dtype:
                    raise TypeError(f"Incompatible dtype ({ds.dtype} vs {dtype})")
            else:
                if not np.can_cast(ds.dtype, dtype):
                    raise TypeError(f"Incompatible dtype ({ds.dtype} vs {dtype})")
        except KeyError:
            ds = await self.create_array(name, shape=shape, dtype=dtype, **kwargs)

        return ds

    async def update_attributes(self, new_attributes: dict[str, Any]) -> AsyncGroup:
        """Update group attributes.

        Parameters
        ----------
        new_attributes : dict
            New attributes to set on the group.

        Returns
        -------
        self : AsyncGroup
        """
        # metadata.attributes is "frozen" so we simply clear and update the dict
        self.metadata.attributes.clear()
        self.metadata.attributes.update(new_attributes)

        # Write new metadata
        await self._save_metadata()

        return self

    def __repr__(self) -> str:
        return f"<AsyncGroup {self.store_path}>"

    async def nmembers(
        self,
        max_depth: int | None = 0,
    ) -> int:
        """Count the number of members in this group.

        Parameters
        ----------
        max_depth : int, default 0
            The maximum number of levels of the hierarchy to include. By
            default, (``max_depth=0``) only immediate children are included. Set
            ``max_depth=None`` to include all nodes, and some positive integer
            to consider children within that many levels of the root Group.

        Returns
        -------
        count : int
        """
        # check if we can use consolidated metadata, which requires that we have non-None
        # consolidated metadata at all points in the hierarchy.
        if self.metadata.consolidated_metadata is not None:
            return len(self.metadata.consolidated_metadata.flattened_metadata)
        # TODO: consider using aioitertools.builtins.sum for this
        # return await aioitertools.builtins.sum((1 async for _ in self.members()), start=0)
        n = 0
        async for _ in self.members(max_depth=max_depth):
            n += 1
        return n

    async def members(
        self,
        max_depth: int | None = 0,
    ) -> AsyncGenerator[
        tuple[str, AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup],
        None,
    ]:
        """
        Returns an AsyncGenerator over the arrays and groups contained in this group.
        This method requires that `store_path.store` supports directory listing.

        The results are not guaranteed to be ordered.

        Parameters
        ----------
        max_depth : int, default 0
            The maximum number of levels of the hierarchy to include. By
            default, (``max_depth=0``) only immediate children are included. Set
            ``max_depth=None`` to include all nodes, and some positive integer
            to consider children within that many levels of the root Group.

        Returns
        -------
        path:
            A string giving the path to the target, relative to the Group ``self``.
        value: AsyncArray or AsyncGroup
            The AsyncArray or AsyncGroup that is a child of ``self``.
        """
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth must be None or >= 0. Got '{max_depth}' instead")
        async for item in self._members(max_depth=max_depth):
            yield item

    def _members_consolidated(
        self, max_depth: int | None, prefix: str = ""
    ) -> Generator[
        tuple[str, AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup],
        None,
    ]:
        consolidated_metadata = self.metadata.consolidated_metadata

        do_recursion = max_depth is None or max_depth > 0

        # we kind of just want the top-level keys.
        if consolidated_metadata is not None:
            for key in consolidated_metadata.metadata:
                obj = self._getitem_consolidated(
                    self.store_path, key, prefix=self.name
                )  # Metadata -> Group/Array
                key = f"{prefix}/{key}".lstrip("/")
                yield key, obj

                if do_recursion and isinstance(obj, AsyncGroup):
                    if max_depth is None:
                        new_depth = None
                    else:
                        new_depth = max_depth - 1
                    yield from obj._members_consolidated(new_depth, prefix=key)

    async def _members(
        self, max_depth: int | None
    ) -> AsyncGenerator[
        tuple[str, AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup], None
    ]:
        skip_keys: tuple[str, ...]
        if self.metadata.zarr_format == 2:
            skip_keys = (".zattrs", ".zgroup", ".zarray", ".zmetadata")
        elif self.metadata.zarr_format == 3:
            skip_keys = ("zarr.json",)
        else:
            raise ValueError(f"Unknown Zarr format: {self.metadata.zarr_format}")

        if self.metadata.consolidated_metadata is not None:
            members = self._members_consolidated(max_depth=max_depth)
            for member in members:
                yield member
            return

        if not self.store_path.store.supports_listing:
            msg = (
                f"The store associated with this group ({type(self.store_path.store)}) "
                "does not support listing, "
                "specifically via the `list_dir` method. "
                "This function requires a store that supports listing."
            )

            raise ValueError(msg)
        # enforce a concurrency limit by passing a semaphore to all the recursive functions
        semaphore = asyncio.Semaphore(config.get("async.concurrency"))
        async for member in _iter_members_deep(
            self, max_depth=max_depth, skip_keys=skip_keys, semaphore=semaphore
        ):
            yield member

    async def keys(self) -> AsyncGenerator[str, None]:
        """Iterate over member names."""
        async for key, _ in self.members():
            yield key

    async def contains(self, member: str) -> bool:
        """Check if a member exists in the group.

        Parameters
        ----------
        member : str
            Member name.

        Returns
        -------
        bool
        """
        # TODO: this can be made more efficient.
        try:
            await self.getitem(member)
        except KeyError:
            return False
        else:
            return True

    async def groups(self) -> AsyncGenerator[tuple[str, AsyncGroup], None]:
        """Iterate over subgroups."""
        async for name, value in self.members():
            if isinstance(value, AsyncGroup):
                yield name, value

    async def group_keys(self) -> AsyncGenerator[str, None]:
        """Iterate over group names."""
        async for key, _ in self.groups():
            yield key

    async def group_values(self) -> AsyncGenerator[AsyncGroup, None]:
        """Iterate over group values."""
        async for _, group in self.groups():
            yield group

    async def arrays(
        self,
    ) -> AsyncGenerator[
        tuple[str, AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]], None
    ]:
        """Iterate over arrays."""
        async for key, value in self.members():
            if isinstance(value, AsyncArray):
                yield key, value

    async def array_keys(self) -> AsyncGenerator[str, None]:
        """Iterate over array names."""
        async for key, _ in self.arrays():
            yield key

    async def array_values(
        self,
    ) -> AsyncGenerator[AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata], None]:
        """Iterate over array values."""
        async for _, array in self.arrays():
            yield array

    async def tree(self, expand: bool | None = None, level: int | None = None) -> Any:
        """
        Return a tree-like representation of a hierarchy.

        This requires the optional ``rich`` dependency.

        Parameters
        ----------
        expand : bool, optional
            This keyword is not yet supported. A NotImplementedError is raised if
            it's used.
        level : int, optional
            The maximum depth below this Group to display in the tree.

        Returns
        -------
        TreeRepr
            A pretty-printable object displaying the hierarchy.
        """
        from zarr.core._tree import group_tree_async

        if expand is not None:
            raise NotImplementedError("'expand' is not yet implemented.")
        return await group_tree_async(self, max_depth=level)

    async def empty(
        self, *, name: str, shape: ChunkCoords, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an empty array in this Group.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Notes
        -----
        The contents of an empty Zarr array are not defined. On attempting to
        retrieve data from an empty Zarr array, any values may be returned,
        and these are not guaranteed to be stable from one access to the next.
        """

        return await empty(shape=shape, store=self.store_path, path=name, **kwargs)

    async def zeros(
        self, *, name: str, shape: ChunkCoords, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an array, with zero being used as the default value for uninitialized portions of the array.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await zeros(shape=shape, store=self.store_path, path=name, **kwargs)

    async def ones(
        self, *, name: str, shape: ChunkCoords, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an array, with one being used as the default value for uninitialized portions of the array.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await ones(shape=shape, store=self.store_path, path=name, **kwargs)

    async def full(
        self, *, name: str, shape: ChunkCoords, fill_value: Any | None, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an array, with "fill_value" being used as the default value for uninitialized portions of the array.

        Parameters
        ----------
        name : str
            Name of the array.
        shape : int or tuple of int
            Shape of the empty array.
        fill_value : scalar
            Value to fill the array with.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await full(
            shape=shape,
            fill_value=fill_value,
            store=self.store_path,
            path=name,
            **kwargs,
        )

    async def empty_like(
        self, *, name: str, data: ArrayLike, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create an empty sub-array like `data`.

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create an empty array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await empty_like(a=data, store=self.store_path, path=name, **kwargs)

    async def zeros_like(
        self, *, name: str, data: ArrayLike, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create a sub-array of zeros like `data`.

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create the new array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await zeros_like(a=data, store=self.store_path, path=name, **kwargs)

    async def ones_like(
        self, *, name: str, data: ArrayLike, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create a sub-array of ones like `data`.

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create the new array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await ones_like(a=data, store=self.store_path, path=name, **kwargs)

    async def full_like(
        self, *, name: str, data: ArrayLike, **kwargs: Any
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Create a sub-array like `data` filled with the `fill_value` of `data` .

        Parameters
        ----------
        name : str
            Name of the array.
        data : array-like
            The array to create the new array like.
        **kwargs
            Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

        Returns
        -------
        AsyncArray
            The new array.
        """
        return await full_like(a=data, store=self.store_path, path=name, **kwargs)

    async def move(self, source: str, dest: str) -> None:
        """Move a sub-group or sub-array from one path to another.

        Notes
        -----
        Not implemented
        """
        raise NotImplementedError


async def _getitem_semaphore(
    node: AsyncGroup, key: str, semaphore: asyncio.Semaphore | None
) -> AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup:
    """
    Combine node.getitem with an optional semaphore. If the semaphore parameter is an
    asyncio.Semaphore instance, then the getitem operation is performed inside an async context
    manager provided by that semaphore. If the semaphore parameter is None, then getitem is invoked
    without a context manager.
    """
    if semaphore is not None:
        async with semaphore:
            return await node.getitem(key)
    else:
        return await node.getitem(key)


async def _iter_members(
    node: AsyncGroup,
    skip_keys: tuple[str, ...],
    semaphore: asyncio.Semaphore | None,
) -> AsyncGenerator[
    tuple[str, AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup], None
]:
    """
    Iterate over the arrays and groups contained in a group.

    Parameters
    ----------
    node : AsyncGroup
        The group to traverse.
    skip_keys : tuple[str, ...]
        A tuple of keys to skip when iterating over the possible members of the group.
    semaphore : asyncio.Semaphore | None
        An optional semaphore to use for concurrency control.

    Yields
    ------
    tuple[str, AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata] | AsyncGroup]
    """

    # retrieve keys from storage
    keys = [key async for key in node.store.list_dir(node.path)]
    keys_filtered = tuple(filter(lambda v: v not in skip_keys, keys))

    node_tasks = tuple(
        asyncio.create_task(_getitem_semaphore(node, key, semaphore), name=key)
        for key in keys_filtered
    )

    for fetched_node_coro in asyncio.as_completed(node_tasks):
        try:
            fetched_node = await fetched_node_coro
        except KeyError as e:
            # keyerror is raised when `key` names an object (in the object storage sense),
            # as opposed to a prefix, in the store under the prefix associated with this group
            # in which case `key` cannot be the name of a sub-array or sub-group.
            warnings.warn(
                f"Object at {e.args[0]} is not recognized as a component of a Zarr hierarchy.",
                UserWarning,
                stacklevel=1,
            )
            continue
        match fetched_node:
            case AsyncArray() | AsyncGroup():
                yield fetched_node.basename, fetched_node
            case _:
                raise ValueError(f"Unexpected type: {type(fetched_node)}")


def parse_zarr_format(data: Any) -> ZarrFormat:
    """Parse the zarr_format field from metadata."""
    if data in (2, 3):
        return cast(ZarrFormat, data)
    msg = f"Invalid zarr_format. Expected one of 2 or 3. Got {data}."
    raise ValueError(msg)


def parse_node_type(data: Any) -> NodeType:
    """Parse the node_type field from metadata."""
    if data in ("array", "group"):
        return cast(Literal["array", "group"], data)
    raise MetadataValidationError("node_type", "array or group", data)


# todo: convert None to empty dict
def parse_attributes(data: Any) -> dict[str, Any]:
    """Parse the attributes field from metadata."""
    if data is None:
        return {}
    elif isinstance(data, dict) and all(isinstance(k, str) for k in data):
        return data
    msg = f"Expected dict with string keys. Got {type(data)} instead."
    raise TypeError(msg)


async def group(
    *,  # Note: this is a change from v2
    store: StoreLike | None = None,
    overwrite: bool = False,
    chunk_store: StoreLike | None = None,  # not used
    cache_attrs: bool | None = None,  # not used, default changed
    synchronizer: Any | None = None,  # not used
    path: str | None = None,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used
    attributes: dict[str, JSON] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> AsyncGroup:
    """Create a group.

    Parameters
    ----------
    store : Store or str, optional
        Store or path to directory in file system.
    overwrite : bool, optional
        If True, delete any pre-existing data in `store` at `path` before
        creating the group.
    chunk_store : Store, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    cache_attrs : bool, optional
        If True (default), user attributes will be cached for attribute read
        operations. If False, user attributes are reloaded from the store prior
        to all attribute read operations.
    synchronizer : object, optional
        Array synchronizer.
    path : str, optional
        Group path within store.
    meta_array : array-like, optional
        An array instance to use for determining arrays to create and return
        to users. Use `numpy.empty(())` by default.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.

    Returns
    -------
    g : group
        The new group.
    """

    zarr_format = _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)

    mode: AccessModeLiteral
    if overwrite:
        mode = "w"
    else:
        mode = "r+"
    store_path = await make_store_path(store, path=path, mode=mode, storage_options=storage_options)

    if chunk_store is not None:
        warnings.warn("chunk_store is not yet implemented", RuntimeWarning, stacklevel=2)
    if cache_attrs is not None:
        warnings.warn("cache_attrs is not yet implemented", RuntimeWarning, stacklevel=2)
    if synchronizer is not None:
        warnings.warn("synchronizer is not yet implemented", RuntimeWarning, stacklevel=2)
    if meta_array is not None:
        warnings.warn("meta_array is not yet implemented", RuntimeWarning, stacklevel=2)

    if attributes is None:
        attributes = {}

    try:
        return await AsyncGroup.open(store=store_path, zarr_format=zarr_format)
    except (KeyError, FileNotFoundError):
        _zarr_format = zarr_format or _default_zarr_format()
        return await AsyncGroup.from_store(
            store=store_path,
            zarr_format=_zarr_format,
            overwrite=overwrite,
            attributes=attributes,
        )


async def create_group(
    *,
    store: StoreLike,
    path: str | None = None,
    overwrite: bool = False,
    zarr_format: ZarrFormat | None = None,
    attributes: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> AsyncGroup:
    """Create a group.

    Parameters
    ----------
    store : Store or str
        Store or path to directory in file system.
    path : str, optional
        Group path within store.
    overwrite : bool, optional
        If True, pre-existing data at ``path`` will be deleted before
        creating the group.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
        If no ``zarr_format`` is provided, the default format will be used.
        This default can be changed by modifying the value of ``default_zarr_format``
        in :mod:`zarr.core.config`.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.

    Returns
    -------
    AsyncGroup
        The new group.
    """

    if zarr_format is None:
        zarr_format = _default_zarr_format()

    mode: Literal["a"] = "a"

    store_path = await make_store_path(store, path=path, mode=mode, storage_options=storage_options)

    return await AsyncGroup.from_store(
        store=store_path,
        zarr_format=zarr_format,
        overwrite=overwrite,
        attributes=attributes,
    )


async def open_group(
    store: StoreLike | None = None,
    *,  # Note: this is a change from v2
    mode: AccessModeLiteral = "a",
    cache_attrs: bool | None = None,  # not used, default changed
    synchronizer: Any = None,  # not used
    path: str | None = None,
    chunk_store: StoreLike | None = None,  # not used
    storage_options: dict[str, Any] | None = None,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used
    attributes: dict[str, JSON] | None = None,
    use_consolidated: bool | str | None = None,
) -> AsyncGroup:
    """Open a group using file-mode-like semantics.

    Parameters
    ----------
    store : Store, str, or mapping, optional
        Store or path to directory in file system or name of zip file.

        Strings are interpreted as paths on the local file system
        and used as the ``root`` argument to :class:`zarr.storage.LocalStore`.

        Dictionaries are used as the ``store_dict`` argument in
        :class:`zarr.storage.MemoryStore``.

        By default (``store=None``) a new :class:`zarr.storage.MemoryStore`
        is created.

    mode : {'r', 'r+', 'a', 'w', 'w-'}, optional
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    cache_attrs : bool, optional
        If True (default), user attributes will be cached for attribute read
        operations. If False, user attributes are reloaded from the store prior
        to all attribute read operations.
    synchronizer : object, optional
        Array synchronizer.
    path : str, optional
        Group path within store.
    chunk_store : Store or str, optional
        Store or path to directory in file system or name of zip file.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    meta_array : array-like, optional
        An array instance to use for determining arrays to create and return
        to users. Use `numpy.empty(())` by default.
    attributes : dict
        A dictionary of JSON-serializable values with user-defined attributes.
    use_consolidated : bool or str, default None
        Whether to use consolidated metadata.

        By default, consolidated metadata is used if it's present in the
        store (in the ``zarr.json`` for Zarr format 3 and in the ``.zmetadata`` file
        for Zarr format 2).

        To explicitly require consolidated metadata, set ``use_consolidated=True``,
        which will raise an exception if consolidated metadata is not found.

        To explicitly *not* use consolidated metadata, set ``use_consolidated=False``,
        which will fall back to using the regular, non consolidated metadata.

        Zarr format 2 allowed configuring the key storing the consolidated metadata
        (``.zmetadata`` by default). Specify the custom key as ``use_consolidated``
        to load consolidated metadata from a non-default key.

    Returns
    -------
    g : group
        The new group.
    """

    zarr_format = _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)

    if cache_attrs is not None:
        warnings.warn("cache_attrs is not yet implemented", RuntimeWarning, stacklevel=2)
    if synchronizer is not None:
        warnings.warn("synchronizer is not yet implemented", RuntimeWarning, stacklevel=2)
    if meta_array is not None:
        warnings.warn("meta_array is not yet implemented", RuntimeWarning, stacklevel=2)
    if chunk_store is not None:
        warnings.warn("chunk_store is not yet implemented", RuntimeWarning, stacklevel=2)

    store_path = await make_store_path(store, mode=mode, storage_options=storage_options, path=path)

    if attributes is None:
        attributes = {}

    try:
        if mode in _READ_MODES:
            return await AsyncGroup.open(
                store_path, zarr_format=zarr_format, use_consolidated=use_consolidated
            )
    except (KeyError, FileNotFoundError):
        pass
    if mode in _CREATE_MODES:
        overwrite = _infer_overwrite(mode)
        _zarr_format = zarr_format or _default_zarr_format()
        return await AsyncGroup.from_store(
            store_path,
            zarr_format=_zarr_format,
            overwrite=overwrite,
            attributes=attributes,
        )
    raise FileNotFoundError(f"Unable to find group: {store_path}")