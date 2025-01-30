from __future__ import annotations

from zarr._compat import _deprecate_positional_args
from zarr.abc.store import Store, set_or_delete
from zarr.core.array import Array, AsyncArray, _parse_deprecated_compressor
from zarr.core.array_spec import ArrayConfig, ArrayConfigLike
from zarr.core.attributes import Attributes
from zarr.core.buffer import NDArrayLike, default_buffer_prototype
from zarr.core.chunk_key_encodings import ChunkKeyEncoding
from zarr.core.common import ChunkCoords
from zarr.core.group import _parse_async_node
from zarr.core.group.core import AsyncGroup
from zarr.core.group.metadata import GroupMetadata
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import SyncMixin, sync
from zarr.storage import StorePath


import asyncio
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal, overload
from typing_extensions import deprecated

@dataclass(frozen=True)
class Group(SyncMixin):
    _async_group: AsyncGroup

    @classmethod
    def from_store(
        cls,
        store: StoreLike,
        *,
        attributes: dict[str, Any] | None = None,
        zarr_format: ZarrFormat = 3,
        overwrite: bool = False,
    ) -> Group:
        """Instantiate a group from an initialized store.

        Parameters
        ----------
        store : StoreLike
            StoreLike containing the Group.
        attributes : dict, optional
            A dictionary of JSON-serializable values with user-defined attributes.
        zarr_format : {2, 3}, optional
            Zarr storage format version.
        overwrite : bool, optional
            If True, do not raise an error if the group already exists.

        Returns
        -------
        Group
            Group instantiated from the store.

        Raises
        ------
        ContainsArrayError, ContainsGroupError, ContainsArrayAndGroupError
        """
        attributes = attributes or {}
        obj = sync(
            AsyncGroup.from_store(
                store,
                attributes=attributes,
                overwrite=overwrite,
                zarr_format=zarr_format,
            ),
        )

        return cls(obj)

    @classmethod
    def open(
        cls,
        store: StoreLike,
        zarr_format: ZarrFormat | None = 3,
    ) -> Group:
        """Open a group from an initialized store.

        Parameters
        ----------
        store : StoreLike
            Store containing the Group.
        zarr_format : {2, 3, None}, optional
            Zarr storage format version.

        Returns
        -------
        Group
            Group instantiated from the store.
        """
        obj = sync(AsyncGroup.open(store, zarr_format=zarr_format))
        return cls(obj)

    def __getitem__(self, path: str) -> Array | Group:
        """Obtain a group member.

        Parameters
        ----------
        path : str
            Group member name.

        Returns
        -------
        Array | Group
            Group member (Array or Group) at the specified key

        Examples
        --------
        >>> import zarr
        >>> group = Group.from_store(zarr.storage.MemoryStore()
        >>> group.create_array(name="subarray", shape=(10,), chunks=(10,))
        >>> group.create_group(name="subgroup").create_array(name="subarray", shape=(10,), chunks=(10,))
        >>> group["subarray"]
        <Array memory://132270269438272/subarray shape=(10,) dtype=float64>
        >>> group["subgroup"]
        <Group memory://132270269438272/subgroup>
        >>> group["subgroup"]["subarray"]
        <Array memory://132270269438272/subgroup/subarray shape=(10,) dtype=float64>

        """
        obj = self._sync(self._async_group.getitem(path))
        if isinstance(obj, AsyncArray):
            return Array(obj)
        else:
            return Group(obj)

    def get(self, path: str, default: DefaultT | None = None) -> Array | Group | DefaultT | None:
        """Obtain a group member, returning default if not found.

        Parameters
        ----------
        path : str
            Group member name.
        default : object
            Default value to return if key is not found (default: None).

        Returns
        -------
        object
            Group member (Array or Group) or default if not found.

        Examples
        --------
        >>> import zarr
        >>> group = Group.from_store(zarr.storage.MemoryStore()
        >>> group.create_array(name="subarray", shape=(10,), chunks=(10,))
        >>> group.create_group(name="subgroup")
        >>> group.get("subarray")
        <Array memory://132270269438272/subarray shape=(10,) dtype=float64>
        >>> group.get("subgroup")
        <Group memory://132270269438272/subgroup>
        >>> group.get("nonexistent", None)

        """
        try:
            return self[path]
        except KeyError:
            return default

    def __delitem__(self, key: str) -> None:
        """Delete a group member.

        Parameters
        ----------
        key : str
            Group member name.

        Examples
        --------
        >>> import zarr
        >>> group = Group.from_store(zarr.storage.MemoryStore()
        >>> group.create_array(name="subarray", shape=(10,), chunks=(10,))
        >>> del group["subarray"]
        >>> "subarray" in group
        False
        """
        self._sync(self._async_group.delitem(key))

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over group member names.
        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> d1 = g1.create_array('baz', shape=(10,), chunks=(10,))
        >>> d2 = g1.create_array('quux', shape=(10,), chunks=(10,))
        >>> for name in g1:
        ...     print(name)
        baz
        bar
        foo
        quux
        """
        yield from self.keys()

    def __len__(self) -> int:
        """Number of members."""
        return self.nmembers()

    def __setitem__(self, key: str, value: Any) -> None:
        """Fastpath for creating a new array.

        New arrays will be created using default settings for the array type.
        If you need to create an array with custom settings, use the `create_array` method.

        Parameters
        ----------
        key : str
            Array name.
        value : Any
            Array data.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group["foo"] = zarr.zeros((10,))
        >>> group["foo"]
        <Array memory://132270269438272/foo shape=(10,) dtype=float64>
        """
        self._sync(self._async_group.setitem(key, value))

    def __repr__(self) -> str:
        return f"<Group {self.store_path}>"

    async def update_attributes_async(self, new_attributes: dict[str, Any]) -> Group:
        """Update the attributes of this group.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> await group.update_attributes_async({"foo": "bar"})
        >>> group.attrs.asdict()
        {'foo': 'bar'}
        """
        new_metadata = replace(self.metadata, attributes=new_attributes)

        # Write new metadata
        to_save = new_metadata.to_buffer_dict(default_buffer_prototype())
        awaitables = [set_or_delete(self.store_path / key, value) for key, value in to_save.items()]
        await asyncio.gather(*awaitables)

        async_group = replace(self._async_group, metadata=new_metadata)
        return replace(self, _async_group=async_group)

    @property
    def store_path(self) -> StorePath:
        """Path-like interface for the Store."""
        return self._async_group.store_path

    @property
    def metadata(self) -> GroupMetadata:
        """Group metadata."""
        return self._async_group.metadata

    @property
    def path(self) -> str:
        """Storage path."""
        return self._async_group.path

    @property
    def name(self) -> str:
        """Group name following h5py convention."""
        return self._async_group.name

    @property
    def basename(self) -> str:
        """Final component of name."""
        return self._async_group.basename

    @property
    def attrs(self) -> Attributes:
        """Attributes of this Group"""
        return Attributes(self)

    @property
    def info(self) -> Any:
        """
        Return the statically known information for a group.

        Returns
        -------
        GroupInfo

        See Also
        --------
        Group.info_complete
            All information about a group, including dynamic information
            like the children members.
        """
        return self._async_group.info

    def info_complete(self) -> Any:
        """
        Return information for a group.

        If this group doesn't contain consolidated metadata then
        this will need to read from the backing Store.

        Returns
        -------
        GroupInfo

        See Also
        --------
        Group.info
        """
        return self._sync(self._async_group.info_complete())

    @property
    def store(self) -> Store:
        # Backwards compatibility for 2.x
        return self._async_group.store

    @property
    def read_only(self) -> bool:
        # Backwards compatibility for 2.x
        return self._async_group.read_only

    @property
    def synchronizer(self) -> None:
        # Backwards compatibility for 2.x
        # Not implemented in 3.x yet.
        return self._async_group.synchronizer

    def update_attributes(self, new_attributes: dict[str, Any]) -> Group:
        """Update the attributes of this group.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.update_attributes({"foo": "bar"})
        >>> group.attrs.asdict()
        {'foo': 'bar'}
        """
        self._sync(self._async_group.update_attributes(new_attributes))
        return self

    def nmembers(self, max_depth: int | None = 0) -> int:
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

        return self._sync(self._async_group.nmembers(max_depth=max_depth))

    def members(self, max_depth: int | None = 0) -> tuple[tuple[str, Array | Group], ...]:
        """
        Return the sub-arrays and sub-groups of this group as a tuple of (name, array | group)
        pairs
        """
        _members = self._sync_iter(self._async_group.members(max_depth=max_depth))

        return tuple((kv[0], _parse_async_node(kv[1])) for kv in _members)

    def keys(self) -> Generator[str, None]:
        """Return an iterator over group member names.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> d1 = g1.create_array('baz', shape=(10,), chunks=(10,))
        >>> d2 = g1.create_array('quux', shape=(10,), chunks=(10,))
        >>> for name in g1.keys():
        ...     print(name)
        baz
        bar
        foo
        quux
        """
        yield from self._sync_iter(self._async_group.keys())

    def __contains__(self, member: str) -> bool:
        """Test for group membership.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> d1 = g1.create_array('bar', shape=(10,), chunks=(10,))
        >>> 'foo' in g1
        True
        >>> 'bar' in g1
        True
        >>> 'baz' in g1
        False

        """
        return self._sync(self._async_group.contains(member))

    def groups(self) -> Generator[tuple[str, Group], None]:
        """Return the sub-groups of this group as a generator of (name, group) pairs.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_group("subgroup")
        >>> for name, subgroup in group.groups():
        ...     print(name, subgroup)
        subgroup <Group memory://132270269438272/subgroup>
        """
        for name, async_group in self._sync_iter(self._async_group.groups()):
            yield name, Group(async_group)

    def group_keys(self) -> Generator[str, None]:
        """Return an iterator over group member names.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_group("subgroup")
        >>> for name in group.group_keys():
        ...     print(name)
        subgroup
        """
        for name, _ in self.groups():
            yield name

    def group_values(self) -> Generator[Group, None]:
        """Return an iterator over group members.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_group("subgroup")
        >>> for subgroup in group.group_values():
        ...     print(subgroup)
        <Group memory://132270269438272/subgroup>
        """
        for _, group in self.groups():
            yield group

    def arrays(self) -> Generator[tuple[str, Array], None]:
        """Return the sub-arrays of this group as a generator of (name, array) pairs

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_array("subarray", shape=(10,), chunks=(10,))
        >>> for name, subarray in group.arrays():
        ...     print(name, subarray)
        subarray <Array memory://140198565357056/subarray shape=(10,) dtype=float64>
        """
        for name, async_array in self._sync_iter(self._async_group.arrays()):
            yield name, Array(async_array)

    def array_keys(self) -> Generator[str, None]:
        """Return an iterator over group member names.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_array("subarray", shape=(10,), chunks=(10,))
        >>> for name in group.array_keys():
        ...     print(name)
        subarray
        """

        for name, _ in self.arrays():
            yield name

    def array_values(self) -> Generator[Array, None]:
        """Return an iterator over group members.

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> group.create_array("subarray", shape=(10,), chunks=(10,))
        >>> for subarray in group.array_values():
        ...     print(subarray)
        <Array memory://140198565357056/subarray shape=(10,) dtype=float64>
        """
        for _, array in self.arrays():
            yield array

    def tree(self, expand: bool | None = None, level: int | None = None) -> Any:
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
        return self._sync(self._async_group.tree(expand=expand, level=level))

    def create_group(self, name: str, **kwargs: Any) -> Group:
        """Create a sub-group.

        Parameters
        ----------
        name : str
            Name of the new subgroup.

        Returns
        -------
        Group

        Examples
        --------
        >>> import zarr
        >>> group = zarr.group()
        >>> subgroup = group.create_group("subgroup")
        >>> subgroup
        <Group memory://132270269438272/subgroup>
        """
        return Group(self._sync(self._async_group.create_group(name, **kwargs)))

    def require_group(self, name: str, **kwargs: Any) -> Group:
        """Obtain a sub-group, creating one if it doesn't exist.

        Parameters
        ----------
        name : str
            Group name.

        Returns
        -------
        g : Group
        """
        return Group(self._sync(self._async_group.require_group(name, **kwargs)))

    def require_groups(self, *names: str) -> tuple[Group, ...]:
        """Convenience method to require multiple groups in a single call.

        Parameters
        ----------
        *names : str
            Group names.

        Returns
        -------
        groups : tuple of Groups
        """
        return tuple(map(Group, self._sync(self._async_group.require_groups(*names))))

    def create(self, *args: Any, **kwargs: Any) -> Array:
        # Backwards compatibility for 2.x
        return self.create_array(*args, **kwargs)

    @_deprecate_positional_args
    def create_array(
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
        order: MemoryOrder | None = "C",
        attributes: dict[str, JSON] | None = None,
        chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None = None,
        dimension_names: Iterable[str] | None = None,
        storage_options: dict[str, Any] | None = None,
        overwrite: bool = False,
        config: ArrayConfig | ArrayConfigLike | None = None,
    ) -> Array:
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
        return Array(
            self._sync(
                self._async_group.create_array(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks,
                    shards=shards,
                    fill_value=fill_value,
                    attributes=attributes,
                    chunk_key_encoding=chunk_key_encoding,
                    compressors=compressors,
                    serializer=serializer,
                    dimension_names=dimension_names,
                    order=order,
                    filters=filters,
                    overwrite=overwrite,
                    storage_options=storage_options,
                    config=config,
                )
            )
        )

    @deprecated("Use Group.create_array instead.")
    def create_dataset(self, name: str, **kwargs: Any) -> Array:
        """Create an array.

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `Group.create_array` instead.


        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.Group.require_dataset` method.

        Parameters
        ----------
        name : str
            Array name.
        **kwargs : dict
            Additional arguments passed to :func:`zarr.Group.create_array`

        Returns
        -------
        a : Array
        """
        return Array(self._sync(self._async_group.create_dataset(name, **kwargs)))

    @deprecated("Use Group.require_array instead.")
    def require_dataset(self, name: str, *, shape: ShapeLike, **kwargs: Any) -> Array:
        """Obtain an array, creating if it doesn't exist.

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `Group.require_array` instead.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.Group.create_dataset` method.

        Other `kwargs` are as per :func:`zarr.Group.create_dataset`.

        Parameters
        ----------
        name : str
            Array name.
        **kwargs :
            See :func:`zarr.Group.create_dataset`.

        Returns
        -------
        a : Array
        """
        return Array(self._sync(self._async_group.require_array(name, shape=shape, **kwargs)))

    def require_array(self, name: str, *, shape: ShapeLike, **kwargs: Any) -> Array:
        """Obtain an array, creating if it doesn't exist.

        Other `kwargs` are as per :func:`zarr.Group.create_array`.

        Parameters
        ----------
        name : str
            Array name.
        **kwargs :
            See :func:`zarr.Group.create_array`.

        Returns
        -------
        a : Array
        """
        return Array(self._sync(self._async_group.require_array(name, shape=shape, **kwargs)))

    @_deprecate_positional_args
    def empty(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> Array:
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
        return Array(self._sync(self._async_group.empty(name=name, shape=shape, **kwargs)))

    @_deprecate_positional_args
    def zeros(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> Array:
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
        Array
            The new array.
        """
        return Array(self._sync(self._async_group.zeros(name=name, shape=shape, **kwargs)))

    @_deprecate_positional_args
    def ones(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> Array:
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
        Array
            The new array.
        """
        return Array(self._sync(self._async_group.ones(name=name, shape=shape, **kwargs)))

    @_deprecate_positional_args
    def full(
        self, *, name: str, shape: ChunkCoords, fill_value: Any | None, **kwargs: Any
    ) -> Array:
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
        Array
            The new array.
        """
        return Array(
            self._sync(
                self._async_group.full(name=name, shape=shape, fill_value=fill_value, **kwargs)
            )
        )

    @_deprecate_positional_args
    def empty_like(self, *, name: str, data: ArrayLike, **kwargs: Any) -> Array:
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
        Array
            The new array.
        """
        return Array(self._sync(self._async_group.empty_like(name=name, data=data, **kwargs)))

    @_deprecate_positional_args
    def zeros_like(self, *, name: str, data: ArrayLike, **kwargs: Any) -> Array:
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
        Array
            The new array.
        """

        return Array(self._sync(self._async_group.zeros_like(name=name, data=data, **kwargs)))

    @_deprecate_positional_args
    def ones_like(self, *, name: str, data: ArrayLike, **kwargs: Any) -> Array:
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
        Array
            The new array.
        """
        return Array(self._sync(self._async_group.ones_like(name=name, data=data, **kwargs)))

    @_deprecate_positional_args
    def full_like(self, *, name: str, data: ArrayLike, **kwargs: Any) -> Array:
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
        Array
            The new array.
        """
        return Array(self._sync(self._async_group.full_like(name=name, data=data, **kwargs)))

    def move(self, source: str, dest: str) -> None:
        """Move a sub-group or sub-array from one path to another.

        Notes
        -----
        Not implemented
        """
        return self._sync(self._async_group.move(source, dest))

    @deprecated("Use Group.create_array instead.")
    @_deprecate_positional_args
    def array(
        self,
        name: str,
        *,
        shape: ShapeLike,
        dtype: npt.DTypeLike,
        chunks: ChunkCoords | Literal["auto"] = "auto",
        shards: ChunkCoords | Literal["auto"] | None = None,
        filters: FiltersLike = "auto",
        compressors: CompressorsLike = "auto",
        compressor: CompressorLike = None,
        serializer: SerializerLike = "auto",
        fill_value: Any | None = 0,
        order: MemoryOrder | None = "C",
        attributes: dict[str, JSON] | None = None,
        chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None = None,
        dimension_names: Iterable[str] | None = None,
        storage_options: dict[str, Any] | None = None,
        overwrite: bool = False,
        config: ArrayConfig | ArrayConfigLike | None = None,
        data: npt.ArrayLike | None = None,
    ) -> Array:
        """Create an array within this group.

        .. deprecated:: 3.0.0
            Use `Group.create_array` instead.

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
        data : array_like
            The data to fill the array with.

        Returns
        -------
        AsyncArray
        """
        compressors = _parse_deprecated_compressor(compressor, compressors)
        return Array(
            self._sync(
                self._async_group.create_dataset(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks,
                    shards=shards,
                    fill_value=fill_value,
                    attributes=attributes,
                    chunk_key_encoding=chunk_key_encoding,
                    compressors=compressors,
                    serializer=serializer,
                    dimension_names=dimension_names,
                    order=order,
                    filters=filters,
                    overwrite=overwrite,
                    storage_options=storage_options,
                    config=config,
                    data=data,
                )
            )
        )


def consolidate_metadata(
    store: StoreLike,
    path: str | None = None,
    zarr_format: ZarrFormat | None = None,
) -> Group:
    """
    Consolidate the metadata of all nodes in a hierarchy.

    Upon completion, the metadata of the root node in the Zarr hierarchy will be
    updated to include all the metadata of child nodes.

    Parameters
    ----------
    store : StoreLike
        The store-like object whose metadata you wish to consolidate.
    path : str, optional
        A path to a group in the store to consolidate at. Only children
        below that group will be consolidated.

        By default, the root node is used so all the metadata in the
        store is consolidated.
    zarr_format : {2, 3, None}, optional
        The zarr format of the hierarchy. By default the zarr format
        is inferred.

    Returns
    -------
    group: Group
        The group, with the ``consolidated_metadata`` field set to include
        the metadata of each child node.
    """
    return Group(sync(consolidate_metadata(store, path=path, zarr_format=zarr_format)))


def save_group(
    store: StoreLike,
    *args: NDArrayLike,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: NDArrayLike,
) -> None:
    """Save several NumPy arrays to the local file system.

    Follows a similar API to the NumPy savez()/savez_compressed() functions.

    Parameters
    ----------
    store : Store or str
        Store or path to directory in file system or name of zip file.
    *args : ndarray
        NumPy arrays with data to save.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        Path within the store where the group will be saved.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    **kwargs
        NumPy arrays with data to save.
    """

    return sync(
        save_group(
            store,
            *args,
            zarr_version=zarr_version,
            zarr_format=zarr_format,
            path=path,
            storage_options=storage_options,
            **kwargs,
        )
    )


@deprecated("Use Group.tree instead.")
def tree(grp: Group, expand: bool | None = None, level: int | None = None) -> Any:
    """Provide a rich display of the hierarchy.

    .. deprecated:: 3.0.0
        `zarr.tree()` is deprecated and will be removed in a future release.
        Use `group.tree()` instead.

    Parameters
    ----------
    grp : Group
        Zarr or h5py group.
    expand : bool, optional
        Only relevant for HTML representation. If True, tree will be fully expanded.
    level : int, optional
        Maximum depth to descend into hierarchy.

    Returns
    -------
    TreeRepr
        A pretty-printable object displaying the hierarchy.
    """
    return sync(tree(grp._async_group, expand=expand, level=level))


@_deprecate_positional_args
def group(
    store: StoreLike | None = None,
    *,
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
) -> Group:
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
    g : Group
        The new group.
    """
    return Group(
        sync(
            group(
                store=store,
                overwrite=overwrite,
                chunk_store=chunk_store,
                cache_attrs=cache_attrs,
                synchronizer=synchronizer,
                path=path,
                zarr_version=zarr_version,
                zarr_format=zarr_format,
                meta_array=meta_array,
                attributes=attributes,
                storage_options=storage_options,
            )
        )
    )


@_deprecate_positional_args
def open_group(
    store: StoreLike | None = None,
    *,
    mode: AccessModeLiteral = "a",
    cache_attrs: bool | None = None,  # default changed, not used in async api
    synchronizer: Any = None,  # not used in async api
    path: str | None = None,
    chunk_store: StoreLike | None = None,  # not used in async api
    storage_options: dict[str, Any] | None = None,  # not used in async api
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used in async api
    attributes: dict[str, JSON] | None = None,
    use_consolidated: bool | str | None = None,
) -> Group:
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

        Zarr format 2 allows configuring the key storing the consolidated metadata
        (``.zmetadata`` by default). Specify the custom key as ``use_consolidated``
        to load consolidated metadata from a non-default key.

    Returns
    -------
    g : Group
        The new group.
    """
    return Group(
        sync(
            open_group(
                store=store,
                mode=mode,
                cache_attrs=cache_attrs,
                synchronizer=synchronizer,
                path=path,
                chunk_store=chunk_store,
                storage_options=storage_options,
                zarr_version=zarr_version,
                zarr_format=zarr_format,
                meta_array=meta_array,
                attributes=attributes,
                use_consolidated=use_consolidated,
            )
        )
    )


def create_group(
    store: StoreLike,
    *,
    path: str | None = None,
    zarr_format: ZarrFormat | None = None,
    overwrite: bool = False,
    attributes: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> Group:
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
    Group
        The new group.
    """
    return Group(
        sync(
            create_group(
                store=store,
                path=path,
                overwrite=overwrite,
                storage_options=storage_options,
                zarr_format=zarr_format,
                attributes=attributes,
            )
        )
    )


@overload
def _parse_async_node(node: AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]) -> Array: ...


@overload
def _parse_async_node(node: AsyncGroup) -> Group: ...


def _parse_async_node(
    node: AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup,
) -> Array | Group:
    """Wrap an AsyncArray in an Array, or an AsyncGroup in a Group."""
    if isinstance(node, AsyncArray):
        return Array(node)
    elif isinstance(node, AsyncGroup):
        return Group(node)
    else:
        raise TypeError(f"Unknown node type, got {type(node)}")