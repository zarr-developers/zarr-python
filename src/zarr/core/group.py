from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import deprecated

import zarr.api.asynchronous as async_api
from zarr.abc.metadata import Metadata
from zarr.abc.store import Store, set_or_delete
from zarr.core.array import Array, AsyncArray, _build_parents
from zarr.core.attributes import Attributes
from zarr.core.buffer import default_buffer_prototype
from zarr.core.common import (
    JSON,
    ZARR_JSON,
    ZARRAY_JSON,
    ZATTRS_JSON,
    ZGROUP_JSON,
    ChunkCoords,
    ShapeLike,
    ZarrFormat,
    parse_shapelike,
)
from zarr.core.config import config
from zarr.core.sync import SyncMixin, sync
from zarr.store import StoreLike, StorePath, make_store_path
from zarr.store.common import ensure_no_existing_node

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Iterable, Iterator
    from typing import Any

    from zarr.abc.codec import Codec
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.chunk_key_encodings import ChunkKeyEncoding

logger = logging.getLogger("zarr.group")


def parse_zarr_format(data: Any) -> ZarrFormat:
    if data in (2, 3):
        return cast(Literal[2, 3], data)
    msg = msg = f"Invalid zarr_format. Expected one 2 or 3. Got {data}."
    raise ValueError(msg)


# todo: convert None to empty dict
def parse_attributes(data: Any) -> dict[str, Any]:
    if data is None:
        return {}
    elif isinstance(data, dict) and all(isinstance(k, str) for k in data):
        return data
    msg = f"Expected dict with string keys. Got {type(data)} instead."
    raise TypeError(msg)


@overload
def _parse_async_node(node: AsyncArray) -> Array: ...


@overload
def _parse_async_node(node: AsyncGroup) -> Group: ...


def _parse_async_node(node: AsyncArray | AsyncGroup) -> Array | Group:
    """
    Wrap an AsyncArray in an Array, or an AsyncGroup in a Group.
    """
    if isinstance(node, AsyncArray):
        return Array(node)
    elif isinstance(node, AsyncGroup):
        return Group(node)
    else:
        raise TypeError(f"Unknown node type, got {type(node)}")


@dataclass(frozen=True)
class GroupMetadata(Metadata):
    attributes: dict[str, Any] = field(default_factory=dict)
    zarr_format: ZarrFormat = 3
    node_type: Literal["group"] = field(default="group", init=False)

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        json_indent = config.get("json_indent")
        if self.zarr_format == 3:
            return {
                ZARR_JSON: prototype.buffer.from_bytes(
                    json.dumps(self.to_dict(), indent=json_indent).encode()
                )
            }
        else:
            return {
                ZGROUP_JSON: prototype.buffer.from_bytes(
                    json.dumps({"zarr_format": self.zarr_format}, indent=json_indent).encode()
                ),
                ZATTRS_JSON: prototype.buffer.from_bytes(
                    json.dumps(self.attributes, indent=json_indent).encode()
                ),
            }

    def __init__(
        self, attributes: dict[str, Any] | None = None, zarr_format: ZarrFormat = 3
    ) -> None:
        attributes_parsed = parse_attributes(attributes)
        zarr_format_parsed = parse_zarr_format(zarr_format)

        object.__setattr__(self, "attributes", attributes_parsed)
        object.__setattr__(self, "zarr_format", zarr_format_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroupMetadata:
        assert data.pop("node_type", None) in ("group", None)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AsyncGroup:
    metadata: GroupMetadata
    store_path: StorePath

    @classmethod
    async def from_store(
        cls,
        store: StoreLike,
        *,
        attributes: dict[str, Any] | None = None,
        exists_ok: bool = False,
        zarr_format: ZarrFormat = 3,
    ) -> AsyncGroup:
        store_path = await make_store_path(store)
        if not exists_ok:
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
        zarr_format: Literal[2, 3, None] = 3,
    ) -> AsyncGroup:
        store_path = await make_store_path(store)

        if zarr_format == 2:
            zgroup_bytes, zattrs_bytes = await asyncio.gather(
                (store_path / ZGROUP_JSON).get(), (store_path / ZATTRS_JSON).get()
            )
            if zgroup_bytes is None:
                raise FileNotFoundError(store_path)
        elif zarr_format == 3:
            zarr_json_bytes = await (store_path / ZARR_JSON).get()
            if zarr_json_bytes is None:
                raise FileNotFoundError(store_path)
        elif zarr_format is None:
            zarr_json_bytes, zgroup_bytes, zattrs_bytes = await asyncio.gather(
                (store_path / ZARR_JSON).get(),
                (store_path / ZGROUP_JSON).get(),
                (store_path / ZATTRS_JSON).get(),
            )
            if zarr_json_bytes is not None and zgroup_bytes is not None:
                # TODO: revisit this exception type
                # alternatively, we could warn and favor v3
                raise ValueError("Both zarr.json and .zgroup objects exist")
            if zarr_json_bytes is None and zgroup_bytes is None:
                raise FileNotFoundError(store_path)
            # set zarr_format based on which keys were found
            if zarr_json_bytes is not None:
                zarr_format = 3
            else:
                zarr_format = 2
        else:
            raise ValueError(f"unexpected zarr_format: {zarr_format}")

        if zarr_format == 2:
            # V2 groups are comprised of a .zgroup and .zattrs objects
            assert zgroup_bytes is not None
            zgroup = json.loads(zgroup_bytes.to_bytes())
            zattrs = json.loads(zattrs_bytes.to_bytes()) if zattrs_bytes is not None else {}
            group_metadata = {**zgroup, "attributes": zattrs}
        else:
            # V3 groups are comprised of a zarr.json object
            assert zarr_json_bytes is not None
            group_metadata = json.loads(zarr_json_bytes.to_bytes())

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

    async def getitem(
        self,
        key: str,
    ) -> AsyncArray | AsyncGroup:
        store_path = self.store_path / key
        logger.debug("key=%s, store_path=%s", key, store_path)

        # Note:
        # in zarr-python v2, we first check if `key` references an Array, else if `key` references
        # a group,using standalone `contains_array` and `contains_group` functions. These functions
        # are reusable, but for v3 they would perform redundant I/O operations.
        # Not clear how much of that strategy we want to keep here.

        if self.metadata.zarr_format == 3:
            zarr_json_bytes = await (store_path / ZARR_JSON).get()
            if zarr_json_bytes is None:
                raise KeyError(key)
            else:
                zarr_json = json.loads(zarr_json_bytes.to_bytes())
            if zarr_json["node_type"] == "group":
                return type(self).from_dict(store_path, zarr_json)
            elif zarr_json["node_type"] == "array":
                return AsyncArray.from_dict(store_path, zarr_json)
            else:
                raise ValueError(f"unexpected node_type: {zarr_json['node_type']}")
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
            # unpack the zattrs, this can be None if no attrs were written
            zattrs = json.loads(zattrs_bytes.to_bytes()) if zattrs_bytes is not None else {}

            if zarray is not None:
                # TODO: update this once the V2 array support is part of the primary array class
                zarr_json = {**zarray, "attributes": zattrs}
                return AsyncArray.from_dict(store_path, zarr_json)
            else:
                zgroup = (
                    json.loads(zgroup_bytes.to_bytes())
                    if zgroup_bytes is not None
                    else {"zarr_format": self.metadata.zarr_format}
                )
                zarr_json = {**zgroup, "attributes": zattrs}
                return type(self).from_dict(store_path, zarr_json)
        else:
            raise ValueError(f"unexpected zarr_format: {self.metadata.zarr_format}")

    async def delitem(self, key: str) -> None:
        store_path = self.store_path / key
        if self.metadata.zarr_format == 3:
            await (store_path / ZARR_JSON).delete()
        elif self.metadata.zarr_format == 2:
            await asyncio.gather(
                (store_path / ZGROUP_JSON).delete(),  # TODO: missing_ok=False
                (store_path / ZARRAY_JSON).delete(),  # TODO: missing_ok=False
                (store_path / ZATTRS_JSON).delete(),  # TODO: missing_ok=True
            )
        else:
            raise ValueError(f"unexpected zarr_format: {self.metadata.zarr_format}")

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
    def info(self) -> None:
        raise NotImplementedError

    @property
    def store(self) -> Store:
        return self.store_path.store

    @property
    def read_only(self) -> bool:
        # Backwards compatibility for 2.x
        return self.store_path.store.mode.readonly

    @property
    def synchronizer(self) -> None:
        # Backwards compatibility for 2.x
        # Not implemented in 3.x yet.
        return None

    async def create_group(
        self,
        name: str,
        *,
        exists_ok: bool = False,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncGroup:
        attributes = attributes or {}
        return await type(self).from_store(
            self.store_path / name,
            attributes=attributes,
            exists_ok=exists_ok,
            zarr_format=self.metadata.zarr_format,
        )

    async def require_group(self, name: str, overwrite: bool = False) -> AsyncGroup:
        """Obtain a sub-group, creating one if it doesn't exist.

        Parameters
        ----------
        name : string
            Group name.
        overwrite : bool, optional
            Overwrite any existing group with given `name` if present.

        Returns
        -------
        g : AsyncGroup
        """
        if overwrite:
            # TODO: check that exists_ok=True errors if an array exists where the group is being created
            grp = await self.create_group(name, exists_ok=True)
        else:
            try:
                item: AsyncGroup | AsyncArray = await self.getitem(name)
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
        """Convenience method to require multiple groups in a single call."""
        if not names:
            return ()
        return tuple(await asyncio.gather(*(self.require_group(name) for name in names)))

    async def create_array(
        self,
        name: str,
        *,
        shape: ShapeLike,
        dtype: npt.DTypeLike = "float64",
        fill_value: Any | None = None,
        attributes: dict[str, JSON] | None = None,
        # v3 only
        chunk_shape: ChunkCoords | None = None,
        chunk_key_encoding: (
            ChunkKeyEncoding
            | tuple[Literal["default"], Literal[".", "/"]]
            | tuple[Literal["v2"], Literal[".", "/"]]
            | None
        ) = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        # v2 only
        chunks: ShapeLike | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        order: Literal["C", "F"] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        exists_ok: bool = False,
        data: npt.ArrayLike | None = None,
    ) -> AsyncArray:
        """
        Create a Zarr array within this AsyncGroup.
        This method lightly wraps AsyncArray.create.

        Parameters
        ----------
        name: str
            The name of the array.
        shape: tuple[int, ...]
            The shape of the array.
        dtype: np.DtypeLike = float64
            The data type of the array.
        chunk_shape: tuple[int, ...] | None = None
            The shape of the chunks of the array. V3 only.
        chunk_key_encoding: ChunkKeyEncoding | tuple[Literal["default"], Literal[".", "/"]] | tuple[Literal["v2"], Literal[".", "/"]] | None = None
            A specification of how the chunk keys are represented in storage.
        codecs: Iterable[Codec | dict[str, JSON]] | None = None
            An iterable of Codec or dict serializations thereof. The elements of
            this collection specify the transformation from array values to stored bytes.
        dimension_names: Iterable[str] | None = None
            The names of the dimensions of the array. V3 only.
        chunks: ChunkCoords | None = None
            The shape of the chunks of the array. V2 only.
        dimension_separator: Literal[".", "/"] | None = None
            The delimiter used for the chunk keys.
        order: Literal["C", "F"] | None = None
            The memory order of the array.
        filters: list[dict[str, JSON]] | None = None
            Filters for the array.
        compressor: dict[str, JSON] | None = None
            The compressor for the array.
        exists_ok: bool = False
            If True, a pre-existing array or group at the path of this array will
            be overwritten. If False, the presence of a pre-existing array or group is
            an error.

        Returns
        -------
        AsyncArray

        """
        return await AsyncArray.create(
            self.store_path / name,
            shape=shape,
            dtype=dtype,
            chunk_shape=chunk_shape,
            fill_value=fill_value,
            chunk_key_encoding=chunk_key_encoding,
            codecs=codecs,
            dimension_names=dimension_names,
            attributes=attributes,
            chunks=chunks,
            dimension_separator=dimension_separator,
            order=order,
            filters=filters,
            compressor=compressor,
            exists_ok=exists_ok,
            zarr_format=self.metadata.zarr_format,
            data=data,
        )

    @deprecated("Use AsyncGroup.create_array instead.")
    async def create_dataset(self, name: str, **kwargs: Any) -> AsyncArray:
        """Create an array.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.AsyncGroup.require_dataset` method.

        Parameters
        ----------
        name : string
            Array name.
        kwargs : dict
            Additional arguments passed to :func:`zarr.AsyncGroup.create_array`.

        Returns
        -------
        a : AsyncArray

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `AsyncGroup.create_array` instead.
        """
        return await self.create_array(name, **kwargs)

    @deprecated("Use AsyncGroup.require_array instead.")
    async def require_dataset(
        self,
        name: str,
        *,
        shape: ChunkCoords,
        dtype: npt.DTypeLike = None,
        exact: bool = False,
        **kwargs: Any,
    ) -> AsyncArray:
        """Obtain an array, creating if it doesn't exist.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.AsyncGroup.create_dataset` method.

        Other `kwargs` are as per :func:`zarr.AsyncGroup.create_dataset`.

        Parameters
        ----------
        name : string
            Array name.
        shape : int or tuple of ints
            Array shape.
        dtype : string or dtype, optional
            NumPy dtype.
        exact : bool, optional
            If True, require `dtype` to match exactly. If false, require
            `dtype` can be cast from array dtype.

        Returns
        -------
        a : AsyncArray

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `AsyncGroup.require_dataset` instead.
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
    ) -> AsyncArray:
        """Obtain an array, creating if it doesn't exist.

        Other `kwargs` are as per :func:`zarr.AsyncGroup.create_dataset`.

        Parameters
        ----------
        name : string
            Array name.
        shape : int or tuple of ints
            Array shape.
        dtype : string or dtype, optional
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
        """
        Count the number of members in this group.

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
        # TODO: consider using aioitertools.builtins.sum for this
        # return await aioitertools.builtins.sum((1 async for _ in self.members()), start=0)
        n = 0
        async for _ in self.members(max_depth=max_depth):
            n += 1
        return n

    async def members(
        self,
        max_depth: int | None = 0,
    ) -> AsyncGenerator[tuple[str, AsyncArray | AsyncGroup], None]:
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

        """
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth must be None or >= 0. Got '{max_depth}' instead")
        async for item in self._members(max_depth=max_depth, current_depth=0):
            yield item

    async def _members(
        self, max_depth: int | None, current_depth: int
    ) -> AsyncGenerator[tuple[str, AsyncArray | AsyncGroup], None]:
        if not self.store_path.store.supports_listing:
            msg = (
                f"The store associated with this group ({type(self.store_path.store)}) "
                "does not support listing, "
                "specifically via the `list_dir` method. "
                "This function requires a store that supports listing."
            )

            raise ValueError(msg)
        # would be nice to make these special keys accessible programmatically,
        # and scoped to specific zarr versions
        _skip_keys = ("zarr.json", ".zgroup", ".zattrs")

        async for key in self.store_path.store.list_dir(self.store_path.path):
            if key in _skip_keys:
                continue
            try:
                obj = await self.getitem(key)
                yield (key, obj)

                if (
                    ((max_depth is None) or (current_depth < max_depth))
                    and hasattr(obj.metadata, "node_type")
                    and obj.metadata.node_type == "group"
                ):
                    # the assert is just for mypy to know that `obj.metadata.node_type`
                    # implies an AsyncGroup, not an AsyncArray
                    assert isinstance(obj, AsyncGroup)
                    async for child_key, val in obj._members(
                        max_depth=max_depth, current_depth=current_depth + 1
                    ):
                        yield f"{key}/{child_key}", val
            except KeyError:
                # keyerror is raised when `key` names an object (in the object storage sense),
                # as opposed to a prefix, in the store under the prefix associated with this group
                # in which case `key` cannot be the name of a sub-array or sub-group.
                logger.warning(
                    "Object at %s is not recognized as a component of a Zarr hierarchy.", key
                )

    async def contains(self, member: str) -> bool:
        # TODO: this can be made more efficient.
        try:
            await self.getitem(member)
        except KeyError:
            return False
        else:
            return True

    async def groups(self) -> AsyncGenerator[tuple[str, AsyncGroup], None]:
        async for name, value in self.members():
            if isinstance(value, AsyncGroup):
                yield name, value

    async def group_keys(self) -> AsyncGenerator[str, None]:
        async for key, _ in self.groups():
            yield key

    async def group_values(self) -> AsyncGenerator[AsyncGroup, None]:
        async for _, group in self.groups():
            yield group

    async def arrays(self) -> AsyncGenerator[tuple[str, AsyncArray], None]:
        async for key, value in self.members():
            if isinstance(value, AsyncArray):
                yield key, value

    async def array_keys(self) -> AsyncGenerator[str, None]:
        async for key, _ in self.arrays():
            yield key

    async def array_values(self) -> AsyncGenerator[AsyncArray, None]:
        async for _, array in self.arrays():
            yield array

    async def tree(self, expand: bool = False, level: int | None = None) -> Any:
        raise NotImplementedError

    async def empty(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> AsyncArray:
        return await async_api.empty(shape=shape, store=self.store_path, path=name, **kwargs)

    async def zeros(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> AsyncArray:
        return await async_api.zeros(shape=shape, store=self.store_path, path=name, **kwargs)

    async def ones(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> AsyncArray:
        return await async_api.ones(shape=shape, store=self.store_path, path=name, **kwargs)

    async def full(
        self, *, name: str, shape: ChunkCoords, fill_value: Any | None, **kwargs: Any
    ) -> AsyncArray:
        return await async_api.full(
            shape=shape, fill_value=fill_value, store=self.store_path, path=name, **kwargs
        )

    async def empty_like(
        self, *, name: str, prototype: async_api.ArrayLike, **kwargs: Any
    ) -> AsyncArray:
        return await async_api.empty_like(a=prototype, store=self.store_path, path=name, **kwargs)

    async def zeros_like(
        self, *, name: str, prototype: async_api.ArrayLike, **kwargs: Any
    ) -> AsyncArray:
        return await async_api.zeros_like(a=prototype, store=self.store_path, path=name, **kwargs)

    async def ones_like(
        self, *, name: str, prototype: async_api.ArrayLike, **kwargs: Any
    ) -> AsyncArray:
        return await async_api.ones_like(a=prototype, store=self.store_path, path=name, **kwargs)

    async def full_like(
        self, *, name: str, prototype: async_api.ArrayLike, **kwargs: Any
    ) -> AsyncArray:
        return await async_api.full_like(a=prototype, store=self.store_path, path=name, **kwargs)

    async def move(self, source: str, dest: str) -> None:
        raise NotImplementedError


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
        exists_ok: bool = False,
    ) -> Group:
        attributes = attributes or {}
        obj = sync(
            AsyncGroup.from_store(
                store,
                attributes=attributes,
                exists_ok=exists_ok,
                zarr_format=zarr_format,
            ),
        )

        return cls(obj)

    @classmethod
    def open(
        cls,
        store: StoreLike,
        zarr_format: Literal[2, 3, None] = 3,
    ) -> Group:
        obj = sync(AsyncGroup.open(store, zarr_format=zarr_format))
        return cls(obj)

    def __getitem__(self, path: str) -> Array | Group:
        obj = self._sync(self._async_group.getitem(path))
        if isinstance(obj, AsyncArray):
            return Array(obj)
        else:
            return Group(obj)

    def __delitem__(self, key: str) -> None:
        self._sync(self._async_group.delitem(key))

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __setitem__(self, key: str, value: Any) -> None:
        """__setitem__ is not supported in v3"""
        raise NotImplementedError

    async def update_attributes_async(self, new_attributes: dict[str, Any]) -> Group:
        new_metadata = replace(self.metadata, attributes=new_attributes)

        # Write new metadata
        to_save = new_metadata.to_buffer_dict(default_buffer_prototype())
        awaitables = [set_or_delete(self.store_path / key, value) for key, value in to_save.items()]
        await asyncio.gather(*awaitables)

        async_group = replace(self._async_group, metadata=new_metadata)
        return replace(self, _async_group=async_group)

    @property
    def store_path(self) -> StorePath:
        return self._async_group.store_path

    @property
    def metadata(self) -> GroupMetadata:
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
        return Attributes(self)

    @property
    def info(self) -> None:
        raise NotImplementedError

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
        self._sync(self._async_group.update_attributes(new_attributes))
        return self

    def nmembers(self, max_depth: int | None = 0) -> int:
        return self._sync(self._async_group.nmembers(max_depth=max_depth))

    def members(self, max_depth: int | None = 0) -> tuple[tuple[str, Array | Group], ...]:
        """
        Return the sub-arrays and sub-groups of this group as a tuple of (name, array | group)
        pairs
        """
        _members = self._sync_iter(self._async_group.members(max_depth=max_depth))

        return tuple((kv[0], _parse_async_node(kv[1])) for kv in _members)

    def __contains__(self, member: str) -> bool:
        return self._sync(self._async_group.contains(member))

    def groups(self) -> Generator[tuple[str, Group], None]:
        for name, async_group in self._sync_iter(self._async_group.groups()):
            yield name, Group(async_group)

    def group_keys(self) -> Generator[str, None]:
        for name, _ in self.groups():
            yield name

    def group_values(self) -> Generator[Group, None]:
        for _, group in self.groups():
            yield group

    def arrays(self) -> Generator[tuple[str, Array], None]:
        for name, async_array in self._sync_iter(self._async_group.arrays()):
            yield name, Array(async_array)

    def array_keys(self) -> Generator[str, None]:
        for name, _ in self.arrays():
            yield name

    def array_values(self) -> Generator[Array, None]:
        for _, array in self.arrays():
            yield array

    def tree(self, expand: bool = False, level: int | None = None) -> Any:
        return self._sync(self._async_group.tree(expand=expand, level=level))

    def create_group(self, name: str, **kwargs: Any) -> Group:
        return Group(self._sync(self._async_group.create_group(name, **kwargs)))

    def require_group(self, name: str, **kwargs: Any) -> Group:
        """Obtain a sub-group, creating one if it doesn't exist.

        Parameters
        ----------
        name : string
            Group name.
        overwrite : bool, optional
            Overwrite any existing group with given `name` if present.

        Returns
        -------
        g : Group
        """
        return Group(self._sync(self._async_group.require_group(name, **kwargs)))

    def require_groups(self, *names: str) -> tuple[Group, ...]:
        """Convenience method to require multiple groups in a single call."""
        return tuple(map(Group, self._sync(self._async_group.require_groups(*names))))

    def create(self, *args: Any, **kwargs: Any) -> Array:
        # Backwards compatibility for 2.x
        return self.create_array(*args, **kwargs)

    def create_array(
        self,
        name: str,
        *,
        shape: ShapeLike,
        dtype: npt.DTypeLike = "float64",
        fill_value: Any | None = None,
        attributes: dict[str, JSON] | None = None,
        # v3 only
        chunk_shape: ChunkCoords | None = None,
        chunk_key_encoding: (
            ChunkKeyEncoding
            | tuple[Literal["default"], Literal[".", "/"]]
            | tuple[Literal["v2"], Literal[".", "/"]]
            | None
        ) = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        # v2 only
        chunks: ShapeLike | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        order: Literal["C", "F"] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        exists_ok: bool = False,
        data: npt.ArrayLike | None = None,
    ) -> Array:
        """
        Create a zarr array within this AsyncGroup.
        This method lightly wraps AsyncArray.create.

        Parameters
        ----------
        name: str
            The name of the array.
        shape: tuple[int, ...]
            The shape of the array.
        dtype: np.DtypeLike = float64
            The data type of the array.
        chunk_shape: tuple[int, ...] | None = None
            The shape of the chunks of the array. V3 only.
        chunk_key_encoding: ChunkKeyEncoding | tuple[Literal["default"], Literal[".", "/"]] | tuple[Literal["v2"], Literal[".", "/"]] | None = None
            A specification of how the chunk keys are represented in storage.
        codecs: Iterable[Codec | dict[str, JSON]] | None = None
            An iterable of Codec or dict serializations thereof. The elements of this collection
            specify the transformation from array values to stored bytes.
        dimension_names: Iterable[str] | None = None
            The names of the dimensions of the array. V3 only.
        chunks: ChunkCoords | None = None
            The shape of the chunks of the array. V2 only.
        dimension_separator: Literal[".", "/"] | None = None
            The delimiter used for the chunk keys.
        order: Literal["C", "F"] | None = None
            The memory order of the array.
        filters: list[dict[str, JSON]] | None = None
            Filters for the array.
        compressor: dict[str, JSON] | None = None
            The compressor for the array.
        exists_ok: bool = False
            If True, a pre-existing array or group at the path of this array will
            be overwritten. If False, the presence of a pre-existing array or group is
            an error.
        data: npt.ArrayLike | None = None
            Array data to initialize the array with.

        Returns
        -------
        Array

        """
        return Array(
            self._sync(
                self._async_group.create_array(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    fill_value=fill_value,
                    attributes=attributes,
                    chunk_shape=chunk_shape,
                    chunk_key_encoding=chunk_key_encoding,
                    codecs=codecs,
                    dimension_names=dimension_names,
                    chunks=chunks,
                    dimension_separator=dimension_separator,
                    order=order,
                    filters=filters,
                    compressor=compressor,
                    exists_ok=exists_ok,
                    data=data,
                )
            )
        )

    @deprecated("Use Group.create_array instead.")
    def create_dataset(self, name: str, **kwargs: Any) -> Array:
        """Create an array.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.Group.require_dataset` method.

        Parameters
        ----------
        name : string
            Array name.
        kwargs : dict
            Additional arguments passed to :func:`zarr.Group.create_array`

        Returns
        -------
        a : Array

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `Group.create_array` instead.
        """
        return Array(self._sync(self._async_group.create_dataset(name, **kwargs)))

    @deprecated("Use Group.require_array instead.")
    def require_dataset(self, name: str, **kwargs: Any) -> Array:
        """Obtain an array, creating if it doesn't exist.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the :func:`zarr.Group.create_dataset` method.

        Other `kwargs` are as per :func:`zarr.Group.create_dataset`.

        Parameters
        ----------
        name : string
            Array name.
        shape : int or tuple of ints
            Array shape.
        dtype : string or dtype, optional
            NumPy dtype.
        exact : bool, optional
            If True, require `dtype` to match exactly. If false, require
            `dtype` can be cast from array dtype.

        Returns
        -------
        a : Array

        .. deprecated:: 3.0.0
            The h5py compatibility methods will be removed in 3.1.0. Use `Group.require_array` instead.
        """
        return Array(self._sync(self._async_group.require_array(name, **kwargs)))

    def require_array(self, name: str, **kwargs: Any) -> Array:
        """Obtain an array, creating if it doesn't exist.


        Other `kwargs` are as per :func:`zarr.Group.create_array`.

        Parameters
        ----------
        name : string
            Array name.
        shape : int or tuple of ints
            Array shape.
        dtype : string or dtype, optional
            NumPy dtype.
        exact : bool, optional
            If True, require `dtype` to match exactly. If false, require
            `dtype` can be cast from array dtype.

        Returns
        -------
        a : Array
        """
        return Array(self._sync(self._async_group.require_array(name, **kwargs)))

    def empty(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.empty(name=name, shape=shape, **kwargs)))

    def zeros(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.zeros(name=name, shape=shape, **kwargs)))

    def ones(self, *, name: str, shape: ChunkCoords, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.ones(name=name, shape=shape, **kwargs)))

    def full(
        self, *, name: str, shape: ChunkCoords, fill_value: Any | None, **kwargs: Any
    ) -> Array:
        return Array(
            self._sync(
                self._async_group.full(name=name, shape=shape, fill_value=fill_value, **kwargs)
            )
        )

    def empty_like(self, *, name: str, prototype: async_api.ArrayLike, **kwargs: Any) -> Array:
        return Array(
            self._sync(self._async_group.empty_like(name=name, prototype=prototype, **kwargs))
        )

    def zeros_like(self, *, name: str, prototype: async_api.ArrayLike, **kwargs: Any) -> Array:
        return Array(
            self._sync(self._async_group.zeros_like(name=name, prototype=prototype, **kwargs))
        )

    def ones_like(self, *, name: str, prototype: async_api.ArrayLike, **kwargs: Any) -> Array:
        return Array(
            self._sync(self._async_group.ones_like(name=name, prototype=prototype, **kwargs))
        )

    def full_like(self, *, name: str, prototype: async_api.ArrayLike, **kwargs: Any) -> Array:
        return Array(
            self._sync(self._async_group.full_like(name=name, prototype=prototype, **kwargs))
        )

    def move(self, source: str, dest: str) -> None:
        return self._sync(self._async_group.move(source, dest))

    @deprecated("Use Group.create_array instead.")
    def array(
        self,
        name: str,
        *,
        shape: ChunkCoords,
        dtype: npt.DTypeLike = "float64",
        fill_value: Any | None = None,
        attributes: dict[str, JSON] | None = None,
        # v3 only
        chunk_shape: ChunkCoords | None = None,
        chunk_key_encoding: (
            ChunkKeyEncoding
            | tuple[Literal["default"], Literal[".", "/"]]
            | tuple[Literal["v2"], Literal[".", "/"]]
            | None
        ) = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        # v2 only
        chunks: ChunkCoords | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        order: Literal["C", "F"] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        exists_ok: bool = False,
        data: npt.ArrayLike | None = None,
    ) -> Array:
        """
        Create a zarr array within this AsyncGroup.
        This method lightly wraps `AsyncArray.create`.

        Parameters
        ----------
        name: str
            The name of the array.
        shape: tuple[int, ...]
            The shape of the array.
        dtype: np.DtypeLike = float64
            The data type of the array.
        chunk_shape: tuple[int, ...] | None = None
            The shape of the chunks of the array. V3 only.
        chunk_key_encoding: ChunkKeyEncoding | tuple[Literal["default"], Literal[".", "/"]] | tuple[Literal["v2"], Literal[".", "/"]] | None = None
            A specification of how the chunk keys are represented in storage.
        codecs: Iterable[Codec | dict[str, JSON]] | None = None
            An iterable of Codec or dict serializations thereof. The elements of
            this collection specify the transformation from array values to stored bytes.
        dimension_names: Iterable[str] | None = None
            The names of the dimensions of the array. V3 only.
        chunks: ChunkCoords | None = None
            The shape of the chunks of the array. V2 only.
        dimension_separator: Literal[".", "/"] | None = None
            The delimiter used for the chunk keys.
        order: Literal["C", "F"] | None = None
            The memory order of the array.
        filters: list[dict[str, JSON]] | None = None
            Filters for the array.
        compressor: dict[str, JSON] | None = None
            The compressor for the array.
        exists_ok: bool = False
            If True, a pre-existing array or group at the path of this array will
            be overwritten. If False, the presence of a pre-existing array or group is
            an error.
        data: npt.ArrayLike | None = None
            Array data to initialize the array with.

        Returns
        -------

        Array

        """
        return Array(
            self._sync(
                self._async_group.create_array(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    fill_value=fill_value,
                    attributes=attributes,
                    chunk_shape=chunk_shape,
                    chunk_key_encoding=chunk_key_encoding,
                    codecs=codecs,
                    dimension_names=dimension_names,
                    chunks=chunks,
                    dimension_separator=dimension_separator,
                    order=order,
                    filters=filters,
                    compressor=compressor,
                    exists_ok=exists_ok,
                    data=data,
                )
            )
        )
