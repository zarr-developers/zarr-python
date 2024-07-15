from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy.typing as npt

from zarr.abc.codec import Codec
from zarr.abc.metadata import Metadata
from zarr.abc.store import set_or_delete
from zarr.array import Array, AsyncArray
from zarr.attributes import Attributes
from zarr.buffer import Buffer, default_buffer_prototype
from zarr.chunk_key_encodings import ChunkKeyEncoding
from zarr.common import (
    JSON,
    ZARR_JSON,
    ZARRAY_JSON,
    ZATTRS_JSON,
    ZGROUP_JSON,
    ChunkCoords,
    ZarrFormat,
)
from zarr.config import config
from zarr.store import StoreLike, StorePath, make_store_path
from zarr.sync import SyncMixin, sync

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable
    from typing import Any, Literal

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
    elif isinstance(data, dict) and all(map(lambda v: isinstance(v, str), data.keys())):
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

    def to_buffer_dict(self) -> dict[str, Buffer]:
        json_indent = config.get("json_indent")
        if self.zarr_format == 3:
            return {
                ZARR_JSON: default_buffer_prototype.buffer.from_bytes(
                    json.dumps(self.to_dict(), indent=json_indent).encode()
                )
            }
        else:
            return {
                ZGROUP_JSON: default_buffer_prototype.buffer.from_bytes(
                    json.dumps({"zarr_format": self.zarr_format}, indent=json_indent).encode()
                ),
                ZATTRS_JSON: default_buffer_prototype.buffer.from_bytes(
                    json.dumps(self.attributes, indent=json_indent).encode()
                ),
            }

    def __init__(self, attributes: dict[str, Any] | None = None, zarr_format: ZarrFormat = 3):
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
    async def create(
        cls,
        store: StoreLike,
        *,
        attributes: dict[str, Any] = {},  # noqa: B006, FIXME
        exists_ok: bool = False,
        zarr_format: ZarrFormat = 3,
    ) -> AsyncGroup:
        store_path = make_store_path(store)
        if not exists_ok:
            if zarr_format == 3:
                assert not await (store_path / ZARR_JSON).exists()
            elif zarr_format == 2:
                assert not await (store_path / ZGROUP_JSON).exists()
        group = cls(
            metadata=GroupMetadata(attributes=attributes, zarr_format=zarr_format),
            store_path=store_path,
        )
        await group._save_metadata()
        return group

    @classmethod
    async def open(
        cls,
        store: StoreLike,
        zarr_format: Literal[2, 3, None] = 3,
    ) -> AsyncGroup:
        store_path = make_store_path(store)

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
                raise KeyError(store_path)  # filenotfounderror?
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
        group = cls(
            metadata=GroupMetadata.from_dict(data),
            store_path=store_path,
        )
        return group

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
                return AsyncArray.from_dict(store_path, zarray)
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

    async def _save_metadata(self) -> None:
        to_save = self.metadata.to_buffer_dict()
        awaitables = [set_or_delete(self.store_path / key, value) for key, value in to_save.items()]
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

    async def create_group(
        self,
        path: str,
        exists_ok: bool = False,
        attributes: dict[str, Any] = {},  # noqa: B006, FIXME
    ) -> AsyncGroup:
        return await type(self).create(
            self.store_path / path,
            attributes=attributes,
            exists_ok=exists_ok,
            zarr_format=self.metadata.zarr_format,
        )

    async def create_array(
        self,
        path: str,
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
    ) -> AsyncArray:
        return await AsyncArray.create(
            self.store_path / path,
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
        )

    async def update_attributes(self, new_attributes: dict[str, Any]) -> AsyncGroup:
        # metadata.attributes is "frozen" so we simply clear and update the dict
        self.metadata.attributes.clear()
        self.metadata.attributes.update(new_attributes)

        # Write new metadata
        await self._save_metadata()

        return self

    def __repr__(self) -> str:
        return f"<AsyncGroup {self.store_path}>"

    async def nmembers(self) -> int:
        # TODO: consider using aioitertools.builtins.sum for this
        # return await aioitertools.builtins.sum((1 async for _ in self.members()), start=0)
        n = 0
        async for _ in self.members():
            n += 1
        return n

    async def members(self) -> AsyncGenerator[tuple[str, AsyncArray | AsyncGroup], None]:
        """
        Returns an AsyncGenerator over the arrays and groups contained in this group.
        This method requires that `store_path.store` supports directory listing.

        The results are not guaranteed to be ordered.
        """
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
                yield (key, await self.getitem(key))
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
            return True
        except KeyError:
            return False

    # todo: decide if this method should be separate from `groups`
    async def group_keys(self) -> AsyncGenerator[str, None]:
        async for key, value in self.members():
            if isinstance(value, AsyncGroup):
                yield key

    # todo: decide if this method should be separate from `group_keys`
    async def groups(self) -> AsyncGenerator[AsyncGroup, None]:
        async for _, value in self.members():
            if isinstance(value, AsyncGroup):
                yield value

    # todo: decide if this method should be separate from `arrays`
    async def array_keys(self) -> AsyncGenerator[str, None]:
        async for key, value in self.members():
            if isinstance(value, AsyncArray):
                yield key

    # todo: decide if this method should be separate from `array_keys`
    async def arrays(self) -> AsyncGenerator[AsyncArray, None]:
        async for _, value in self.members():
            if isinstance(value, AsyncArray):
                yield value

    async def tree(self, expand: bool = False, level: int | None = None) -> Any:
        raise NotImplementedError

    async def empty(self, **kwargs: Any) -> AsyncArray:
        raise NotImplementedError

    async def zeros(self, **kwargs: Any) -> AsyncArray:
        raise NotImplementedError

    async def ones(self, **kwargs: Any) -> AsyncArray:
        raise NotImplementedError

    async def full(self, **kwargs: Any) -> AsyncArray:
        raise NotImplementedError

    async def empty_like(self, prototype: AsyncArray, **kwargs: Any) -> AsyncArray:
        raise NotImplementedError

    async def zeros_like(self, prototype: AsyncArray, **kwargs: Any) -> AsyncArray:
        raise NotImplementedError

    async def ones_like(self, prototype: AsyncArray, **kwargs: Any) -> AsyncArray:
        raise NotImplementedError

    async def full_like(self, prototype: AsyncArray, **kwargs: Any) -> AsyncArray:
        raise NotImplementedError

    async def move(self, source: str, dest: str) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class Group(SyncMixin):
    _async_group: AsyncGroup

    @classmethod
    def create(
        cls,
        store: StoreLike,
        *,
        attributes: dict[str, Any] = {},  # noqa: B006, FIXME
        zarr_format: ZarrFormat = 3,
        exists_ok: bool = False,
    ) -> Group:
        obj = sync(
            AsyncGroup.create(
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
    ) -> Group:
        obj = sync(AsyncGroup.open(store))
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
        to_save = new_metadata.to_buffer_dict()
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

    def update_attributes(self, new_attributes: dict[str, Any]) -> Group:
        self._sync(self._async_group.update_attributes(new_attributes))
        return self

    @property
    def nmembers(self) -> int:
        return self._sync(self._async_group.nmembers())

    @property
    def members(self) -> tuple[tuple[str, Array | Group], ...]:
        """
        Return the sub-arrays and sub-groups of this group as a tuple of (name, array | group)
        pairs
        """
        _members = self._sync_iter(self._async_group.members())

        result = tuple(map(lambda kv: (kv[0], _parse_async_node(kv[1])), _members))
        return result

    def __contains__(self, member: str) -> bool:
        return self._sync(self._async_group.contains(member))

    def group_keys(self) -> tuple[str, ...]:
        return tuple(self._sync_iter(self._async_group.group_keys()))

    def groups(self) -> tuple[Group, ...]:
        # TODO: in v2 this was a generator that return key: Group
        return tuple(Group(obj) for obj in self._sync_iter(self._async_group.groups()))

    def array_keys(self) -> tuple[str, ...]:
        return tuple(self._sync_iter(self._async_group.array_keys()))

    def arrays(self) -> tuple[Array, ...]:
        return tuple(Array(obj) for obj in self._sync_iter(self._async_group.arrays()))

    def tree(self, expand: bool = False, level: int | None = None) -> Any:
        return self._sync(self._async_group.tree(expand=expand, level=level))

    def create_group(self, name: str, **kwargs: Any) -> Group:
        return Group(self._sync(self._async_group.create_group(name, **kwargs)))

    def create_array(self, name: str, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.create_array(name, **kwargs)))

    def empty(self, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.empty(**kwargs)))

    def zeros(self, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.zeros(**kwargs)))

    def ones(self, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.ones(**kwargs)))

    def full(self, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.full(**kwargs)))

    def empty_like(self, prototype: AsyncArray, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.empty_like(prototype, **kwargs)))

    def zeros_like(self, prototype: AsyncArray, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.zeros_like(prototype, **kwargs)))

    def ones_like(self, prototype: AsyncArray, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.ones_like(prototype, **kwargs)))

    def full_like(self, prototype: AsyncArray, **kwargs: Any) -> Array:
        return Array(self._sync(self._async_group.full_like(prototype, **kwargs)))

    def move(self, source: str, dest: str) -> None:
        return self._sync(self._async_group.move(source, dest))
