from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import asdict, dataclass, field, replace

import asyncio
import json
import logging

if TYPE_CHECKING:
    from typing import (
        Any,
        AsyncGenerator,
        Literal,
        AsyncIterator,
        Iterator,
    )
from zarr.v3.abc.metadata import Metadata

from zarr.v3.array import AsyncArray, Array
from zarr.v3.attributes import Attributes
from zarr.v3.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON
from zarr.v3.config import RuntimeConfiguration, SyncConfiguration
from zarr.v3.store import StoreLike, StorePath, make_store_path
from zarr.v3.sync import SyncMixin, sync

logger = logging.getLogger("zarr.group")


def parse_zarr_format(data: Any) -> Literal[2, 3]:
    if data in (2, 3):
        return data
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


@dataclass(frozen=True)
class GroupMetadata(Metadata):
    attributes: dict[str, Any] = field(default_factory=dict)
    zarr_format: Literal[2, 3] = 3
    node_type: Literal["group"] = field(default="group", init=False)

    # todo: rename this, since it doesn't return bytes
    def to_bytes(self) -> dict[str, bytes]:
        if self.zarr_format == 3:
            return {ZARR_JSON: json.dumps(self.to_dict()).encode()}
        else:
            return {
                ZGROUP_JSON: json.dumps({"zarr_format": self.zarr_format}).encode(),
                ZATTRS_JSON: json.dumps(self.attributes).encode(),
            }

    def __init__(self, attributes: dict[str, Any] = {}, zarr_format: Literal[2, 3] = 3):
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
    runtime_configuration: RuntimeConfiguration = RuntimeConfiguration()

    @classmethod
    async def create(
        cls,
        store: StoreLike,
        *,
        attributes: dict[str, Any] = {},
        exists_ok: bool = False,
        zarr_format: Literal[2, 3] = 3,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
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
            runtime_configuration=runtime_configuration,
        )
        await group._save_metadata()
        return group

    @classmethod
    async def open(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
        zarr_format: Literal[2, 3] = 3,
    ) -> AsyncGroup:
        store_path = make_store_path(store)
        zarr_json_bytes = await (store_path / ZARR_JSON).get_async()
        assert zarr_json_bytes is not None

        # TODO: consider trying to autodiscover the zarr-format here
        if zarr_format == 3:
            # V3 groups are comprised of a zarr.json object
            # (it is optional in the case of implicit groups)
            zarr_json_bytes = await (store_path / ZARR_JSON).get()
            zarr_json = (
                json.loads(zarr_json_bytes) if zarr_json_bytes is not None else {"zarr_format": 3}
            )

        elif zarr_format == 2:
            # V2 groups are comprised of a .zgroup and .zattrs objects
            # (both are optional in the case of implicit groups)
            zgroup_bytes, zattrs_bytes = await asyncio.gather(
                (store_path / ZGROUP_JSON).get(), (store_path / ZATTRS_JSON).get()
            )
            zgroup = (
                json.loads(json.loads(zgroup_bytes))
                if zgroup_bytes is not None
                else {"zarr_format": 2}
            )
            zattrs = json.loads(json.loads(zattrs_bytes)) if zattrs_bytes is not None else {}
            zarr_json = {**zgroup, "attributes": zattrs}
        else:
            raise ValueError(f"unexpected zarr_format: {zarr_format}")
        return cls.from_dict(store_path, zarr_json, runtime_configuration)

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        data: dict[str, Any],
        runtime_configuration: RuntimeConfiguration,
    ) -> Group:
        group = cls(
            metadata=GroupMetadata.from_dict(data),
            store_path=store_path,
            runtime_configuration=runtime_configuration,
        )
        return group

    async def getitem(
        self,
        key: str,
    ) -> AsyncArray | AsyncGroup:

        store_path = self.store_path / key

        # Note:
        # in zarr-python v2, we first check if `key` references an Array, else if `key` references
        # a group,using standalone `contains_array` and `contains_group` functions. These functions
        # are reusable, but for v3 they would perform redundant I/O operations.
        # Not clear how much of that strategy we want to keep here.

        # if `key` names an object in storage, it cannot be an array or group
        if await store_path.exists():
            raise KeyError(key)

        if self.metadata.zarr_format == 3:
            zarr_json_bytes = await (store_path / ZARR_JSON).get()
            if zarr_json_bytes is None:
                # implicit group?
                logger.warning("group at %s is an implicit group", store_path)
                zarr_json = {
                    "zarr_format": self.metadata.zarr_format,
                    "node_type": "group",
                    "attributes": {},
                }
            else:
                zarr_json = json.loads(zarr_json_bytes)
            if zarr_json["node_type"] == "group":
                return type(self).from_dict(store_path, zarr_json, self.runtime_configuration)
            if zarr_json["node_type"] == "array":
                return AsyncArray.from_dict(
                    store_path, zarr_json, runtime_configuration=self.runtime_configuration
                )
        elif self.metadata.zarr_format == 2:
            # Q: how do we like optimistically fetching .zgroup, .zarray, and .zattrs?
            # This guarantees that we will always make at least one extra request to the store
            zgroup_bytes, zarray_bytes, zattrs_bytes = await asyncio.gather(
                (store_path / ZGROUP_JSON).get(),
                (store_path / ZARRAY_JSON).get(),
                (store_path / ZATTRS_JSON).get(),
            )

            # unpack the zarray, if this is None then we must be opening a group
            zarray = json.loads(zarray_bytes) if zarray_bytes else None
            # unpack the zattrs, this can be None if no attrs were written
            zattrs = json.loads(zattrs_bytes) if zattrs_bytes is not None else {}

            if zarray is not None:
                # TODO: update this once the V2 array support is part of the primary array class
                zarr_json = {**zarray, "attributes": zattrs}
                return AsyncArray.from_dict(
                    store_path, zarray, runtime_configuration=self.runtime_configuration
                )
            else:
                if zgroup_bytes is None:
                    # implicit group?
                    logger.warning("group at %s is an implicit group", store_path)
                zgroup = (
                    json.loads(zgroup_bytes)
                    if zgroup_bytes is not None
                    else {"zarr_format": self.metadata.zarr_format}
                )
                zarr_json = {**zgroup, "attributes": zattrs}
                return type(self).from_dict(store_path, zarr_json, self.runtime_configuration)
        else:
            raise ValueError(f"unexpected zarr_format: {self.metadata.zarr_format}")

    async def delitem(self, key: str) -> None:
        store_path = self.store_path / key
        if self.metadata.zarr_format == 3:
            await (store_path / ZARR_JSON).delete()
        elif self.metadata.zarr_format == 2:
            await asyncio.gather(
                (store_path / ZGROUP_JSON).delete(),  # TODO: missing_ok=False
                (store_path / ZATTRS_JSON).delete(),  # TODO: missing_ok=True
            )
        else:
            raise ValueError(f"unexpected zarr_format: {self.metadata.zarr_format}")

    async def _save_metadata(self) -> None:
        to_save = self.metadata.to_bytes()
        awaitables = [(self.store_path / key).set(value) for key, value in to_save.items()]
        await asyncio.gather(*awaitables)

    @property
    def attrs(self):
        return self.metadata.attributes

    @property
    def info(self):
        return self.metadata.info

    async def create_group(self, path: str, **kwargs) -> AsyncGroup:
        runtime_configuration = kwargs.pop("runtime_configuration", self.runtime_configuration)
        return await type(self).create(
            self.store_path / path,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )

    async def create_array(self, path: str, **kwargs) -> AsyncArray:
        runtime_configuration = kwargs.pop("runtime_configuration", self.runtime_configuration)
        return await AsyncArray.create(
            self.store_path / path,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )

    async def update_attributes(self, new_attributes: dict[str, Any]):
        # metadata.attributes is "frozen" so we simply clear and update the dict
        self.metadata.attributes.clear()
        self.metadata.attributes.update(new_attributes)

        # Write new metadata
        to_save = self.metadata.to_bytes()
        if self.metadata.zarr_format == 2:
            # only save the .zattrs object
            await (self.store_path / ZATTRS_JSON).set(to_save[ZATTRS_JSON])
        else:
            await (self.store_path / ZARR_JSON).set(to_save[ZARR_JSON])

        self.metadata.attributes.clear()
        self.metadata.attributes.update(new_attributes)

        return self

    def __repr__(self):
        return f"<AsyncGroup {self.store_path}>"

    async def nchildren(self) -> int:
        raise NotImplementedError

    async def children(self) -> AsyncGenerator[AsyncArray, AsyncGroup]:
        """
        Returns an AsyncGenerator over the arrays and groups contained in this group.
        This method requires that `store_path.store` supports directory listing.
        """
        if not self.store_path.store.supports_listing:
            msg = (
                f"The store associated with this group ({type(self.store_path.store)}) "
                "does not support listing, "
                "specifically via the `list_dir` method. "
                "This function requires a store that supports listing."
            )

            raise ValueError(msg)
        subkeys = await self.store_path.store.list_dir(self.store_path.path)
        # would be nice to make these special keys accessible programmatically,
        # and scoped to specific zarr versions
        subkeys_filtered = filter(lambda v: v not in ("zarr.json", ".zgroup", ".zattrs"), subkeys)
        # is there a better way to schedule this?
        for subkey in subkeys_filtered:
            try:
                yield await self.getitem(subkey)
            except KeyError:
                # keyerror is raised when `subkey``names an object in the store
                # in which case `subkey` cannot be the name of a sub-array or sub-group.
                pass

    async def contains(self, child: str) -> bool:
        raise NotImplementedError

    async def group_keys(self) -> AsyncIterator[str]:
        raise NotImplementedError

    async def groups(self) -> AsyncIterator[AsyncGroup]:
        raise NotImplementedError

    async def array_keys(self) -> AsyncIterator[str]:
        raise NotImplementedError

    async def arrays(self) -> AsyncIterator[AsyncArray]:
        raise NotImplementedError

    async def tree(self, expand=False, level=None) -> Any:
        raise NotImplementedError

    async def empty(self, **kwargs) -> AsyncArray:
        raise NotImplementedError

    async def zeros(self, **kwargs) -> AsyncArray:
        raise NotImplementedError

    async def ones(self, **kwargs) -> AsyncArray:
        raise NotImplementedError

    async def full(self, **kwargs) -> AsyncArray:
        raise NotImplementedError

    async def empty_like(self, prototype: AsyncArray, **kwargs) -> AsyncArray:
        raise NotImplementedError

    async def zeros_like(self, prototype: AsyncArray, **kwargs) -> AsyncArray:
        raise NotImplementedError

    async def ones_like(self, prototype: AsyncArray, **kwargs) -> AsyncArray:
        raise NotImplementedError

    async def full_like(self, prototype: AsyncArray, **kwargs) -> AsyncArray:
        raise NotImplementedError

    async def move(self, source: str, dest: str) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class Group(SyncMixin):
    _async_group: AsyncGroup
    _sync_configuration: SyncConfiguration = field(init=True, default=SyncConfiguration())

    @classmethod
    def create(
        cls,
        store: StoreLike,
        *,
        attributes: dict[str, Any] = {},
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        obj = sync(
            AsyncGroup.create(
                store,
                attributes=attributes,
                exists_ok=exists_ok,
                runtime_configuration=runtime_configuration,
            ),
            loop=runtime_configuration.asyncio_loop,
        )

        return cls(obj)

    @classmethod
    def open(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        obj = sync(
            AsyncGroup.open(store, runtime_configuration), loop=runtime_configuration.asyncio_loop
        )
        return cls(obj)

    def __getitem__(self, path: str) -> Array | Group:
        obj = self._sync(self._async_group.getitem(path))
        if isinstance(obj, AsyncArray):
            return Array(obj)
        else:
            return Group(obj)

    def __delitem__(self, key) -> None:
        self._sync(self._async_group.delitem(key))

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __setitem__(self, key, value):
        """__setitem__ is not supported in v3"""
        raise NotImplementedError

    async def update_attributes_async(self, new_attributes: dict[str, Any]) -> Group:
        new_metadata = replace(self.metadata, attributes=new_attributes)

        # Write new metadata
        await (self.store_path / ZARR_JSON).set_async(new_metadata.to_bytes())
        return replace(self, metadata=new_metadata)

    @property
    def metadata(self) -> GroupMetadata:
        return self._async_group.metadata

    @property
    def attrs(self) -> Attributes:
        return Attributes(self)

    @property
    def info(self):
        return self._async_group.info

    def update_attributes(self, new_attributes: dict[str, Any]):
        self._sync(self._async_group.update_attributes(new_attributes))
        return self

    @property
    def nchildren(self) -> int:
        return self._sync(self._async_group.nchildren)

    @property
    def children(self) -> list[Array | Group]:
        _children = self._sync_iter(self._async_group.children)
        return [Array(obj) if isinstance(obj, AsyncArray) else Group(obj) for obj in _children]

    def __contains__(self, child) -> bool:
        return self._sync(self._async_group.contains(child))

    def group_keys(self) -> Iterator[str]:
        return self._sync_iter(self._async_group.group_keys)

    def groups(self) -> list[Group]:
        # TODO: in v2 this was a generator that return key: Group
        return [Group(obj) for obj in self._sync_iter(self._async_group.groups)]

    def array_keys(self) -> list[str]:
        return self._sync_iter(self._async_group.array_keys)

    def arrays(self) -> list[Array]:
        return [Array(obj) for obj in self._sync_iter(self._async_group.arrays)]

    def tree(self, expand=False, level=None) -> Any:
        return self._sync(self._async_group.tree(expand=expand, level=level))

    def create_group(self, name: str, **kwargs) -> Group:
        return Group(self._sync(self._async_group.create_group(name, **kwargs)))

    def create_array(self, name: str, **kwargs) -> Array:
        return Array(self._sync(self._async_group.create_array(name, **kwargs)))

    def empty(self, **kwargs) -> Array:
        return Array(self._sync(self._async_group.empty(**kwargs)))

    def zeros(self, **kwargs) -> Array:
        return Array(self._sync(self._async_group.zeros(**kwargs)))

    def ones(self, **kwargs) -> Array:
        return Array(self._sync(self._async_group.ones(**kwargs)))

    def full(self, **kwargs) -> Array:
        return Array(self._sync(self._async_group.full(**kwargs)))

    def empty_like(self, prototype: AsyncArray, **kwargs) -> Array:
        return Array(self._sync(self._async_group.empty_like(prototype, **kwargs)))

    def zeros_like(self, prototype: AsyncArray, **kwargs) -> Array:
        return Array(self._sync(self._async_group.zeros_like(prototype, **kwargs)))

    def ones_like(self, prototype: AsyncArray, **kwargs) -> Array:
        return Array(self._sync(self._async_group.ones_like(prototype, **kwargs)))

    def full_like(self, prototype: AsyncArray, **kwargs) -> Array:
        return Array(self._sync(self._async_group.full_like(prototype, **kwargs)))

    def move(self, source: str, dest: str) -> None:
        return self._sync(self._async_group.move(source, dest))
