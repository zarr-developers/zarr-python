from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Literal, Optional, Union, AsyncIterator, Iterator, List

from attr import asdict, evolve, field, frozen  # , validators

from zarr.v3.abc import AsyncGroup, SyncGroup
from zarr.v3.array import AsyncArray, Array
from zarr.v3.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, make_cattr
from zarr.v3.config import RuntimeConfiguration, SyncConfiguration
from zarr.v3.store import StoreLike, StorePath, make_store_path
from zarr.v3.sync import SyncMixin


@frozen
class GroupMetadata:
    attributes: Dict[str, Any] = field(factory=dict)
    zarr_format: Literal[2, 3] = 3  # field(default=3, validator=validators.in_([2, 3]))
    node_type: Literal["group"] = field(default="group", init=False)

    def to_bytes(self) -> Dict[str, bytes]:
        if self.zarr_format == 3:
            return {ZARR_JSON: json.dumps(asdict(self)).encode()}
        elif self.zarr_format == 2:
            return {
                ZGROUP_JSON: self.zarr_format,
                ZATTRS_JSON: json.dumps(self.attributes).encode(),
            }
        else:
            raise ValueError(f"unexpected zarr_format: {self.zarr_format}")

    @classmethod
    def from_json(cls, zarr_json: Any) -> GroupMetadata:
        return make_cattr().structure(zarr_json, GroupMetadata)


@frozen
class AsyncGroup(AsyncGroup):
    metadata: GroupMetadata
    store_path: StorePath
    runtime_configuration: RuntimeConfiguration

    @classmethod
    async def create(
        cls,
        store: StoreLike,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
        zarr_format: Literal[2, 3] = 3,  # field(default=3, validator=validators.in_([2, 3])),
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> "AsyncGroup":
        store_path = make_store_path(store)
        if not exists_ok:
            if zarr_format == 3:
                assert not await (store_path / ZARR_JSON).exists_async()
            elif zarr_format == 2:
                assert not await (store_path / ZGROUP_JSON).exists_async()
        print(zarr_format, type(zarr_format))
        group = cls(
            metadata=GroupMetadata(attributes=attributes or {}, zarr_format=zarr_format),
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
    ) -> "AsyncGroup":
        store_path = make_store_path(store)

        # TODO: consider trying to autodiscover the zarr-format here
        if zarr_format == 3:
            # V3 groups are comprised of a zarr.json object
            # (it is optional in the case of implicit groups)
            zarr_json_bytes = await (store_path / ZARR_JSON).get_async()
            zarr_json = (
                json.loads(zarr_json_bytes) if zarr_json_bytes is not None else {"zarr_format": 3}
            )

        elif zarr_format == 2:
            # V2 groups are comprised of a .zgroup and .zattrs objects
            # (both are optional in the case of implicit groups)
            zgroup_bytes, zattrs_bytes = await asyncio.gather(
                (store_path / ZGROUP_JSON).get_async(), (store_path / ZATTRS_JSON).get_async()
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
        return cls.from_json(store_path, zarr_json, runtime_configuration)

    @classmethod
    def from_json(
        cls,
        store_path: StorePath,
        zarr_json: Any,
        runtime_configuration: RuntimeConfiguration,
    ) -> Group:
        group = cls(
            metadata=GroupMetadata.from_json(zarr_json),
            store_path=store_path,
            runtime_configuration=runtime_configuration,
        )
        return group

    async def getitem(
        self,
        key: str,
    ) -> Union[Array, Group]:

        store_path = self.store_path / key

        if self.zarr_format == 3:
            zarr_json_bytes = await (store_path / ZARR_JSON).get_async()
            if zarr_json_bytes is None:
                # implicit group?
                zarr_json = {
                    "zarr_format": self.zarr_format,
                    "node_type": "group",
                    "attributes": {},
                }
            else:
                zarr_json = json.loads(zarr_json_bytes)
            if zarr_json["node_type"] == "group":
                return type(self).from_json(store_path, zarr_json, self.runtime_configuration)
            if zarr_json["node_type"] == "array":
                return Array.from_json(
                    store_path, zarr_json, runtime_configuration=self.runtime_configuration
                )
        elif self.zarr_format == 2:
            # Q: how do we like optimistically fetching .zgroup, .zarray, and .zattrs?
            # This guarantees that we will always make at least one extra request to the store
            zgroup_bytes, zarray_bytes, zattrs_bytes = await asyncio.gather(
                (store_path / ZGROUP_JSON).get_async(),
                (store_path / ZARRAY_JSON).get_async(),
                (store_path / ZATTRS_JSON).get_async(),
            )

            # unpack the zarray, if this is None then we must be opening a group
            zarray = json.loads(zarray_bytes) if zarray_bytes else None
            # unpack the zattrs, this can be None if no attrs were written
            zattrs = json.loads(zattrs_bytes) if zattrs_bytes is not None else {}

            if zarray is not None:
                # TODO: update this once the V2 array support is part of the primary array class
                zarr_json = {**zarray, "attributes": zattrs}
                return Array.from_json(
                    store_path, zarray, runtime_configuration=self.runtime_configuration
                )
            else:
                zgroup = (
                    json.loads(zgroup_bytes)
                    if zgroup_bytes is not None
                    else {"zarr_format": self.zarr_format}
                )
                zarr_json = {**zgroup, "attributes": zattrs}
                return type(self).from_json(store_path, zarr_json, self.runtime_configuration)
        else:
            raise ValueError(f"unexpected zarr_format: {self.zarr_format}")

    async def _save_metadata(self) -> None:
        to_save = self.metadata.to_bytes()
        awaitables = [(self.store_path / key).set_async(value) for key, value in to_save.items()]
        await asyncio.gather(*awaitables)

    @property
    def attrs(self):
        return self.metadata.attributes

    @property
    def info(self):
        return self.metadata.info

    async def create_group(self, path: str, **kwargs) -> Group:
        runtime_configuration = kwargs.pop("runtime_configuration", self.runtime_configuration)
        return await type(self).create(
            self.store_path / path,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )

    async def create_array(self, path: str, **kwargs) -> Array:
        runtime_configuration = kwargs.pop("runtime_configuration", self.runtime_configuration)
        return await AsyncArray.create(
            self.store_path / path,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )

    async def update_attributes(self, new_attributes: Dict[str, Any]):
        new_metadata = evolve(self.metadata, attributes=new_attributes)

        # Write new metadata
        to_save = new_metadata.to_bytes()
        if new_metadata.zarr_format == 2:
            # only save the .zattrs object
            await (self.store_path / ZATTRS_JSON).set_async(to_save[ZATTRS_JSON])
        else:
            await (self.store_path / ZARR_JSON).set_async(to_save[ZARR_JSON])
        return evolve(self, metadata=new_metadata)

    def __repr__(self):
        return f"<Group {self.store_path}>"

    async def nchildren(self) -> int:
        raise NotImplementedError

    async def children(self) -> AsyncIterator[AsyncArray, "AsyncGroup"]:
        raise NotImplementedError

    async def contains(self, child: str) -> bool:
        raise NotImplementedError

    async def group_keys(self) -> AsyncIterator[str]:
        raise NotImplementedError

    async def groups(self) -> AsyncIterator["AsyncGroup"]:
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


@frozen
class Group(SyncGroup, SyncMixin):
    _async_group: AsyncGroup
    _sync_configuration: field(factory=SyncConfiguration)

    @classmethod
    def create(
        cls,
        store: StoreLike,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        return cls._sync(
            AsyncGroup.create,
            store,
            attributes=attributes,
            exists_ok=exists_ok,
            runtime_configuration=runtime_configuration,
        )

    @classmethod
    def open(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        obj = cls._sync(AsyncGroup.open, store, runtime_configuration)
        return cls(obj)

    def __getitem__(self, path: str) -> Union[Array, Group]:
        obj = self._sync(self._async_group.getitem, path)
        if isinstance(obj, AsyncArray):
            return Array(obj)
        else:
            return Group(obj)

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    @property
    def attrs(self):
        return self._async_group.attrs

    @property
    def info(self):
        return self._async_group.info

    def update_attributes(self, new_attributes: Dict[str, Any]):
        self._sync(self._async_group.update_attributes, new_attributes)
        return self

    @property
    def nchildren(self) -> int:
        return self._sync(self._async_group.nchildren)

    @property
    def children(self) -> List[Array, "Group"]:
        _children = self._sync_iter(self._async_group.children)
        return [Array(obj) if isinstance(obj, AsyncArray) else Group(obj) for obj in _children]

    def __contains__(self, child) -> bool:
        return self._sync(self._async_group.contains, child)

    def group_keys(self) -> Iterator[str]:
        return self._sync_iter(self._async_group.group_keys)

    def groups(self) -> List["Group"]:
        # TODO: in v2 this was a generator that return key: Group
        return [Group(obj) for obj in self._sync_iter(self._async_group.groups)]

    def array_keys(self) -> List[str]:
        return self._sync_iter(self._async_group.array_keys)

    def arrays(self) -> List[Array]:
        return [Array(obj) for obj in self._sync_iter(self._async_group.arrays)]

    def tree(self, expand=False, level=None) -> Any:
        return self._sync(self._async_group.tree, expand=expand, level=level)

    def create_group(self, name: str, **kwargs) -> "Group":
        return Group(self._sync(self._async_group.create_group, name, **kwargs))

    def create_array(self, name: str, **kwargs) -> Array:
        return Array(self._sync(self._async_group.create_array, name, **kwargs))

    def empty(self, **kwargs) -> "Array":
        return Array(self._sync(self._async_group.empty, **kwargs))

    def zeros(self, **kwargs) -> "Array":
        return Array(self._sync(self._async_group.zeros, **kwargs))

    def ones(self, **kwargs) -> "Array":
        return Array(self._sync(self._async_group.ones, **kwargs))

    def full(self, **kwargs) -> "Array":
        return Array(self._sync(self._async_group.full, **kwargs))

    def empty_like(self, prototype: AsyncArray, **kwargs) -> "Array":
        return Array(self._sync(self._async_group.empty_like, prototype, **kwargs))

    def zeros_like(self, prototype: AsyncArray, **kwargs) -> "Array":
        return Array(self._sync(self._async_group.zeros_like, prototype, **kwargs))

    def ones_like(self, prototype: AsyncArray, **kwargs) -> "Array":
        return Array(self._sync(self._async_group.ones_like, prototype, **kwargs))

    def full_like(self, prototype: AsyncArray, **kwargs) -> "Array":
        return Array(self._sync(self._async_group.full_like, prototype, **kwargs))

    def move(self, source: str, dest: str) -> None:
        return self._sync(self._async_group.move, source, dest)
