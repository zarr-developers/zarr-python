from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

from attr import asdict, evolve, frozen

from zarr.v3.array.v2 import AsyncArray
from zarr.v3.common import ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, make_cattr
from zarr.v3.array.base import RuntimeConfiguration
from zarr.v3.store import StoreLike, StorePath, make_store_path
from zarr.v3.sync import sync
from zarr.v3.types import Attributes

if TYPE_CHECKING:
    from zarr.v3.group import Group


@frozen
class GroupV2Metadata:
    zarr_format: Literal[2] = 2

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_json(cls, zarr_json: Any) -> GroupV2Metadata:
        return make_cattr().structure(zarr_json, cls)


@frozen
class GroupV2:
    metadata: GroupV2Metadata
    store_path: StorePath
    runtime_configuration: RuntimeConfiguration
    attributes: Optional[Attributes] = None

    @classmethod
    async def create_async(
        cls,
        store: StoreLike,
        *,
        attributes: Optional[Attributes] = None,
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> GroupV2:
        store_path = make_store_path(store)
        if not exists_ok:
            assert not await (store_path / ZGROUP_JSON).exists_async()
        group = cls(
            metadata=GroupV2Metadata(),
            attributes=attributes,
            store_path=store_path,
            runtime_configuration=runtime_configuration,
        )
        await group._save_metadata()
        return group

    @classmethod
    def create(
        cls,
        store: StoreLike,
        *,
        attributes: Optional[Attributes] = None,
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> GroupV2:
        return sync(
            cls.create_async(
                store,
                attributes=attributes,
                exists_ok=exists_ok,
                runtime_configuration=runtime_configuration,
            ),
            runtime_configuration.asyncio_loop if runtime_configuration else None,
        )

    @classmethod
    async def open_async(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> GroupV2:
        store_path = make_store_path(store)
        zgroup_bytes = await (store_path / ZGROUP_JSON).get_async()
        assert zgroup_bytes is not None
        zattrs_bytes = await (store_path / ZATTRS_JSON).get_async()
        metadata = json.loads(zgroup_bytes)
        attributes = json.loads(zattrs_bytes) if zattrs_bytes is not None else None

        return cls.from_json(
            store_path,
            metadata,
            runtime_configuration,
            attributes,
        )

    @classmethod
    def open(
        cls,
        store_path: StorePath,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> GroupV2:
        return sync(
            cls.open_async(store_path, runtime_configuration),
            runtime_configuration.asyncio_loop,
        )

    @classmethod
    def from_json(
        cls,
        store_path: StorePath,
        zarr_json: Any,
        runtime_configuration: RuntimeConfiguration,
        attributes: Optional[Attributes] = None,
    ) -> GroupV2:
        group = cls(
            metadata=GroupV2Metadata.from_json(zarr_json),
            store_path=store_path,
            runtime_configuration=runtime_configuration,
            attributes=attributes,
        )
        return group

    @staticmethod
    async def open_or_array(
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Union[AsyncArray, GroupV2]:
        store_path = make_store_path(store)
        zgroup_bytes, zattrs_bytes = await asyncio.gather(
            (store_path / ZGROUP_JSON).get_async(),
            (store_path / ZATTRS_JSON).get_async(),
        )
        attributes = json.loads(zattrs_bytes) if zattrs_bytes is not None else None
        if zgroup_bytes is not None:
            return GroupV2.from_json(
                store_path, json.loads(zgroup_bytes), runtime_configuration, attributes
            )
        zarray_bytes = await (store_path / ZARRAY_JSON).get_async()
        if zarray_bytes is not None:
            return AsyncArray.from_json(
                store_path, json.loads(zarray_bytes), attributes, runtime_configuration
            )
        raise KeyError

    async def _save_metadata(self) -> None:
        await (self.store_path / ZGROUP_JSON).set_async(self.metadata.to_bytes())
        if self.attributes is not None and len(self.attributes) > 0:
            await (self.store_path / ZATTRS_JSON).set_async(
                json.dumps(self.attributes).encode(),
            )
        else:
            await (self.store_path / ZATTRS_JSON).delete_async()

    async def get_async(self, path: str) -> Union[AsyncArray, GroupV2]:
        return await self.__class__.open_or_array(
            self.store_path / path, self.runtime_configuration
        )

    def __getitem__(self, path: str) -> Union[AsyncArray, GroupV2]:
        return sync(self.get_async(path), self.runtime_configuration.asyncio_loop)

    async def create_group_async(self, path: str, **kwargs) -> GroupV2:
        runtime_configuration = kwargs.pop("runtime_configuration", self.runtime_configuration)
        return await self.__class__.create_async(
            self.store_path / path,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )

    def create_group(self, path: str, **kwargs) -> GroupV2:
        return sync(self.create_group_async(path), self.runtime_configuration.asyncio_loop)

    async def create_array_async(self, path: str, **kwargs) -> AsyncArray:
        runtime_configuration = kwargs.pop("runtime_configuration", self.runtime_configuration)
        return await AsyncArray.create(
            self.store_path / path,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )

    def create_array(self, path: str, **kwargs) -> AsyncArray:
        return sync(
            self.create_array_async(path, **kwargs),
            self.runtime_configuration.asyncio_loop,
        )

    async def convert_to_v3_async(self) -> Group:
        from zarr.v3.common import ZARR_JSON
        from zarr.v3.group import Group, GroupMetadata

        new_metadata = GroupMetadata(attributes=self.attributes or {})
        new_metadata_bytes = new_metadata.to_bytes()

        await (self.store_path / ZARR_JSON).set_async(new_metadata_bytes)

        return Group.from_json(
            store_path=self.store_path,
            zarr_json=json.loads(new_metadata_bytes),
            runtime_configuration=self.runtime_configuration,
        )

    async def update_attributes_async(self, new_attributes: Attributes) -> GroupV2:
        await (self.store_path / ZATTRS_JSON).set_async(json.dumps(new_attributes).encode())
        return evolve(self, attributes=new_attributes)

    def update_attributes(self, new_attributes: Attributes) -> GroupV2:
        return sync(
            self.update_attributes_async(new_attributes),
            self.runtime_configuration.asyncio_loop,
        )

    def convert_to_v3(self) -> Group:
        return sync(self.convert_to_v3_async(), loop=self.runtime_configuration.asyncio_loop)

    def __repr__(self):
        return f"<Group_v2 {self.store_path}>"
