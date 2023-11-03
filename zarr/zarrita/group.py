from __future__ import annotations

import json
from typing import Any, Dict, Literal, Optional, Union

from attr import asdict, evolve, field, frozen

from zarrita.array import Array
from zarrita.common import ZARR_JSON, make_cattr
from zarrita.metadata import RuntimeConfiguration
from zarrita.store import StoreLike, StorePath, make_store_path
from zarrita.sync import sync


@frozen
class GroupMetadata:
    attributes: Dict[str, Any] = field(factory=dict)
    zarr_format: Literal[3] = 3
    node_type: Literal["group"] = "group"

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_json(cls, zarr_json: Any) -> GroupMetadata:
        return make_cattr().structure(zarr_json, GroupMetadata)


@frozen
class Group:
    metadata: GroupMetadata
    store_path: StorePath
    runtime_configuration: RuntimeConfiguration

    @classmethod
    async def create_async(
        cls,
        store: StoreLike,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        store_path = make_store_path(store)
        if not exists_ok:
            assert not await (store_path / ZARR_JSON).exists_async()
        group = cls(
            metadata=GroupMetadata(attributes=attributes or {}),
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
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        return sync(
            cls.create_async(
                store,
                attributes=attributes,
                exists_ok=exists_ok,
                runtime_configuration=runtime_configuration,
            ),
            runtime_configuration.asyncio_loop,
        )

    @classmethod
    async def open_async(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        store_path = make_store_path(store)
        zarr_json_bytes = await (store_path / ZARR_JSON).get_async()
        assert zarr_json_bytes is not None
        return cls.from_json(
            store_path, json.loads(zarr_json_bytes), runtime_configuration
        )

    @classmethod
    def open(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Group:
        return sync(
            cls.open_async(store, runtime_configuration),
            runtime_configuration.asyncio_loop,
        )

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

    @classmethod
    async def open_or_array(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Union[Array, Group]:
        store_path = make_store_path(store)
        zarr_json_bytes = await (store_path / ZARR_JSON).get_async()
        if zarr_json_bytes is None:
            raise KeyError
        zarr_json = json.loads(zarr_json_bytes)
        if zarr_json["node_type"] == "group":
            return cls.from_json(store_path, zarr_json, runtime_configuration)
        if zarr_json["node_type"] == "array":
            return Array.from_json(
                store_path, zarr_json, runtime_configuration=runtime_configuration
            )
        raise KeyError

    async def _save_metadata(self) -> None:
        await (self.store_path / ZARR_JSON).set_async(self.metadata.to_bytes())

    async def get_async(self, path: str) -> Union[Array, Group]:
        return await self.__class__.open_or_array(
            self.store_path / path, self.runtime_configuration
        )

    def __getitem__(self, path: str) -> Union[Array, Group]:
        return sync(self.get_async(path), self.runtime_configuration.asyncio_loop)

    async def create_group_async(self, path: str, **kwargs) -> Group:
        runtime_configuration = kwargs.pop(
            "runtime_configuration", self.runtime_configuration
        )
        return await self.__class__.create_async(
            self.store_path / path,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )

    def create_group(self, path: str, **kwargs) -> Group:
        return sync(
            self.create_group_async(path), self.runtime_configuration.asyncio_loop
        )

    async def create_array_async(self, path: str, **kwargs) -> Array:
        runtime_configuration = kwargs.pop(
            "runtime_configuration", self.runtime_configuration
        )
        return await Array.create_async(
            self.store_path / path,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )

    def create_array(self, path: str, **kwargs) -> Array:
        return sync(
            self.create_array_async(path, **kwargs),
            self.runtime_configuration.asyncio_loop,
        )

    async def update_attributes_async(self, new_attributes: Dict[str, Any]) -> Group:
        new_metadata = evolve(self.metadata, attributes=new_attributes)

        # Write new metadata
        await (self.store_path / ZARR_JSON).set_async(new_metadata.to_bytes())
        return evolve(self, metadata=new_metadata)

    def update_attributes(self, new_attributes: Dict[str, Any]) -> Group:
        return sync(
            self.update_attributes_async(new_attributes),
            self.runtime_configuration.asyncio_loop,
        )

    def __repr__(self):
        return f"<Group {self.store_path}>"
