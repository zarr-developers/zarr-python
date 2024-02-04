from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace

import json
from typing import Any, Dict, Literal, Optional, Union
from zarr.v3.abc.metadata import Metadata


from zarr.v3.array import Array
from zarr.v3.common import ZARR_JSON
from zarr.v3.metadata import RuntimeConfiguration
from zarr.v3.store import StoreLike, StorePath, make_store_path
from zarr.v3.sync import sync


@dataclass(frozen=True)
class GroupMetadata(Metadata):
    attributes: Dict[str, Any] = field(default_factory=dict)
    zarr_format: Literal[3] = 3
    node_type: Literal["group"] = "group"

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroupMetadata:
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
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
        return cls.from_dict(store_path, json.loads(zarr_json_bytes), runtime_configuration)

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
    def from_dict(
        cls,
        store_path: StorePath,
        data: Dict[str, Any],
        runtime_configuration: RuntimeConfiguration,
    ) -> Group:
        group = cls(
            metadata=GroupMetadata.from_dict(data),
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
            return cls.from_dict(store_path, zarr_json, runtime_configuration)
        if zarr_json["node_type"] == "array":
            return Array.from_dict(
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
        runtime_configuration = kwargs.pop("runtime_configuration", self.runtime_configuration)
        return await self.__class__.create_async(
            self.store_path / path,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )

    def create_group(self, path: str, **kwargs) -> Group:
        return sync(self.create_group_async(path), self.runtime_configuration.asyncio_loop)

    async def create_array_async(self, path: str, **kwargs) -> Array:
        runtime_configuration = kwargs.pop("runtime_configuration", self.runtime_configuration)
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
        new_metadata = replace(self.metadata, attributes=new_attributes)

        # Write new metadata
        await (self.store_path / ZARR_JSON).set_async(new_metadata.to_bytes())
        return replace(self, metadata=new_metadata)

    def update_attributes(self, new_attributes: Dict[str, Any]) -> Group:
        return sync(
            self.update_attributes_async(new_attributes),
            self.runtime_configuration.asyncio_loop,
        )

    def __repr__(self):
        return f"<Group {self.store_path}>"
