from datetime import datetime, timedelta

from dataclasses import dataclass, field, asdict
from typing import Any, Tuple, List, Dict, Optional, Union

from zarr.util import json_dumps, json_loads

# https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
ScalarType = Union[int, float, complex, bytes, str, bool, datetime, timedelta]


class BaseZOM:
    """
    Base class for all Zarr Object Models (ZOMs)
    """

    pass


class _GroupMixin:
    members: Dict[str, Any]


class BaseV2ZOM(BaseZOM):
    """
    Base class for all V2 Zarr Object Models (ZOMs)
    """

    pass


@dataclass
class V2ArrayZOM(BaseV2ZOM):
    """
    V2 Array Zarr Object Model (ZOM)
    """

    shape: Tuple[int, ...]
    chunks: Tuple[int, ...]
    dtype: str
    compressor: Optional[Dict[str, Any]]
    fill_value: Optional[ScalarType]
    order: str
    filters: Optional[List[str]]
    zarr_format: int = field(default=2)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # pydantic would have done this for us :(
        if not isinstance(self.shape, tuple):
            self.shape = tuple(self.shape)
        if not isinstance(self.chunks, tuple):
            self.chunks = tuple(self.chunks)
        if self.zarr_format != 2:
            raise ValueError("zarr_format != 2")

    def serialize(self, **kwargs) -> Dict[str, bytes]:
        """serialize this array into JSON strings

        Returns a dict with two keys (`.zarray` and `.zattrs`)
        """
        data = asdict(self)

        attrs = data.pop("attributes")
        # Q: should we return an empty dict if no attrs
        zattrs = json_dumps(attrs, **kwargs)

        zarray = json_dumps(data, **kwargs)
        return {".zattrs": zattrs, ".zarray": zarray}

    @classmethod
    def deserialize(cls, objs: Dict[str, bytes]):
        zarray = json_loads(objs[".zarray"])
        if ".zattrs" in objs:
            zattrs = json_loads(objs[".zattrs"])
        else:
            zattrs = {}
        return cls(attributes=zattrs, **zarray)


@dataclass
class V2GroupZOM(BaseV2ZOM, _GroupMixin):
    """
    V2 Group Zarr Object Model (ZOM)
    """

    zarr_format: int = field(default=2)
    members: Dict[str, Union["V2ArrayZOM", "V2GroupZOM"]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)  # TODO: replace any with JsonType

    def __post_init__(self):
        if self.zarr_format != 2:
            raise ValueError("zarr_format != 2")

    def serialize(self, **kwargs) -> Dict[str, bytes]:
        data = asdict(self)

        attrs = data.pop("attributes")
        zattrs = json_dumps(attrs, **kwargs)

        del data["members"]  # TODO: decide what to do with members
        zgroup = json_dumps(data, **kwargs)
        return {".zattrs": zattrs, ".zgroup": zgroup}

    @classmethod
    def deserialize(cls, objs: Dict[str, bytes]):
        if ".zgroup" in objs:
            zgroup = json_loads(objs[".zgroup"])
        else:
            zgroup = {}
        if ".zattrs" in objs:
            zattrs = json_loads(objs[".zattrs"])
        else:
            zattrs = {}
        return cls(attributes=zattrs, **zgroup)


@dataclass
class V3RegularChunkConfiguration:
    """
    V3 Regular Chunk Configuration
    """

    chunk_shape: Tuple[int, ...]


@dataclass
class V3ChunkGrid:
    """
    V3 Chunk Grid
    """

    name: str
    configuration: V3RegularChunkConfiguration


@dataclass
class V3ChunkKeyEncodingConfiguration:
    """
    V3 Chunk Key Encoding Configuration
    """

    separator: str


@dataclass
class V3ChunkKeyEncoding:
    """
    V3 Chunk Key Encoding
    """

    name: str
    configuration: V3ChunkKeyEncodingConfiguration


@dataclass
class BaseV3ZOM(BaseZOM):
    """
    Base class for all V3 Zarr Object Models (ZOMs)
    """

    def serialize(self, **kwargs) -> Dict[str, bytes]:
        data = asdict(self)
        _ = data.pop("members", {})  # TODO: decide what to do with members
        zarr_json = json_dumps(data, **kwargs)
        return {"zarr.json": zarr_json}

    @classmethod
    def deserialize(cls, objs: Dict[str, bytes]):
        data = json_loads(objs["zarr.json"])
        return cls(**data)


@dataclass
class V3ArrayZOM(BaseV3ZOM):
    """
    V3 Array Zarr Object Model (ZOM)
    """

    shape: Tuple[int, ...]
    dtype: str
    chunk_grid: V3ChunkGrid
    chunk_key_encoding: V3ChunkKeyEncoding
    fill_value: Any
    codecs: List[Any]
    storage_transformers: List[Any]
    node_type: str = field(default="array")
    zarr_format: int = field(default=3)
    dimension_names: Optional[List[str]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.zarr_format != 3:
            raise ValueError("zarr_format != 3")
        if self.node_type != "array":
            raise ValueError(f"node_type ({self.node_type}) != array")
        if not isinstance(self.shape, tuple):
            self.shape = tuple(self.shape)
        if not self.storage_transformers:
            self.storage_transformers = []
        if not self.storage_transformers:
            self.storage_transformers = []


@dataclass
class V3GroupZOM(BaseV3ZOM, _GroupMixin):
    """
    V3 Group Zarr Object Model (ZOM)
    """

    zarr_format: int = field(default=3)
    node_type: str = field(default="group")
    members: Dict[str, Union["V3ArrayZOM", "V3GroupZOM"]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.zarr_format != 3:
            raise ValueError("zarr_format != 3")
        if self.node_type != "group":
            raise ValueError(f"node_type ({self.node_type}) != group")
