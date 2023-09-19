from dataclasses import dataclass, field
from typing import Any, Tuple, List, Dict, Optional, Union


@dataclass
class BaseZOM:
    """
    Base class for all Zarr Object Models (ZOMs)
    """

    pass


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
    compressor: str
    fill_value: Any
    order: str
    filters: List[str]
    zarr_format: int = field(init=False, default=2)
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class V2GroupZOM(BaseV2ZOM):
    """
    V2 Group Zarr Object Model (ZOM)
    """

    members: Dict[str, Union["V2ArrayZOM", "V2GroupZOM"]] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseV3ZOM(BaseZOM):
    """
    Base class for all V3 Zarr Object Models (ZOMs)
    """

    zarr_format: int
    node_type: str


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
    dimension_names: Optional[List[str]] = None
    node_type: str = field(init=False, default="array")
    attrs: Dict[str, Any] = field(default_factory=dict)
    zarr_format: int = field(init=False, default=3)


@dataclass
class V3GroupZOM(BaseV3ZOM):
    """
    V3 Group Zarr Object Model (ZOM)
    """

    zarr_format: int = field(init=False, default=3)
    node_type: str = field(init=False, default="group")
    members: Dict[str, Union["V3ArrayZOM", "V3GroupZOM"]] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)
