from __future__ import annotations
from dataclasses import asdict, dataclass, field

import json
from asyncio import AbstractEventLoop
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
from zarr.v3.abc.metadata import Metadata

from zarr.v3.common import ChunkCoords


@dataclass(frozen=True)
class RuntimeConfiguration:
    order: Literal["C", "F"] = "C"
    concurrency: Optional[int] = None
    asyncio_loop: Optional[AbstractEventLoop] = None


def runtime_configuration(
    order: Literal["C", "F"], concurrency: Optional[int] = None
) -> RuntimeConfiguration:
    return RuntimeConfiguration(order=order, concurrency=concurrency)


class DataType(Enum):
    bool = "bool"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    float32 = "float32"
    float64 = "float64"

    @property
    def byte_count(self) -> int:
        data_type_byte_counts = {
            DataType.bool: 1,
            DataType.int8: 1,
            DataType.int16: 2,
            DataType.int32: 4,
            DataType.int64: 8,
            DataType.uint8: 1,
            DataType.uint16: 2,
            DataType.uint32: 4,
            DataType.uint64: 8,
            DataType.float32: 4,
            DataType.float64: 8,
        }
        return data_type_byte_counts[self]

    def to_numpy_shortname(self) -> str:
        data_type_to_numpy = {
            DataType.bool: "bool",
            DataType.int8: "i1",
            DataType.int16: "i2",
            DataType.int32: "i4",
            DataType.int64: "i8",
            DataType.uint8: "u1",
            DataType.uint16: "u2",
            DataType.uint32: "u4",
            DataType.uint64: "u8",
            DataType.float32: "f4",
            DataType.float64: "f8",
        }
        return data_type_to_numpy[self]


dtype_to_data_type = {
    "|b1": "bool",
    "bool": "bool",
    "|i1": "int8",
    "<i2": "int16",
    "<i4": "int32",
    "<i8": "int64",
    "|u1": "uint8",
    "<u2": "uint16",
    "<u4": "uint32",
    "<u8": "uint64",
    "<f4": "float32",
    "<f8": "float64",
}


@dataclass(frozen=True)
class RegularChunkGridConfigurationMetadata(Metadata):
    chunk_shape: ChunkCoords


@dataclass(frozen=True)
class RegularChunkGridMetadata(Metadata):
    configuration: RegularChunkGridConfigurationMetadata
    name: Literal["regular"] = field(default="regular", init=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(configuration=RegularChunkGridConfigurationMetadata.from_dict(data["configuration"]))

@dataclass(frozen=True)
class DefaultChunkKeyEncodingConfigurationMetadata(Metadata):
    separator: Literal[".", "/"] = "/"


@dataclass(frozen=True)
class DefaultChunkKeyEncodingMetadata(Metadata):
    configuration: DefaultChunkKeyEncodingConfigurationMetadata = (
        DefaultChunkKeyEncodingConfigurationMetadata()
    )
    name: Literal["default"] = "default"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        if chunk_key == "c":
            return ()
        return tuple(map(int, chunk_key[1:].split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.configuration.separator.join(map(str, ("c",) + chunk_coords))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(configuration=DefaultChunkKeyEncodingConfigurationMetadata.from_dict(data["configuration"]))

@dataclass(frozen=True)
class V2ChunkKeyEncodingConfigurationMetadata(Metadata):
    separator: Literal[".", "/"] = "."


@dataclass(frozen=True)
class V2ChunkKeyEncodingMetadata(Metadata):
    configuration: V2ChunkKeyEncodingConfigurationMetadata = (
        V2ChunkKeyEncodingConfigurationMetadata()
    )
    name: Literal["v2"] = "v2"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        return tuple(map(int, chunk_key.split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.configuration.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier


ChunkKeyEncodingMetadata = Union[DefaultChunkKeyEncodingMetadata, V2ChunkKeyEncodingMetadata]


class CodecMetadata(Protocol):
    @property
    def name(self) -> str:
        pass


class ShardingCodecIndexLocation(Enum):
    start = "start"
    end = "end"


@dataclass(frozen=True)
class CoreArrayMetadata:
    shape: ChunkCoords
    chunk_shape: ChunkCoords
    data_type: DataType
    fill_value: Any
    runtime_configuration: RuntimeConfiguration

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)

    @property
    def ndim(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class ArrayMetadata(Metadata):
    shape: ChunkCoords
    data_type: DataType
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: ChunkKeyEncodingMetadata
    fill_value: Any
    codecs: List[CodecMetadata]
    attributes: Dict[str, Any] = field(default_factory=dict)
    dimension_names: Optional[Tuple[str, ...]] = None
    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default='array', init=False)

    def __init__(self, *, shape, data_type, chunk_grid, chunk_key_encoding, fill_value, codecs, attributes, dimension_names):
        """
        Because the class is a frozen dataclass, we have to set attributes using object.__setattr__
        """
        shape_parsed = parse_shape(shape)
        data_type_parsed = parse_dtype(data_type)
        chunk_grid_parsed = parse_chunk_grid(chunk_grid)
        chunk_key_encoding_parsed = parse_chunk_key_encoding(chunk_key_encoding)
        codecs_parsed = parse_codecs(codecs)
        dimension_names_parsed = parse_dimension_names(dimension_names)
        fill_value_parsed = parse_fill_value(fill_value)
        attributes_parsed = parse_attributes(attributes)
        object.__setattr__(self, "", )
        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "data_type", data_type_parsed)
        object.__setattr__(self, "chunk_grid", chunk_grid_parsed)
        object.__setattr__(self, "chunk_key_encoding", chunk_key_encoding_parsed)
        object.__setattr__(self, "codecs", codecs_parsed)
        object.__setattr__(self, "dimension_names", dimension_names_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "attributes", attributes_parsed)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_core_metadata(self, runtime_configuration: RuntimeConfiguration) -> CoreArrayMetadata:
        return CoreArrayMetadata(
            shape=self.shape,
            chunk_shape=self.chunk_grid.configuration.chunk_shape,
            data_type=self.data_type,
            fill_value=self.fill_value,
            runtime_configuration=runtime_configuration,
        )

    def to_bytes(self) -> bytes:
        def _json_convert(o):
            if isinstance(o, Enum):
                return o.name
            raise TypeError

        return json.dumps(
            self.to_dict(),
            default=_json_convert,
        ).encode()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArrayMetadata:
        _ = parse_zarr_format(data.pop('zarr_format'))
        _ = parse_node_type(data.pop('node_type'))

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        self_dict = asdict(self)
        if self_dict['dimension_names'] is None:
            self_dict.pop('dimension_names')
        return self_dict

@dataclass(frozen=True)
class ArrayV2Metadata(Metadata):
    shape: ChunkCoords
    chunks: ChunkCoords
    dtype: np.dtype
    fill_value: Union[None, int, float] = 0
    order: Literal["C", "F"] = "C"
    filters: Optional[List[Dict[str, Any]]] = None
    dimension_separator: Literal[".", "/"] = "."
    compressor: Optional[Dict[str, Any]] = None
    zarr_format: Literal[2] = 2

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to_bytes(self) -> bytes:
        def _json_convert(o):
            if isinstance(o, np.dtype):
                if o.fields is None:
                    return o.str
                else:
                    return o.descr
            raise TypeError

        return json.dumps(asdict(self), default=_json_convert).encode()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArrayV2Metadata:
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
def parse_dtype(data: Any) -> np.dtype:
    return np.dtype(data)

def parse_shape(data: Any) -> Tuple[int, ...]:
    return tuple(int(x) for x in data)

def parse_chunk_grid(data: Any) -> RegularChunkGridMetadata:
    if isinstance(data, dict):
        return RegularChunkGridMetadata.from_dict(data)
    if isinstance(data, RegularChunkGridMetadata):
        return data
    raise TypeError

def parse_chunk_key_encoding(data: Any) -> ChunkKeyEncodingMetadata:
    if isinstance(data, dict):
        if data["name"] == "v2":
            return V2ChunkKeyEncodingMetadata.from_dict(data)
        return RegularChunkGridMetadata.from_dict(data)
    if isinstance(data, RegularChunkGridMetadata):
        return data
    if isinstance(data, V2ChunkKeyEncodingMetadata)
def parse_fill_value(data: Any) -> Any:
    return data

def parse_codecs(data: Any) -> List[CodecMetadata]:
    return data

def parse_dimension_names(data: Any) -> Tuple[str, ...] | None:
    if data is None:
        return data
    return tuple(map(str, data))
    
def parse_attributes(data: Any) -> Any:
    return data

def parse_zarr_format(data: Any) -> Literal[3]:
    if data == 3:
        return data
    f"Invalid value for `zarr_format`, got {data}, expected 3"
    raise ValueError()

def parse_node_type(data: Any) -> Literal[3]:
    if data == "array":
        return data
    msg = f"Invalid value for `node_type`, got {data}, expected 'array'"
    raise ValueError(msg)

    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default='array', init=False)