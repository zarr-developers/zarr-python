from __future__ import annotations

import json
from asyncio import AbstractEventLoop
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from attr import asdict, field, frozen

from zarrita.common import ChunkCoords, make_cattr


@frozen
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


@frozen
class RegularChunkGridConfigurationMetadata:
    chunk_shape: ChunkCoords


@frozen
class RegularChunkGridMetadata:
    configuration: RegularChunkGridConfigurationMetadata
    name: Literal["regular"] = "regular"


@frozen
class DefaultChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "/"


@frozen
class DefaultChunkKeyEncodingMetadata:
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


@frozen
class V2ChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "."


@frozen
class V2ChunkKeyEncodingMetadata:
    configuration: V2ChunkKeyEncodingConfigurationMetadata = (
        V2ChunkKeyEncodingConfigurationMetadata()
    )
    name: Literal["v2"] = "v2"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        return tuple(map(int, chunk_key.split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.configuration.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier


ChunkKeyEncodingMetadata = Union[
    DefaultChunkKeyEncodingMetadata, V2ChunkKeyEncodingMetadata
]


BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]


@frozen
class BloscCodecConfigurationMetadata:
    typesize: int
    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"] = "zstd"
    clevel: int = 5
    shuffle: BloscShuffle = "noshuffle"
    blocksize: int = 0


blosc_shuffle_int_to_str: Dict[int, BloscShuffle] = {
    0: "noshuffle",
    1: "shuffle",
    2: "bitshuffle",
}


@frozen
class BloscCodecMetadata:
    configuration: BloscCodecConfigurationMetadata
    name: Literal["blosc"] = "blosc"


@frozen
class BytesCodecConfigurationMetadata:
    endian: Optional[Literal["big", "little"]] = "little"


@frozen
class BytesCodecMetadata:
    configuration: BytesCodecConfigurationMetadata
    name: Literal["bytes"] = "bytes"


@frozen
class TransposeCodecConfigurationMetadata:
    order: Union[Literal["C", "F"], Tuple[int, ...]] = "C"


@frozen
class TransposeCodecMetadata:
    configuration: TransposeCodecConfigurationMetadata
    name: Literal["transpose"] = "transpose"


@frozen
class GzipCodecConfigurationMetadata:
    level: int = 5


@frozen
class GzipCodecMetadata:
    configuration: GzipCodecConfigurationMetadata
    name: Literal["gzip"] = "gzip"


@frozen
class ZstdCodecConfigurationMetadata:
    level: int = 0
    checksum: bool = False


@frozen
class ZstdCodecMetadata:
    configuration: ZstdCodecConfigurationMetadata
    name: Literal["zstd"] = "zstd"


@frozen
class Crc32cCodecMetadata:
    name: Literal["crc32c"] = "crc32c"


@frozen
class ShardingCodecConfigurationMetadata:
    chunk_shape: ChunkCoords
    codecs: List["CodecMetadata"]
    index_codecs: List["CodecMetadata"]


@frozen
class ShardingCodecMetadata:
    configuration: ShardingCodecConfigurationMetadata
    name: Literal["sharding_indexed"] = "sharding_indexed"


CodecMetadata = Union[
    BloscCodecMetadata,
    BytesCodecMetadata,
    TransposeCodecMetadata,
    GzipCodecMetadata,
    ZstdCodecMetadata,
    ShardingCodecMetadata,
    Crc32cCodecMetadata,
]


@frozen
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


@frozen
class ArrayMetadata:
    shape: ChunkCoords
    data_type: DataType
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: ChunkKeyEncodingMetadata
    fill_value: Any
    codecs: List[CodecMetadata]
    attributes: Dict[str, Any] = field(factory=dict)
    dimension_names: Optional[Tuple[str, ...]] = None
    zarr_format: Literal[3] = 3
    node_type: Literal["array"] = "array"

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_core_metadata(
        self, runtime_configuration: RuntimeConfiguration
    ) -> CoreArrayMetadata:
        return CoreArrayMetadata(
            shape=self.shape,
            chunk_shape=self.chunk_grid.configuration.chunk_shape,
            data_type=self.data_type,
            fill_value=self.fill_value,
            runtime_configuration=runtime_configuration,
        )

    def to_bytes(self) -> bytes:
        def _json_convert(o):
            if isinstance(o, DataType):
                return o.name
            raise TypeError

        return json.dumps(
            asdict(
                self,
                filter=lambda attr, value: attr.name != "dimension_names"
                or value is not None,
            ),
            default=_json_convert,
        ).encode()

    @classmethod
    def from_json(cls, zarr_json: Any) -> ArrayMetadata:
        return make_cattr().structure(zarr_json, cls)


@frozen
class ArrayV2Metadata:
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
    def from_json(cls, zarr_json: Any) -> ArrayV2Metadata:
        return make_cattr().structure(zarr_json, cls)
