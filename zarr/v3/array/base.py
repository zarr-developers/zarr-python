from __future__ import annotations

import json
from asyncio import AbstractEventLoop
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from attr import frozen

""" from zarr.v3.array import v3
from zarr.v3.array import v2
 """
from zarr.v3.common import BytesLike, ChunkCoords, SliceSelection, to_thread
from zarr.v3.store import StorePath
import numcodecs
from numcodecs.compat import ensure_bytes, ensure_ndarray


@frozen
class RuntimeConfiguration:
    order: Literal["C", "F"] = "C"
    concurrency: Optional[int] = None
    asyncio_loop: Optional[AbstractEventLoop] = None


def runtime_configuration(
    order: Literal["C", "F"], concurrency: Optional[int] = None
) -> RuntimeConfiguration:
    return RuntimeConfiguration(order=order, concurrency=concurrency)


""" class DataType(Enum):
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
        return data_type_to_numpy[self] """


def byte_count(dtype: np.dtype) -> int:
    return dtype.itemsize


def to_numpy_shortname(dtype: np.dtype) -> str:
    return dtype.str.lstrip("|").lstrip("^").lstrip("<").lstrip(">")


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
class ChunkMetadata:
    array_shape: ChunkCoords
    chunk_shape: ChunkCoords
    # data_type: DataType
    dtype: np.dtype
    fill_value: Any
    runtime_configuration: RuntimeConfiguration

    @property
    def ndim(self) -> int:
        return len(self.shape)


""" @frozen
class ZArray:
    shape: ChunkCoords
    chunk_shape: ChunkCoords
    dtype: DataType
    T: "ZArray" 
    size: int
    ndim: int
    attrs: Dict[str, Any]
    order: Literal["C", "F"]
    metadata: Union[v2.ZArrayMetadata, v3.ZArrayMetadata]
    chunk_store: StorePath
    metadata_store: StorePath

    @property
    def ndim(self) -> int:
        return len(self.metadata.shape)
    
    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.metadata.dtype) 
    
    @property
    def size(self) -> int
        return np.prod(self.metadata.shape)
    
    @property
    def T(self) -> 'ZArray':
        ...

    def __getitem__(*args):
        return _chunk_getitem_sync(*args):
    
    def __setitem__(*args):
        return _chunk_setitem_sync(*args)
    """
