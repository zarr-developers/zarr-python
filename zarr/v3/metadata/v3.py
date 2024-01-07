"""
Models for objects described in zarr version 3
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
from attr import frozen

from zarr.v3.types import Attributes

# not clear how useful these protocols are, but lets try it
@runtime_checkable
class NamedConfig(Protocol):
    name: str
    configuration: Any


@runtime_checkable
class CodecMetadata(Protocol):
    name: str


@frozen
class RegularChunkGridConfig:
    chunk_shape: Tuple[int, ...]


@frozen
class RegularChunkGrid(NamedConfig):
    configuration: RegularChunkGridConfig
    name: Literal["regular"] = "regular"


@frozen
class DefaultChunkKeyConfig:
    separator: Literal[".", "/"] = "/"


@frozen
class DefaultChunkKeyEncoding(NamedConfig):
    configuration: DefaultChunkKeyConfig = DefaultChunkKeyConfig()
    name: Literal["default", "V2"] = "default"


@frozen
class V2ChunkKeyEncoding(NamedConfig):
    configuration: DefaultChunkKeyConfig = DefaultChunkKeyConfig()
    name: Literal["default"] = "V2"


ChunkKeyEncoding = Union[DefaultChunkKeyEncoding, V2ChunkKeyEncoding]


@dataclass(frozen=True)
class ArrayMetadata:
    """
    A representation of v3 array metadata with no behavior besides
    input validation and to / from JSON serialization
    """

    shape: Tuple[int, ...]
    data_type: np.dtype
    chunk_grid: RegularChunkGrid
    chunk_key_encoding: DefaultChunkKeyEncoding
    fill_value: Any
    codecs: list[CodecMetadata]
    attributes: Attributes
    dimension_names: Optional[Tuple[str, ...]] = None
    zarr_format: Literal[3] = 3
    node_type: Literal["array"] = "array"

    def __init__(
        self,
        shape,
        data_type,
        chunk_grid,
        chunk_key_encoding,
        fill_value,
        codecs,
        attributes,
    ):
        """
        The only thing we need to do here is validate inputs.
        """
        self.shape = parse_shape(shape)
        self.data_type = parse_data_type(data_type)
        self.chunk_grid = parse_chunk_grid(chunk_grid)
        self.chunk_key_encoding = parse_chunk_key_encoding(chunk_key_encoding)
        self.fill_value = parse_fill_value(fill_value)
        self.codecs = parse_codecs(codecs)
        self.attributes = parse_attributes(attributes)
        parse_metadata(self)

    @classmethod
    def from_json(cls, json: bytes) -> "ArrayMetadata":
        ...

    def to_json(self) -> bytes:
        ...


@dataclass(frozen=True)
class GroupMetadata:
    attributes: Attributes
    node_type: Literal["group"] = "group"

    @classmethod
    def from_json(cls, json: bytes) -> "GroupMetadata":
        ...

    def to_json(self) -> bytes:
        ...


def from_json(blob: bytes) -> Union[ArrayMetadata, GroupMetadata]:
    """The class methods can very lightly wrap this function"""
    ...


def to_json(obj: Union[ArrayMetadata, GroupMetadata]) -> bytes:
    """The class methods can very lightly wrap this function"""
    ...


def parse_shape(shape: Any) -> Tuple[int, ...]:
    ...


def parse_data_type(data_type: Any) -> Any:
    ...


def parse_chunk_grid(chunk_grid: Any) -> RegularChunkGrid:
    ...


def parse_chunk_key_encoding(chunk_key_encoding: Any) -> DefaultChunkKeyEncoding:
    ...


def parse_fill_value(fill_value: Any) -> Any:
    ...


def parse_codecs(codecs: Any) -> list[CodecMetadata]:
    ...


def parse_attributes(attributes: Any) -> Attributes:
    ...


def parse_metadata(metadata: ArrayMetadata):
    """
    Check that all properties are consistent
    """
    ...


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


def byte_count(dtype: np.dtype) -> int:
    return dtype.itemsize


def to_numpy_shortname(dtype: np.dtype) -> str:
    return dtype.str.lstrip("|").lstrip("^").lstrip("<").lstrip(">")
