"""
Models for objects described in zarr version 3
"""

from dataclasses import dataclass
import json
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    runtime_checkable,
)

import numpy as np
import attr

from zarr.v3.types import Attributes


class NamedConfigDict(TypedDict):
    name: str
    configuration: Attributes


# not clear how useful these protocols are, but lets try it
@runtime_checkable
class NamedConfig(Protocol):
    name: str
    configuration: Any


@runtime_checkable
class CodecMetadata(Protocol):
    name: str


class RegularChunkGridConfigDict(TypedDict):
    chunk_shape: tuple[int, ...]


class RegularChunkGridConfig:
    chunk_shape: Tuple[int, ...]

    def __init__(self, chunk_shape) -> None:
        self.chunk_shape = chunk_shape

    def to_dict(self) -> RegularChunkGridConfigDict:
        return {"chunk_shape": self.chunk_shape}


class RegularChunkGridDict(TypedDict):
    configuration: RegularChunkGridConfigDict
    name: str


class RegularChunkGrid(NamedConfig):
    configuration: RegularChunkGridConfig
    name: Literal["regular"] = "regular"

    def __init__(self, configuration: RegularChunkGridConfig) -> None:
        self.configuration = configuration
        self.name = "regular"

    def to_dict(self) -> RegularChunkGridDict:
        return {"configuration": self.configuration.to_dict(), "name": self.name}


class DefaultChunkKeyConfigDict(TypedDict):
    separator: Literal[".", "/"]


class DefaultChunkKeyConfig:
    separator: Literal[".", "/"]

    def __init__(self, *, separator: Literal[".", "/"] = "/") -> None:
        self.separator = parse_dimension_separator

    def to_dict(self) -> DefaultChunkKeyConfigDict:
        return {"separator": self.separator}


def parse_dimension_separator(separator: Any) -> Literal[".", "/"]:
    if separator not in (".", "/"):
        raise ValueError
    return separator


class DefaultChunkKeyEncodingDict(TypedDict):
    configuration: DefaultChunkKeyConfigDict
    name: Literal["default", "v2"]


class DefaultChunkKeyEncoding(NamedConfig):
    configuration: DefaultChunkKeyConfig
    name: Literal["default", "V2"]

    def __init__(self, *, configuration=DefaultChunkKeyConfig(), name="default") -> None:
        self.configuration = configuration
        self.name = name

    def to_dict(self) -> DefaultChunkKeyEncodingDict:
        return {"configuration": self.configuration.to_dict(), "name": self.name}


class V2ChunkKeyEncodingDict(TypedDict):
    configuration: DefaultChunkKeyConfigDict
    name: Literal["V2"]


class V2ChunkKeyEncoding(NamedConfig):
    configuration: DefaultChunkKeyConfig = DefaultChunkKeyConfig()
    name: Literal["V2"] = "V2"

    def __init__(self, configuration: DefaultChunkKeyConfig) -> None:
        self.configuration = configuration
        self.name = "V2"

    def to_dict(self) -> V2ChunkKeyEncodingDict:
        return {"configuration": self.configuration.to_dict(), "name": self.name}


ChunkKeyEncoding = Union[DefaultChunkKeyEncoding, V2ChunkKeyEncoding]


class _ArrayMetadataDictBase(TypedDict):
    """
    This is a private base class with all the required attributes.
    Because `dimension_names` is an optional attribute, we need a subclass to express this.
    See https://peps.python.org/pep-0655/ for a cleaner way
    """

    shape: Tuple[int, ...]
    data_type: str
    chunk_grid: RegularChunkGridDict
    chunk_key_encoding: Union[DefaultChunkKeyConfigDict, V2ChunkKeyEncodingDict]
    fill_value: Any
    codecs: list[NamedConfigDict]
    zarr_format: Literal["3"]
    node_type: Literal["array"]


class ArrayMetadataDict(_ArrayMetadataDictBase, total=False):
    """
    This inherits from a private base class with all the required attributes.
    Because `dimension_names` is an optional attribute, we need a subclass to express this.
    See https://peps.python.org/pep-0655/ for a cleaner way
    """

    dimension_names: list[str]


class ArrayMetadata:
    """
    A representation of v3 array metadata with no behavior besides
    input validation and to / from JSON serialization
    """

    shape: Tuple[int, ...]
    data_type: np.dtype
    chunk_grid: RegularChunkGrid
    chunk_key_encoding: Union[DefaultChunkKeyEncoding, V2ChunkKeyEncoding]
    fill_value: Any
    codecs: list[CodecMetadata]
    dimension_names: Optional[Tuple[str, ...]]
    zarr_format: Literal[3] = 3
    node_type: Literal["array"] = "array"

    def __init__(
        self,
        *,
        shape,
        data_type,
        chunk_grid,
        chunk_key_encoding,
        fill_value,
        codecs,
        dimension_names: Optional[Tuple[str]] = None,
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
        self.dimension_names = parse_dimension_names(dimension_names)
        self = parse_array_metadata(self)

    def to_dict(self) -> ArrayMetadataDict:

        self_dict: ArrayMetadataDict = {
            "shape": self.shape,
            "data_type": self.data_type.str,
            "chunk_grid": self.chunk_grid.to_dict(),
            "fill_value": self.fill_value,
            "chunk_key_encoding": self.chunk_grid.to_dict(),
            "codecs": [codec.to_dict() for codec in self.codecs],
            "node_type": "array",
            "zarr_format": 3,
        }
        if self.dimension_names is not None:
            # dimension names cannot by Null in JSON
            self_dict["dimension_names"] = self.dimension_names

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict()).encode()

    @classmethod
    def from_json(cls, json: bytes) -> "ArrayMetadata":
        ...


class GroupMetadata:
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
    return shape


def parse_data_type(data_type: Any) -> Any:
    return data_type


def parse_chunk_grid(chunk_grid: Any) -> RegularChunkGrid:
    return chunk_grid


def parse_chunk_key_encoding(chunk_key_encoding: Any) -> DefaultChunkKeyEncoding:
    return chunk_key_encoding


def parse_fill_value(fill_value: Any) -> Any:
    return fill_value


def parse_codecs(codecs: Any) -> list[CodecMetadata]:
    return codecs


def parse_dimension_names(dimension_names: Optional[tuple[str, ...]]):
    return dimension_names


def parse_array_metadata(metadata: ArrayMetadata):
    """
    Check that all properties are consistent
    """
    # todo: check that dimensional attributes like shape and dimension_names are consistent
    return metadata


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
