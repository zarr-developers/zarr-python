"""
Models for objects described in zarr version 2
"""

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from numcodecs.abc import Codec
import numpy as np

from zarr.v3.types import Attributes

from typing import TypedDict

V2CodecDict = Dict[str, Attributes]


class ArrayMetadataDict(TypedDict):
    shape: Tuple[int, ...]
    dtype: np.dtype
    chunks: tuple[int, ...]
    fill_value: Any
    filters: Optional[list[V2CodecDict]]
    compressor: V2CodecDict
    zarr_format: Literal["2"]


class ArrayMetadata:
    """
    A representation of v2 array metadata with no behavior besides
    input validation and to / from JSON serialization
    """

    shape: Tuple[int, ...]
    dtype: np.dtype
    chunks: Tuple[int, ...]
    fill_value: Any
    filters: Optional[List[Codec]]
    dimension_separator: Literal["/", "."]
    order: Literal["C", "F"]
    compressor: Optional[Codec]
    zarr_format: Literal[2] = 2

    def __init__(
        self, shape, dtype, chunks, fill_value, filters, dimension_separator, order, compressor
    ):
        """
        The only thing we need to do here is validate inputs.
        """
        self.shape = parse_shape(shape)
        self.dtype = parse_data_type(dtype)
        self.chunks = parse_chunks(chunks)
        self.fill_value = parse_fill_value(fill_value)
        self.order = parse_order(order)
        self.compressor = parse_compressor(compressor)
        self.filters = parse_filters(filters)
        self.dimension_separator = parse_dimension_separator(dimension_separator)

        self = parse_metadata(self)

    @classmethod
    def from_json(cls, json: bytes) -> "ArrayMetadata":
        ...

    def to_dict(self) -> ArrayMetadataDict:
        if self.compressor is not None:
            compressor = self.compressor.get_config()
        else:
            compressor = self.compressor
        if self.filters is not None:
            filters = [f.get_config() for f in self.filters]
        else:
            filters = self.filters
        return {
            "shape": self.shape,
            "dtype": self.dtype.str,
            "chunks": self.chunks,
            "fill_value": self.fill_value,
            "order": self.order,
            "compressor": compressor,
            "filters": filters,
            "dimension_separator": self.dimension_separator,
        }

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict()).encode()


@dataclass(frozen=True)
class GroupMetadata:
    attrs: Attributes

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


def parse_data_type(data_type: Any) -> np.dtype:
    return data_type


def parse_chunks(chunks: Any) -> Tuple[int, ...]:
    return chunks


def parse_order(order: Any) -> Literal["C", "F"]:
    return order


def parse_fill_value(fill_value: Any) -> Any:
    return fill_value


def parse_compressor(compressor: Any) -> Codec:
    return compressor


def parse_filters(filters: Any) -> Optional[List[Codec]]:
    return filters


def parse_dimension_separator(dimension_separator: Any) -> Literal["/", "."]:
    return dimension_separator


def parse_attrs(attrs: Any) -> Attributes:
    return attrs


def parse_metadata(metadata: ArrayMetadata):
    """
    Check that all properties are consistent
    """
    return metadata
