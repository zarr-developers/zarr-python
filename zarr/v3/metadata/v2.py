"""
Models for objects described in zarr version 2
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from numcodecs.abc import Codec
import numpy as np

from zarr.v3.types import Attributes


@dataclass(frozen=True)
class ArrayMetadata:
    """
    A representation of v2 array metadata with no behavior besides
    input validation and to / from JSON serialization
    """

    shape: Tuple[int, ...]
    data_type: np.dtype
    chunks: Tuple[int, ...]
    fill_value: Any
    filters: Optional[List[Codec]]
    compressor: Optional[Codec]
    attrs: Attributes
    zarr_format: Literal[2] = 2

    def __init__(self, shape, data_type, chunks, fill_value, filters, compressor, attrs):
        """
        The only thing we need to do here is validate inputs.
        """
        self.shape = parse_shape(shape)
        self.data_type = parse_data_type(data_type)
        self.chunks = parse_chunks(chunks)
        self.fill_value = parse_fill_value(fill_value)
        self.compressor = parse_compressor(compressor)
        self.filters = parse_filters(filters)
        self.attrs = parse_attrs(attrs)

        parse_metadata(self)

    @classmethod
    def from_json(cls, json: bytes) -> "ArrayMetadata":
        ...

    def to_json(self) -> bytes:
        ...


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
    ...


def parse_data_type(data_type: Any) -> np.dtype:
    ...


def parse_chunks(chunks: Any) -> Tuple[int, ...]:
    ...


def parse_fill_value(fill_value: Any) -> Any:
    ...


def parse_compressor(compressor: Any) -> Codec:
    ...


def parse_filters(filters: Any) -> Optional[List[Codec]]:
    ...


def parse_attrs(attrs: Any) -> Attributes:
    ...


def parse_metadata(metadata: ArrayMetadata):
    """
    Check that all properties are consistent
    """
    ...
