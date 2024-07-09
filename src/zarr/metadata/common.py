from __future__ import annotations

import itertools
import operator
from collections.abc import Iterable, Iterator
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal

from zarr.abc.codec import Codec
from zarr.codecs.registry import get_codec_class
from zarr.common import (
    ChunkCoords,
    ChunkCoordsLike,
    MemoryOrder,
    parse_fill_value,
    parse_named_configuration,
)
from zarr.indexing import ceildiv

if TYPE_CHECKING:
    from typing import Any, Literal

    from typing_extensions import Self

    from zarr.common import JSON, ChunkCoords, ZarrFormat

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from zarr.abc.metadata import Metadata
from zarr.buffer import Buffer, BufferPrototype


def parse_attributes(data: None | dict[str, JSON]) -> dict[str, JSON]:
    if data is None:
        return {}

    return data


@dataclass(frozen=True)
class ArraySpec:
    shape: ChunkCoords
    dtype: np.dtype[Any]
    fill_value: Any
    order: Literal["C", "F"]
    prototype: BufferPrototype

    def __init__(
        self,
        shape: ChunkCoords,
        dtype: np.dtype[Any],
        fill_value: Any,
        order: Literal["C", "F"],
        prototype: BufferPrototype,
    ) -> None:
        shape_parsed = parse_shapelike(shape)
        dtype_parsed = parse_dtype(dtype)
        fill_value_parsed = parse_fill_value(fill_value)
        order_parsed = parse_order(order)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "dtype", dtype_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "order", order_parsed)
        object.__setattr__(self, "prototype", prototype)

    @classmethod
    def from_array(cls, array: Any, order: MemoryOrder, prototype: BufferPrototype) -> Self:
        return cls(
            shape=array.chunk_grid.chunk_shape,
            dtype=array.dtype,
            fill_value=array.fill_value,
            order=order,
            prototype=prototype,
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class ChunkGrid(Metadata):
    @classmethod
    def from_dict(cls, data: dict[str, JSON] | ChunkGrid) -> ChunkGrid:
        if isinstance(data, ChunkGrid):
            return data

        name_parsed, _ = parse_named_configuration(data)
        if name_parsed == "regular":
            return RegularChunkGrid._from_dict(data)
        raise ValueError(f"Unknown chunk grid. Got {name_parsed}.")

    @abstractmethod
    def all_chunk_coords(self, array_shape: ChunkCoords) -> Iterator[ChunkCoords]:
        pass

    @abstractmethod
    def get_nchunks(self, array_shape: ChunkCoords) -> int:
        pass


@dataclass(frozen=True)
class RegularChunkGrid(ChunkGrid):
    chunk_shape: ChunkCoords

    def __init__(self, *, chunk_shape: ChunkCoordsLike) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)

    @classmethod
    def _from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "regular")

        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "regular", "configuration": {"chunk_shape": list(self.chunk_shape)}}

    def all_chunk_coords(self, array_shape: ChunkCoords) -> Iterator[ChunkCoords]:
        return itertools.product(
            *(range(0, ceildiv(s, c)) for s, c in zip(array_shape, self.chunk_shape, strict=False))
        )

    def get_nchunks(self, array_shape: ChunkCoords) -> int:
        return reduce(
            operator.mul,
            (ceildiv(s, c) for s, c in zip(array_shape, self.chunk_shape, strict=True)),
            1,
        )


@dataclass(frozen=True, kw_only=True)
class ArrayMetadataBase(Metadata, ABC):
    shape: ChunkCoords
    fill_value: Any
    chunk_grid: ChunkGrid
    attributes: dict[str, JSON]
    zarr_format: ZarrFormat

    @property
    @abstractmethod
    def dtype(self) -> np.dtype[Any]:
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @abstractmethod
    def get_chunk_spec(
        self, _chunk_coords: ChunkCoords, order: Literal["C", "F"], prototype: BufferPrototype
    ) -> ArraySpec:
        pass

    @abstractmethod
    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        pass

    @abstractmethod
    def to_buffer_dict(self) -> dict[str, Buffer]:
        pass

    @abstractmethod
    def update_shape(self, shape: ChunkCoords) -> Self:
        pass

    @abstractmethod
    def update_attributes(self, attributes: dict[str, JSON]) -> Self:
        pass


def parse_shapelike(data: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(data, int):
        if data < 0:
            raise ValueError(f"Expected a non-negative integer. Got {data} instead")
        return (data,)
    try:
        data_tuple = tuple(data)
    except TypeError as e:
        msg = f"Expected an integer or an iterable of integers. Got {data} instead."
        raise TypeError(msg) from e

    if not all(isinstance(v, int) for v in data_tuple):
        msg = f"Expected an iterable of integers. Got {data} instead."
        raise TypeError(msg)
    if not all(v > -1 for v in data_tuple):
        msg = f"Expected all values to be non-negative. Got {data} instead."
        raise ValueError(msg)
    return data_tuple


def parse_dtype(data: Any) -> np.dtype[Any]:
    # todo: real validation
    return np.dtype(data)


def parse_order(data: Any) -> Literal["C", "F"]:
    if data == "C":
        return "C"
    if data == "F":
        return "F"
    raise ValueError(f"Expected one of ('C', 'F'), got {data} instead.")


def parse_codecs(data: Any) -> tuple[Codec, ...]:
    from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec

    out: tuple[Codec, ...] = ()

    if not isinstance(data, Iterable):
        raise TypeError(f"Expected iterable, got {type(data)}")

    for c in data:
        if isinstance(
            c, ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec
        ):  # Can't use Codec here because of mypy limitation
            out += (c,)
        else:
            name_parsed, _ = parse_named_configuration(c, require_configuration=False)
            out += (get_codec_class(name_parsed).from_dict(c),)

    return out
