from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec, Codec, CodecPipeline
from zarr.abc.metadata import Metadata
from zarr.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.chunk_grids import ChunkGrid, RegularChunkGrid
from zarr.chunk_key_encodings import ChunkKeyEncoding, parse_separator
from zarr.codecs.registry import get_codec_class
from zarr.config import config

if TYPE_CHECKING:
    from typing_extensions import Self

import numcodecs.abc

from zarr.array_spec import ArraySpec
from zarr.common import (
    JSON,
    ZARR_JSON,
    ZARRAY_JSON,
    ZATTRS_JSON,
    ChunkCoords,
    ZarrFormat,
    parse_dtype,
    parse_fill_value,
    parse_named_configuration,
    parse_shapelike,
)
from zarr.config import parse_indexing_order

# For type checking
_bool = bool


__all__ = ["ArrayMetadata"]


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

    @property
    def has_endianness(self) -> _bool:
        # This might change in the future, e.g. for a complex with 2 8-bit floats
        return self.byte_count != 1

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

    @classmethod
    def from_dtype(cls, dtype: np.dtype[Any]) -> DataType:
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
        return DataType[dtype_to_data_type[dtype.str]]


@dataclass(frozen=True, kw_only=True)
class ArrayMetadata(Metadata, ABC):
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


@dataclass(frozen=True, kw_only=True)
class ArrayV3Metadata(ArrayMetadata):
    shape: ChunkCoords
    data_type: np.dtype[Any]
    chunk_grid: ChunkGrid
    chunk_key_encoding: ChunkKeyEncoding
    fill_value: Any
    codecs: tuple[Codec, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    dimension_names: tuple[str, ...] | None = None
    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default="array", init=False)

    def __init__(
        self,
        *,
        shape: Iterable[int],
        data_type: npt.DTypeLike,
        chunk_grid: dict[str, JSON] | ChunkGrid,
        chunk_key_encoding: dict[str, JSON] | ChunkKeyEncoding,
        fill_value: Any,
        codecs: Iterable[Codec | dict[str, JSON]],
        attributes: None | dict[str, JSON],
        dimension_names: None | Iterable[str],
    ) -> None:
        """
        Because the class is a frozen dataclass, we set attributes using object.__setattr__
        """
        shape_parsed = parse_shapelike(shape)
        data_type_parsed = parse_dtype(data_type)
        chunk_grid_parsed = ChunkGrid.from_dict(chunk_grid)
        chunk_key_encoding_parsed = ChunkKeyEncoding.from_dict(chunk_key_encoding)
        dimension_names_parsed = parse_dimension_names(dimension_names)
        fill_value_parsed = parse_fill_value(fill_value)
        attributes_parsed = parse_attributes(attributes)
        codecs_parsed_partial = parse_codecs(codecs)

        array_spec = ArraySpec(
            shape=shape_parsed,
            dtype=data_type_parsed,
            fill_value=fill_value_parsed,
            order="C",  # TODO: order is not needed here.
            prototype=default_buffer_prototype,  # TODO: prototype is not needed here.
        )
        codecs_parsed = [c.evolve_from_array_spec(array_spec) for c in codecs_parsed_partial]

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "data_type", data_type_parsed)
        object.__setattr__(self, "chunk_grid", chunk_grid_parsed)
        object.__setattr__(self, "chunk_key_encoding", chunk_key_encoding_parsed)
        object.__setattr__(self, "codecs", codecs_parsed)
        object.__setattr__(self, "dimension_names", dimension_names_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "attributes", attributes_parsed)

        self._validate_metadata()

    def _validate_metadata(self) -> None:
        if isinstance(self.chunk_grid, RegularChunkGrid) and len(self.shape) != len(
            self.chunk_grid.chunk_shape
        ):
            raise ValueError(
                "`chunk_shape` and `shape` need to have the same number of dimensions."
            )
        if self.dimension_names is not None and len(self.shape) != len(self.dimension_names):
            raise ValueError(
                "`dimension_names` and `shape` need to have the same number of dimensions."
            )
        if self.fill_value is None:
            raise ValueError("`fill_value` is required.")
        for codec in self.codecs:
            codec.validate(shape=self.shape, dtype=self.data_type, chunk_grid=self.chunk_grid)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data_type

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_chunk_spec(
        self, _chunk_coords: ChunkCoords, order: Literal["C", "F"], prototype: BufferPrototype
    ) -> ArraySpec:
        assert isinstance(
            self.chunk_grid, RegularChunkGrid
        ), "Currently, only regular chunk grid is supported"
        return ArraySpec(
            shape=self.chunk_grid.chunk_shape,
            dtype=self.dtype,
            fill_value=self.fill_value,
            order=order,
            prototype=prototype,
        )

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.chunk_key_encoding.encode_chunk_key(chunk_coords)

    def to_buffer_dict(self) -> dict[str, Buffer]:
        def _json_convert(o: np.dtype[Any] | Enum | Codec) -> str | dict[str, Any]:
            if isinstance(o, np.dtype):
                return str(o)
            if isinstance(o, Enum):
                return o.name
            # this serializes numcodecs compressors
            # todo: implement to_dict for codecs
            elif isinstance(o, numcodecs.abc.Codec):
                config: dict[str, Any] = o.get_config()
                return config
            raise TypeError

        json_indent = config.get("json_indent")
        return {
            ZARR_JSON: default_buffer_prototype.buffer.from_bytes(
                json.dumps(self.to_dict(), default=_json_convert, indent=json_indent).encode()
            )
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> ArrayV3Metadata:
        # TODO: Remove the type: ignores[] comments below and use a TypedDict to type `data`
        # check that the zarr_format attribute is correct
        _ = parse_zarr_format_v3(data.pop("zarr_format"))  # type: ignore[arg-type]
        # check that the node_type attribute is correct
        _ = parse_node_type_array(data.pop("node_type"))  # type: ignore[arg-type]

        data["dimension_names"] = data.pop("dimension_names", None)

        return cls(**data)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        out_dict = super().to_dict()

        if not isinstance(out_dict, dict):
            raise TypeError(f"Expected dict. Got {type(out_dict)}.")

        # if `dimension_names` is `None`, we do not include it in
        # the metadata document
        if out_dict["dimension_names"] is None:
            out_dict.pop("dimension_names")
        return out_dict

    def update_shape(self, shape: ChunkCoords) -> Self:
        return replace(self, shape=shape)

    def update_attributes(self, attributes: dict[str, JSON]) -> Self:
        return replace(self, attributes=attributes)


@dataclass(frozen=True, kw_only=True)
class ArrayV2Metadata(ArrayMetadata):
    shape: ChunkCoords
    chunk_grid: RegularChunkGrid
    data_type: np.dtype[Any]
    fill_value: None | int | float = 0
    order: Literal["C", "F"] = "C"
    filters: list[dict[str, JSON]] | None = None
    dimension_separator: Literal[".", "/"] = "."
    compressor: dict[str, JSON] | None = None
    attributes: dict[str, JSON] = field(default_factory=dict)
    zarr_format: Literal[2] = field(init=False, default=2)

    def __init__(
        self,
        *,
        shape: ChunkCoords,
        dtype: npt.DTypeLike,
        chunks: ChunkCoords,
        fill_value: Any,
        order: Literal["C", "F"],
        dimension_separator: Literal[".", "/"] = ".",
        compressor: dict[str, JSON] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        attributes: dict[str, JSON] | None = None,
    ):
        """
        Metadata for a Zarr version 2 array.
        """
        shape_parsed = parse_shapelike(shape)
        data_type_parsed = parse_dtype(dtype)
        chunks_parsed = parse_shapelike(chunks)
        compressor_parsed = parse_compressor(compressor)
        order_parsed = parse_indexing_order(order)
        dimension_separator_parsed = parse_separator(dimension_separator)
        filters_parsed = parse_filters(filters)
        fill_value_parsed = parse_fill_value(fill_value)
        attributes_parsed = parse_attributes(attributes)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "data_type", data_type_parsed)
        object.__setattr__(self, "chunk_grid", RegularChunkGrid(chunk_shape=chunks_parsed))
        object.__setattr__(self, "compressor", compressor_parsed)
        object.__setattr__(self, "order", order_parsed)
        object.__setattr__(self, "dimension_separator", dimension_separator_parsed)
        object.__setattr__(self, "filters", filters_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "attributes", attributes_parsed)

        # ensure that the metadata document is consistent
        _ = parse_v2_metadata(self)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data_type

    @property
    def chunks(self) -> ChunkCoords:
        return self.chunk_grid.chunk_shape

    def to_buffer_dict(self) -> dict[str, Buffer]:
        def _json_convert(
            o: np.dtype[Any],
        ) -> str | list[tuple[str, str] | tuple[str, str, tuple[int, ...]]]:
            if isinstance(o, np.dtype):
                if o.fields is None:
                    return o.str
                else:
                    return o.descr
            raise TypeError

        zarray_dict = self.to_dict()
        assert isinstance(zarray_dict, dict)
        zattrs_dict = zarray_dict.pop("attributes", {})
        assert isinstance(zattrs_dict, dict)
        json_indent = config.get("json_indent")
        return {
            ZARRAY_JSON: default_buffer_prototype.buffer.from_bytes(
                json.dumps(zarray_dict, default=_json_convert, indent=json_indent).encode()
            ),
            ZATTRS_JSON: default_buffer_prototype.buffer.from_bytes(
                json.dumps(zattrs_dict, indent=json_indent).encode()
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArrayV2Metadata:
        # check that the zarr_format attribute is correct
        _ = parse_zarr_format_v2(data.pop("zarr_format"))
        return cls(**data)

    def to_dict(self) -> JSON:
        zarray_dict = super().to_dict()

        assert isinstance(zarray_dict, dict)

        _ = zarray_dict.pop("chunk_grid")
        zarray_dict["chunks"] = self.chunk_grid.chunk_shape

        _ = zarray_dict.pop("data_type")
        zarray_dict["dtype"] = self.data_type.str

        return zarray_dict

    def get_chunk_spec(
        self, _chunk_coords: ChunkCoords, order: Literal["C", "F"], prototype: BufferPrototype
    ) -> ArraySpec:
        return ArraySpec(
            shape=self.chunk_grid.chunk_shape,
            dtype=self.dtype,
            fill_value=self.fill_value,
            order=order,
            prototype=prototype,
        )

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.dimension_separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier

    def update_shape(self, shape: ChunkCoords) -> Self:
        return replace(self, shape=shape)

    def update_attributes(self, attributes: dict[str, JSON]) -> Self:
        return replace(self, attributes=attributes)


def parse_dimension_names(data: None | Iterable[str]) -> tuple[str, ...] | None:
    if data is None:
        return data
    elif all(isinstance(x, str) for x in data):
        return tuple(data)
    else:
        msg = f"Expected either None or a iterable of str, got {type(data)}"
        raise TypeError(msg)


# todo: real validation
def parse_attributes(data: None | dict[str, JSON]) -> dict[str, JSON]:
    if data is None:
        return {}

    return data


# todo: move to its own module and drop _v3 suffix
# todo: consider folding all the literal parsing into a single function
# that takes 2 arguments
def parse_zarr_format_v3(data: Literal[3]) -> Literal[3]:
    if data == 3:
        return data
    raise ValueError(f"Invalid value. Expected 3. Got {data}.")


# todo: move to its own module and drop _v2 suffix
def parse_zarr_format_v2(data: Literal[2]) -> Literal[2]:
    if data == 2:
        return data
    raise ValueError(f"Invalid value. Expected 2. Got {data}.")


def parse_node_type_array(data: Literal["array"]) -> Literal["array"]:
    if data == "array":
        return data
    raise ValueError(f"Invalid value. Expected 'array'. Got {data}.")


# todo: real validation
def parse_filters(data: list[dict[str, JSON]] | None) -> list[dict[str, JSON]] | None:
    return data


# todo: real validation
def parse_compressor(data: dict[str, JSON] | None) -> dict[str, JSON] | None:
    return data


def parse_v2_metadata(data: ArrayV2Metadata) -> ArrayV2Metadata:
    if (l_chunks := len(data.chunks)) != (l_shape := len(data.shape)):
        msg = (
            f"The `shape` and `chunks` attributes must have the same length. "
            f"`chunks` has length {l_chunks}, but `shape` has length {l_shape}."
        )
        raise ValueError(msg)
    return data


def create_pipeline(data: Iterable[Codec | JSON]) -> CodecPipeline:
    from zarr.codecs import BatchedCodecPipeline

    if not isinstance(data, Iterable):
        raise TypeError(f"Expected iterable, got {type(data)}")
    return BatchedCodecPipeline.from_dict(data)


def parse_codecs(data: Iterable[Codec | dict[str, JSON]]) -> tuple[Codec, ...]:
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
