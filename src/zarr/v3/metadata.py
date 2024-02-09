from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
import json
import numpy as np

from zarr.v3.chunk_grids import ChunkGrid, RegularChunkGrid
from zarr.v3.chunk_key_encodings import ChunkKeyEncoding


if TYPE_CHECKING:
    from typing import Any, Dict, Literal, Union, Iterable, List, Optional, Tuple
    from typing_extensions import Self
    from zarr.v3.codecs.pipeline import CodecPipeline


from zarr.v3.abc.codec import Codec
from zarr.v3.abc.metadata import Metadata

from zarr.v3.common import (
    ArraySpec,
    ChunkCoords,
    NamedConfig,
    RuntimeConfiguration,
    parse_dtype,
    parse_fill_value,
    parse_shapelike,
)


def runtime_configuration(
    order: Literal["C", "F"], concurrency: Optional[int] = None
) -> RuntimeConfiguration:
    return RuntimeConfiguration(order=order, concurrency=concurrency)


# For type checking
_bool = bool


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
    def from_dtype(cls, dtype: np.dtype) -> Self:
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


@dataclass(frozen=True)
class ArrayMetadata(Metadata):
    shape: ChunkCoords
    data_type: np.dtype
    chunk_grid: ChunkGrid
    chunk_key_encoding: ChunkKeyEncoding
    fill_value: Any
    codecs: CodecPipeline
    attributes: Dict[str, Any] = field(default_factory=dict)
    dimension_names: Optional[Tuple[str, ...]] = None
    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default="array", init=False)

    def __init__(
        self,
        *,
        shape,
        data_type,
        chunk_grid,
        chunk_key_encoding,
        fill_value,
        codecs,
        attributes,
        dimension_names,
    ):
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

        array_spec = ArraySpec(
            shape=shape_parsed, dtype=data_type_parsed, fill_value=fill_value_parsed
        )
        codecs_parsed = parse_codecs(codecs).evolve(array_spec)

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
        if len(self.shape) != len(self.chunk_grid.chunk_shape):
            raise ValueError(
                "`chunk_shape` and `shape` need to have the same number of dimensions."
            )
        if self.dimension_names is not None and len(self.shape) != len(self.dimension_names):
            raise ValueError(
                "`dimension_names` and `shape` need to have the same number of dimensions."
            )
        if self.fill_value is None:
            raise ValueError("`fill_value` is required.")
        self.codecs.validate(self)

    @property
    def dtype(self) -> np.dtype:
        return self.data_type

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_chunk_spec(self, _chunk_coords: ChunkCoords) -> ArraySpec:
        assert isinstance(
            self.chunk_grid, RegularChunkGrid
        ), "Currently, only regular chunk grid is supported"
        return ArraySpec(
            shape=self.chunk_grid.chunk_shape,
            dtype=self.dtype,
            fill_value=self.fill_value,
        )

    def to_bytes(self) -> bytes:
        def _json_convert(o):
            if isinstance(o, np.dtype):
                return str(o)
            if isinstance(o, Enum):
                return o.name
            # this serializes numcodecs compressors
            # todo: implement to_dict for codecs
            elif hasattr(o, "get_config"):
                return o.get_config()
            raise TypeError

        return json.dumps(
            self.to_dict(),
            default=_json_convert,
        ).encode()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArrayMetadata:
        # check that the zarr_format attribute is correct
        _ = parse_zarr_format_v3(data.pop("zarr_format"))
        # check that the node_type attribute is correct
        _ = parse_node_type_array(data.pop("node_type"))

        dimension_names = data.pop("dimension_names", None)

        return cls(**data, dimension_names=dimension_names)

    def to_dict(self) -> Dict[str, Any]:
        out_dict = super().to_dict()

        # if `dimension_names` is `None`, we do not include it in
        # the metadata document
        if out_dict["dimension_names"] is None:
            out_dict.pop("dimension_names")
        return out_dict


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
    attributes: Dict[str, Any] = field(default_factory=dict)
    zarr_format: Literal[2] = field(init=False, default=2)

    def __init__(self, *, shape, dtype, chunks, fill_value, compressor, filters, attributes):
        """
        Because the class is a frozen dataclass, we set attributes using object.__setattr__
        """
        shape_parsed = parse_shapelike(shape)
        data_type_parsed = parse_dtype(dtype)
        chunks_parsed = parse_shapelike(chunks)
        compressor_parsed = parse_compressor(compressor)
        filters_parsed = parse_filters(filters)
        fill_value_parsed = parse_fill_value(fill_value)
        attributes_parsed = parse_attributes(attributes)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "data_type", data_type_parsed)
        object.__setattr__(self, "chunks", chunks_parsed)
        object.__setattr__(self, "compressor", compressor_parsed)
        object.__setattr__(self, "filters", filters_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "attributes", attributes_parsed)

        # ensure that the metadata document is consistent
        _ = parse_v2_metadata(self)

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

        return json.dumps(self.to_dict(), default=_json_convert).encode()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArrayV2Metadata:
        # check that the zarr_format attribute is correct
        _ = parse_zarr_format_v2(data.pop("zarr_format"))
        return cls(**data)


def parse_dimension_names(data: Any) -> Tuple[str, ...] | None:
    if data is None:
        return data
    return tuple(map(str, data))


# todo: real validation
def parse_attributes(data: Any) -> Any:
    return data


# todo: move to its own module and drop _v3 suffix
def parse_zarr_format_v3(data: Any) -> Literal[3]:
    if data == 3:
        return data
    raise ValueError(f"Invalid value for `zarr_format`. Expected 3. Got {data}.")


# todo: move to its own module and drop _v2 suffix
def parse_zarr_format_v2(data: Any) -> Literal[2]:
    if data == 2:
        return data
    raise ValueError(f"Invalid value for `zarr_format`. Expected 3. Got {data}.")


def parse_node_type_array(data: Any) -> Literal["array"]:
    if data == "array":
        return data
    raise ValueError(f"Invalid value for `node_type`. Expected 'array'. Got {data}.")


# todo: real validation
def parse_filters(data: Any) -> List[Codec]:
    return data


# todo: real validation
def parse_compressor(data: Any) -> Codec:
    return data


def parse_v2_metadata(data: ArrayV2Metadata) -> ArrayV2Metadata:
    if (l_chunks := len(data.chunks)) != (l_shape := len(data.shape)):
        msg = (
            f"The `shape` and `chunks` attributes must have the same length. "
            f"`chunks` has length {l_chunks}, but `shape` has length {l_shape}."
        )
        raise ValueError(msg)
    return data


def parse_codecs(data: Iterable[NamedConfig]) -> CodecPipeline:
    from zarr.v3.codecs.pipeline import CodecPipeline

    return CodecPipeline.from_dict(data)
