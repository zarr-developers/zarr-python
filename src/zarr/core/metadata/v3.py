from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import Self

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.common import JSON, ChunkCoords

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal

import numcodecs.abc
import numpy as np

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec, Codec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.chunk_grids import ChunkGrid, RegularChunkGrid
from zarr.core.chunk_key_encodings import ChunkKeyEncoding
from zarr.core.common import ZARR_JSON, parse_dtype, parse_named_configuration, parse_shapelike
from zarr.core.config import config
from zarr.core.metadata.common import ArrayMetadata, parse_attributes
from zarr.registry import get_codec_class


def parse_zarr_format(data: object) -> Literal[3]:
    if data == 3:
        return 3
    raise ValueError(f"Invalid value. Expected 3. Got {data}.")


def parse_node_type_array(data: object) -> Literal["array"]:
    if data == "array":
        return "array"
    raise ValueError(f"Invalid value. Expected 'array'. Got {data}.")


def parse_codecs(data: object) -> tuple[Codec, ...]:
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


def parse_dimension_names(data: object) -> tuple[str | None, ...] | None:
    if data is None:
        return data
    elif isinstance(data, Iterable) and all(isinstance(x, type(None) | str) for x in data):
        return tuple(data)
    else:
        msg = f"Expected either None or a iterable of str, got {type(data)}"
        raise TypeError(msg)


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
        fill_value_parsed = parse_fill_value(fill_value, dtype=data_type_parsed)
        attributes_parsed = parse_attributes(attributes)
        codecs_parsed_partial = parse_codecs(codecs)

        array_spec = ArraySpec(
            shape=shape_parsed,
            dtype=data_type_parsed,
            fill_value=fill_value_parsed,
            order="C",  # TODO: order is not needed here.
            prototype=default_buffer_prototype(),  # TODO: prototype is not needed here.
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

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        def _json_convert(o: object) -> Any:
            if isinstance(o, np.dtype):
                return str(o)
            if np.isscalar(o):
                out: Any
                if hasattr(o, "dtype") and o.dtype.kind == "M" and hasattr(o, "view"):
                    # https://github.com/zarr-developers/zarr-python/issues/2119
                    # `.item()` on a datetime type might or might not return an
                    # integer, depending on the value.
                    # Explicitly cast to an int first, and then grab .item()
                    out = o.view("i8").item()
                else:
                    # convert numpy scalar to python type, and pass
                    # python types through
                    out = getattr(o, "item", lambda: o)()
                    if isinstance(out, complex):
                        # python complex types are not JSON serializable, so we use the
                        # serialization defined in the zarr v3 spec
                        return [out.real, out.imag]
                return out
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
            ZARR_JSON: prototype.buffer.from_bytes(
                json.dumps(self.to_dict(), default=_json_convert, indent=json_indent).encode()
            )
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        # make a copy because we are modifying the dict
        _data = data.copy()

        # check that the zarr_format attribute is correct
        _ = parse_zarr_format(_data.pop("zarr_format"))
        # check that the node_type attribute is correct
        _ = parse_node_type_array(_data.pop("node_type"))

        # dimension_names key is optional, normalize missing to `None`
        _data["dimension_names"] = _data.pop("dimension_names", None)
        # attributes key is optional, normalize missing to `None`
        _data["attributes"] = _data.pop("attributes", None)
        return cls(**_data)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
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


BOOL = np.bool_
BOOL_DTYPE = np.dtypes.BoolDType
INTEGER_DTYPE = (
    np.dtypes.Int8DType
    | np.dtypes.Int16DType
    | np.dtypes.Int32DType
    | np.dtypes.Int64DType
    | np.dtypes.UInt8DType
    | np.dtypes.UInt16DType
    | np.dtypes.UInt32DType
    | np.dtypes.UInt64DType
)
INTEGER = np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64
FLOAT_DTYPE = np.dtypes.Float16DType | np.dtypes.Float32DType | np.dtypes.Float64DType
FLOAT = np.float16 | np.float32 | np.float64
COMPLEX_DTYPE = np.dtypes.Complex64DType | np.dtypes.Complex128DType
COMPLEX = np.complex64 | np.complex128


@overload
def parse_fill_value(fill_value: object, dtype: BOOL_DTYPE) -> BOOL: ...


@overload
def parse_fill_value(fill_value: object, dtype: INTEGER_DTYPE) -> INTEGER: ...


@overload
def parse_fill_value(fill_value: object, dtype: FLOAT_DTYPE) -> FLOAT: ...


@overload
def parse_fill_value(fill_value: object, dtype: COMPLEX_DTYPE) -> COMPLEX: ...


@overload
def parse_fill_value(fill_value: object, dtype: np.dtype[Any]) -> Any:
    # This dtype[Any] is unfortunately necessary right now.
    # See https://github.com/zarr-developers/zarr-python/issues/2131#issuecomment-2318010899
    # for more details, but `dtype` here (which comes from `parse_dtype`)
    # is np.dtype[Any].
    #
    # If you want the specialized types rather than Any, you need to use `np.dtypes.<dtype>`
    # rather than np.dtypes(<type>)
    ...


def parse_fill_value(
    fill_value: object,
    dtype: BOOL_DTYPE | INTEGER_DTYPE | FLOAT_DTYPE | COMPLEX_DTYPE | np.dtype[Any],
) -> BOOL | INTEGER | FLOAT | COMPLEX | Any:
    """
    Parse `fill_value`, a potential fill value, into an instance of `dtype`, a data type.
    If `fill_value` is `None`, then this function will return the result of casting the value 0
    to the provided data type. Otherwise, `fill_value` will be cast to the provided data type.

    Note that some numpy dtypes use very permissive casting rules. For example,
    `np.bool_({'not remotely a bool'})` returns `True`. Thus this function should not be used for
    validating that the provided fill value is a valid instance of the data type.

    Parameters
    ----------
    fill_value: Any
        A potential fill value.
    dtype: BOOL_DTYPE | INTEGER_DTYPE | FLOAT_DTYPE | COMPLEX_DTYPE
        A numpy data type that models a data type defined in the Zarr V3 specification.

    Returns
    -------
    A scalar instance of `dtype`
    """
    if fill_value is None:
        return dtype.type(0)
    if isinstance(fill_value, Sequence) and not isinstance(fill_value, str | bytes):
        if dtype in (np.complex64, np.complex128):
            dtype = cast(COMPLEX_DTYPE, dtype)
            if len(fill_value) == 2:
                # complex datatypes serialize to JSON arrays with two elements
                return dtype.type(complex(*fill_value))
            else:
                msg = (
                    f"Got an invalid fill value for complex data type {dtype}."
                    f"Expected a sequence with 2 elements, but {fill_value} has "
                    f"length {len(fill_value)}."
                )
                raise ValueError(msg)
        msg = f"Cannot parse non-string sequence {fill_value} as a scalar with type {dtype}."
        raise TypeError(msg)
    return dtype.type(fill_value)  # type: ignore[arg-type]


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
