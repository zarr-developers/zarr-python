from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, TypedDict, overload

from zarr.abc.metadata import Metadata
from zarr.core.buffer.core import default_buffer_prototype

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.common import JSON, ChunkCoords

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal, cast

import numcodecs.abc
import numpy as np
import numpy.typing as npt

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec, Codec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.chunk_grids import ChunkGrid, RegularChunkGrid
from zarr.core.chunk_key_encodings import ChunkKeyEncoding, ChunkKeyEncodingLike
from zarr.core.common import (
    JSON,
    ZARR_JSON,
    ChunkCoords,
    parse_named_configuration,
    parse_shapelike,
)
from zarr.core.config import config
from zarr.core.metadata.common import parse_attributes
from zarr.core.strings import _NUMPY_SUPPORTS_VLEN_STRING
from zarr.core.strings import _STRING_DTYPE as STRING_NP_DTYPE
from zarr.errors import MetadataValidationError, NodeTypeValidationError
from zarr.registry import get_codec_class

DEFAULT_DTYPE = "float64"

# Keep in sync with _replace_special_floats
SPECIAL_FLOATS_ENCODED = {
    "Infinity": np.inf,
    "-Infinity": -np.inf,
    "NaN": np.nan,
}


def parse_zarr_format(data: object) -> Literal[3]:
    if data == 3:
        return 3
    raise MetadataValidationError("zarr_format", 3, data)


def parse_node_type_array(data: object) -> Literal["array"]:
    if data == "array":
        return "array"
    raise NodeTypeValidationError("node_type", "array", data)


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


def validate_array_bytes_codec(codecs: tuple[Codec, ...]) -> ArrayBytesCodec:
    # ensure that we have at least one ArrayBytesCodec
    abcs: list[ArrayBytesCodec] = [codec for codec in codecs if isinstance(codec, ArrayBytesCodec)]
    if len(abcs) == 0:
        raise ValueError("At least one ArrayBytesCodec is required.")
    elif len(abcs) > 1:
        raise ValueError("Only one ArrayBytesCodec is allowed.")

    return abcs[0]


def validate_codecs(codecs: tuple[Codec, ...], dtype: DataType) -> None:
    """Check that the codecs are valid for the given dtype"""
    from zarr.codecs.sharding import ShardingCodec

    abc = validate_array_bytes_codec(codecs)

    # Recursively resolve array-bytes codecs within sharding codecs
    while isinstance(abc, ShardingCodec):
        abc = validate_array_bytes_codec(abc.codecs)

    # we need to have special codecs if we are decoding vlen strings or bytestrings
    # TODO: use codec ID instead of class name
    codec_class_name = abc.__class__.__name__
    if dtype == DataType.string and not codec_class_name == "VLenUTF8Codec":
        raise ValueError(
            f"For string dtype, ArrayBytesCodec must be `VLenUTF8Codec`, got `{codec_class_name}`."
        )
    if dtype == DataType.bytes and not codec_class_name == "VLenBytesCodec":
        raise ValueError(
            f"For bytes dtype, ArrayBytesCodec must be `VLenBytesCodec`, got `{codec_class_name}`."
        )


def parse_dimension_names(data: object) -> tuple[str | None, ...] | None:
    if data is None:
        return data
    elif isinstance(data, Iterable) and all(isinstance(x, type(None) | str) for x in data):
        return tuple(data)
    else:
        msg = f"Expected either None or a iterable of str, got {type(data)}"
        raise TypeError(msg)


def parse_storage_transformers(data: object) -> tuple[dict[str, JSON], ...]:
    """
    Parse storage_transformers. Zarr python cannot use storage transformers
    at this time, so this function doesn't attempt to validate them.
    """
    if data is None:
        return ()
    if isinstance(data, Iterable):
        if len(tuple(data)) >= 1:
            return data  # type: ignore[return-value]
        else:
            return ()
    raise TypeError(
        f"Invalid storage_transformers. Expected an iterable of dicts. Got {type(data)} instead."
    )


class V3JsonEncoder(json.JSONEncoder):
    def __init__(
        self,
        *,
        skipkeys: bool = False,
        ensure_ascii: bool = True,
        check_circular: bool = True,
        allow_nan: bool = True,
        sort_keys: bool = False,
        indent: int | None = None,
        separators: tuple[str, str] | None = None,
        default: Callable[[object], object] | None = None,
    ) -> None:
        if indent is None:
            indent = config.get("json_indent")
        super().__init__(
            skipkeys=skipkeys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
            default=default,
        )

    def default(self, o: object) -> Any:
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
                    return _replace_special_floats([out.real, out.imag])
                elif np.isnan(out):
                    return "NaN"
                elif np.isinf(out):
                    return "Infinity" if out > 0 else "-Infinity"
            return out
        elif isinstance(o, Enum):
            return o.name
        # this serializes numcodecs compressors
        # todo: implement to_dict for codecs
        elif isinstance(o, numcodecs.abc.Codec):
            config: dict[str, Any] = o.get_config()
            return config
        else:
            return super().default(o)


def _replace_special_floats(obj: object) -> Any:
    """Helper function to replace NaN/Inf/-Inf values with special strings

    Note: this cannot be done in the V3JsonEncoder because Python's `json.dumps` optimistically
    converts NaN/Inf values to special types outside of the encoding step.
    """
    if isinstance(obj, float):
        if np.isnan(obj):
            return "NaN"
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
    elif isinstance(obj, dict):
        # Recursively replace in dictionaries
        return {k: _replace_special_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively replace in lists
        return [_replace_special_floats(item) for item in obj]
    return obj


class ArrayV3MetadataDict(TypedDict):
    """
    A typed dictionary model for zarr v3 metadata.
    """

    zarr_format: Literal[3]
    attributes: dict[str, JSON]


@dataclass(frozen=True, kw_only=True)
class ArrayV3Metadata(Metadata):
    shape: ChunkCoords
    data_type: DataType
    chunk_grid: ChunkGrid
    chunk_key_encoding: ChunkKeyEncoding
    fill_value: Any
    codecs: tuple[Codec, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    dimension_names: tuple[str, ...] | None = None
    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default="array", init=False)
    storage_transformers: tuple[dict[str, JSON], ...]

    def __init__(
        self,
        *,
        shape: Iterable[int],
        data_type: npt.DTypeLike | DataType,
        chunk_grid: dict[str, JSON] | ChunkGrid,
        chunk_key_encoding: ChunkKeyEncodingLike,
        fill_value: Any,
        codecs: Iterable[Codec | dict[str, JSON]],
        attributes: dict[str, JSON] | None,
        dimension_names: Iterable[str] | None,
        storage_transformers: Iterable[dict[str, JSON]] | None = None,
    ) -> None:
        """
        Because the class is a frozen dataclass, we set attributes using object.__setattr__
        """
        shape_parsed = parse_shapelike(shape)
        data_type_parsed = DataType.parse(data_type)
        chunk_grid_parsed = ChunkGrid.from_dict(chunk_grid)
        chunk_key_encoding_parsed = ChunkKeyEncoding.from_dict(chunk_key_encoding)
        dimension_names_parsed = parse_dimension_names(dimension_names)
        if fill_value is None:
            fill_value = default_fill_value(data_type_parsed)
        # we pass a string here rather than an enum to make mypy happy
        fill_value_parsed = parse_fill_value(
            fill_value, dtype=cast(ALL_DTYPES, data_type_parsed.value)
        )
        attributes_parsed = parse_attributes(attributes)
        codecs_parsed_partial = parse_codecs(codecs)
        storage_transformers_parsed = parse_storage_transformers(storage_transformers)

        array_spec = ArraySpec(
            shape=shape_parsed,
            dtype=data_type_parsed.to_numpy(),
            fill_value=fill_value_parsed,
            config=ArrayConfig.from_dict({}),  # TODO: config is not needed here.
            prototype=default_buffer_prototype(),  # TODO: prototype is not needed here.
        )
        codecs_parsed = tuple(c.evolve_from_array_spec(array_spec) for c in codecs_parsed_partial)
        validate_codecs(codecs_parsed_partial, data_type_parsed)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "data_type", data_type_parsed)
        object.__setattr__(self, "chunk_grid", chunk_grid_parsed)
        object.__setattr__(self, "chunk_key_encoding", chunk_key_encoding_parsed)
        object.__setattr__(self, "codecs", codecs_parsed)
        object.__setattr__(self, "dimension_names", dimension_names_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "attributes", attributes_parsed)
        object.__setattr__(self, "storage_transformers", storage_transformers_parsed)

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
            codec.validate(
                shape=self.shape, dtype=self.data_type.to_numpy(), chunk_grid=self.chunk_grid
            )

    @property
    def dtype(self) -> np.dtype[Any]:
        """Interpret Zarr dtype as NumPy dtype"""
        return self.data_type.to_numpy()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def chunks(self) -> ChunkCoords:
        if isinstance(self.chunk_grid, RegularChunkGrid):
            from zarr.codecs.sharding import ShardingCodec

            if len(self.codecs) == 1 and isinstance(self.codecs[0], ShardingCodec):
                sharding_codec = self.codecs[0]
                assert isinstance(sharding_codec, ShardingCodec)  # for mypy
                return sharding_codec.chunk_shape
            else:
                return self.chunk_grid.chunk_shape

        msg = (
            f"The `chunks` attribute is only defined for arrays using `RegularChunkGrid`."
            f"This array has a {self.chunk_grid} instead."
        )
        raise NotImplementedError(msg)

    @property
    def shards(self) -> ChunkCoords | None:
        if isinstance(self.chunk_grid, RegularChunkGrid):
            from zarr.codecs.sharding import ShardingCodec

            if len(self.codecs) == 1 and isinstance(self.codecs[0], ShardingCodec):
                return self.chunk_grid.chunk_shape
            else:
                return None

        msg = (
            f"The `shards` attribute is only defined for arrays using `RegularChunkGrid`."
            f"This array has a {self.chunk_grid} instead."
        )
        raise NotImplementedError(msg)

    @property
    def inner_codecs(self) -> tuple[Codec, ...]:
        if isinstance(self.chunk_grid, RegularChunkGrid):
            from zarr.codecs.sharding import ShardingCodec

            if len(self.codecs) == 1 and isinstance(self.codecs[0], ShardingCodec):
                return self.codecs[0].codecs
        return self.codecs

    def get_chunk_spec(
        self, _chunk_coords: ChunkCoords, array_config: ArrayConfig, prototype: BufferPrototype
    ) -> ArraySpec:
        assert isinstance(self.chunk_grid, RegularChunkGrid), (
            "Currently, only regular chunk grid is supported"
        )
        return ArraySpec(
            shape=self.chunk_grid.chunk_shape,
            dtype=self.dtype,
            fill_value=self.fill_value,
            config=array_config,
            prototype=prototype,
        )

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.chunk_key_encoding.encode_chunk_key(chunk_coords)

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        d = _replace_special_floats(self.to_dict())
        return {ZARR_JSON: prototype.buffer.from_bytes(json.dumps(d, cls=V3JsonEncoder).encode())}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        # make a copy because we are modifying the dict
        _data = data.copy()

        # check that the zarr_format attribute is correct
        _ = parse_zarr_format(_data.pop("zarr_format"))
        # check that the node_type attribute is correct
        _ = parse_node_type_array(_data.pop("node_type"))

        # check that the data_type attribute is valid
        data_type = DataType.parse(_data.pop("data_type"))

        # dimension_names key is optional, normalize missing to `None`
        _data["dimension_names"] = _data.pop("dimension_names", None)
        # attributes key is optional, normalize missing to `None`
        _data["attributes"] = _data.pop("attributes", None)
        return cls(**_data, data_type=data_type)  # type: ignore[arg-type]

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


# enum Literals can't be used in typing, so we have to restate all of the V3 dtypes as types
# https://github.com/python/typing/issues/781

BOOL_DTYPE = Literal["bool"]
BOOL = np.bool_
INTEGER_DTYPE = Literal["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
INTEGER = np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64
FLOAT_DTYPE = Literal["float16", "float32", "float64"]
FLOAT = np.float16 | np.float32 | np.float64
COMPLEX_DTYPE = Literal["complex64", "complex128"]
COMPLEX = np.complex64 | np.complex128
STRING_DTYPE = Literal["string"]
STRING = np.str_
BYTES_DTYPE = Literal["bytes"]
BYTES = np.bytes_

ALL_DTYPES = BOOL_DTYPE | INTEGER_DTYPE | FLOAT_DTYPE | COMPLEX_DTYPE | STRING_DTYPE | BYTES_DTYPE


@overload
def parse_fill_value(
    fill_value: complex | str | bytes | np.generic | Sequence[Any] | bool,
    dtype: BOOL_DTYPE,
) -> BOOL: ...


@overload
def parse_fill_value(
    fill_value: complex | str | bytes | np.generic | Sequence[Any] | bool,
    dtype: INTEGER_DTYPE,
) -> INTEGER: ...


@overload
def parse_fill_value(
    fill_value: complex | str | bytes | np.generic | Sequence[Any] | bool,
    dtype: FLOAT_DTYPE,
) -> FLOAT: ...


@overload
def parse_fill_value(
    fill_value: complex | str | bytes | np.generic | Sequence[Any] | bool,
    dtype: COMPLEX_DTYPE,
) -> COMPLEX: ...


@overload
def parse_fill_value(
    fill_value: complex | str | bytes | np.generic | Sequence[Any] | bool,
    dtype: STRING_DTYPE,
) -> STRING: ...


@overload
def parse_fill_value(
    fill_value: complex | str | bytes | np.generic | Sequence[Any] | bool,
    dtype: BYTES_DTYPE,
) -> BYTES: ...


def parse_fill_value(
    fill_value: Any,
    dtype: ALL_DTYPES,
) -> Any:
    """
    Parse `fill_value`, a potential fill value, into an instance of `dtype`, a data type.
    If `fill_value` is `None`, then this function will return the result of casting the value 0
    to the provided data type. Otherwise, `fill_value` will be cast to the provided data type.

    Note that some numpy dtypes use very permissive casting rules. For example,
    `np.bool_({'not remotely a bool'})` returns `True`. Thus this function should not be used for
    validating that the provided fill value is a valid instance of the data type.

    Parameters
    ----------
    fill_value : Any
        A potential fill value.
    dtype : str
        A valid Zarr format 3 DataType.

    Returns
    -------
    A scalar instance of `dtype`
    """
    data_type = DataType(dtype)
    if fill_value is None:
        raise ValueError("Fill value cannot be None")
    if data_type == DataType.string:
        return np.str_(fill_value)
    if data_type == DataType.bytes:
        return np.bytes_(fill_value)

    # the rest are numeric types
    np_dtype = cast(np.dtype[Any], data_type.to_numpy())

    if isinstance(fill_value, Sequence) and not isinstance(fill_value, str):
        if data_type in (DataType.complex64, DataType.complex128):
            if len(fill_value) == 2:
                decoded_fill_value = tuple(
                    SPECIAL_FLOATS_ENCODED.get(value, value) for value in fill_value
                )
                # complex datatypes serialize to JSON arrays with two elements
                return np_dtype.type(complex(*decoded_fill_value))
            else:
                msg = (
                    f"Got an invalid fill value for complex data type {data_type.value}."
                    f"Expected a sequence with 2 elements, but {fill_value!r} has "
                    f"length {len(fill_value)}."
                )
                raise ValueError(msg)
        msg = f"Cannot parse non-string sequence {fill_value!r} as a scalar with type {data_type.value}."
        raise TypeError(msg)

    # Cast the fill_value to the given dtype
    try:
        # This warning filter can be removed after Zarr supports numpy>=2.0
        # The warning is saying that the future behavior of out of bounds casting will be to raise
        # an OverflowError. In the meantime, we allow overflow and catch cases where
        # fill_value != casted_value below.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            casted_value = np.dtype(np_dtype).type(fill_value)
    except (ValueError, OverflowError, TypeError) as e:
        raise ValueError(f"fill value {fill_value!r} is not valid for dtype {data_type}") from e
    # Check if the value is still representable by the dtype
    if (fill_value == "NaN" and np.isnan(casted_value)) or (
        fill_value in ["Infinity", "-Infinity"] and not np.isfinite(casted_value)
    ):
        pass
    elif np_dtype.kind == "f":
        # float comparison is not exact, especially when dtype <float64
        # so we use np.isclose for this comparison.
        # this also allows us to compare nan fill_values
        if not np.isclose(fill_value, casted_value, equal_nan=True):
            raise ValueError(f"fill value {fill_value!r} is not valid for dtype {data_type}")
    elif np_dtype.kind == "c":
        # confusingly np.isclose(np.inf, np.inf + 0j) is False on numpy<2, so compare real and imag parts
        # explicitly.
        if not (
            np.isclose(np.real(fill_value), np.real(casted_value), equal_nan=True)
            and np.isclose(np.imag(fill_value), np.imag(casted_value), equal_nan=True)
        ):
            raise ValueError(f"fill value {fill_value!r} is not valid for dtype {data_type}")
    else:
        if fill_value != casted_value:
            raise ValueError(f"fill value {fill_value!r} is not valid for dtype {data_type}")

    return casted_value


def default_fill_value(dtype: DataType) -> str | bytes | np.generic:
    if dtype == DataType.string:
        return ""
    elif dtype == DataType.bytes:
        return b""
    else:
        np_dtype = dtype.to_numpy()
        np_dtype = cast(np.dtype[Any], np_dtype)
        return np_dtype.type(0)  # type: ignore[misc]


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
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    complex64 = "complex64"
    complex128 = "complex128"
    string = "string"
    bytes = "bytes"

    @property
    def byte_count(self) -> int | None:
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
            DataType.float16: 2,
            DataType.float32: 4,
            DataType.float64: 8,
            DataType.complex64: 8,
            DataType.complex128: 16,
        }
        try:
            return data_type_byte_counts[self]
        except KeyError:
            # string and bytes have variable length
            return None

    @property
    def has_endianness(self) -> _bool:
        return self.byte_count is not None and self.byte_count != 1

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
            DataType.float16: "f2",
            DataType.float32: "f4",
            DataType.float64: "f8",
            DataType.complex64: "c8",
            DataType.complex128: "c16",
        }
        return data_type_to_numpy[self]

    def to_numpy(self) -> np.dtypes.StringDType | np.dtypes.ObjectDType | np.dtype[Any]:
        # note: it is not possible to round trip DataType <-> np.dtype
        # due to the fact that DataType.string and DataType.bytes both
        # generally return np.dtype("O") from this function, even though
        # they can originate as fixed-length types (e.g. "<U10", "|S5")
        if self == DataType.string:
            return STRING_NP_DTYPE
        elif self == DataType.bytes:
            # for now always use object dtype for bytestrings
            # TODO: consider whether we can use fixed-width types (e.g. '|S5') instead
            return np.dtype("O")
        else:
            return np.dtype(self.to_numpy_shortname())

    @classmethod
    def from_numpy(cls, dtype: np.dtype[Any]) -> DataType:
        if dtype.kind in "UT":
            return DataType.string
        elif dtype.kind == "S":
            return DataType.bytes
        elif not _NUMPY_SUPPORTS_VLEN_STRING and dtype.kind == "O":
            # numpy < 2.0 does not support vlen string dtype
            # so we fall back on object array of strings
            return DataType.string
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
            "<f2": "float16",
            "<f4": "float32",
            "<f8": "float64",
            "<c8": "complex64",
            "<c16": "complex128",
        }
        return DataType[dtype_to_data_type[dtype.str]]

    @classmethod
    def parse(cls, dtype: DataType | Any | None) -> DataType:
        if dtype is None:
            return DataType[DEFAULT_DTYPE]
        if isinstance(dtype, DataType):
            return dtype
        try:
            return DataType(dtype)
        except ValueError:
            pass
        try:
            dtype = np.dtype(dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid Zarr format 3 data_type: {dtype}") from e
        # check that this is a valid v3 data_type
        try:
            data_type = DataType.from_numpy(dtype)
        except KeyError as e:
            raise ValueError(f"Invalid Zarr format 3 data_type: {dtype}") from e
        return data_type
