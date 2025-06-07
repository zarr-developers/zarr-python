from __future__ import annotations

import base64
import warnings
from collections.abc import Iterable, Sequence
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict, cast

import numcodecs.abc

from zarr.abc.metadata import Metadata

if TYPE_CHECKING:
    from typing import Literal, Self

    import numpy.typing as npt

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import ChunkCoords

import json
import numbers
from dataclasses import dataclass, field, fields, replace

import numcodecs
import numpy as np

from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.chunk_key_encodings import parse_separator
from zarr.core.common import JSON, ZARRAY_JSON, ZATTRS_JSON, MemoryOrder, parse_shapelike
from zarr.core.config import config, parse_indexing_order
from zarr.core.metadata.common import parse_attributes


class ArrayV2MetadataDict(TypedDict):
    """
    A typed dictionary model for Zarr format 2 metadata.
    """

    zarr_format: Literal[2]
    attributes: dict[str, JSON]


# Union of acceptable types for v2 compressors
CompressorLikev2: TypeAlias = dict[str, JSON] | numcodecs.abc.Codec | None


@dataclass(frozen=True, kw_only=True)
class ArrayV2Metadata(Metadata):
    shape: ChunkCoords
    chunks: ChunkCoords
    dtype: np.dtype[Any]
    fill_value: int | float | str | bytes | None = 0
    order: MemoryOrder = "C"
    filters: tuple[numcodecs.abc.Codec, ...] | None = None
    dimension_separator: Literal[".", "/"] = "."
    compressor: CompressorLikev2
    attributes: dict[str, JSON] = field(default_factory=dict)
    zarr_format: Literal[2] = field(init=False, default=2)

    def __init__(
        self,
        *,
        shape: ChunkCoords,
        dtype: npt.DTypeLike,
        chunks: ChunkCoords,
        fill_value: Any,
        order: MemoryOrder,
        dimension_separator: Literal[".", "/"] = ".",
        compressor: CompressorLikev2 = None,
        filters: Iterable[numcodecs.abc.Codec | dict[str, JSON]] | None = None,
        attributes: dict[str, JSON] | None = None,
    ) -> None:
        """
        Metadata for a Zarr format 2 array.
        """
        shape_parsed = parse_shapelike(shape)
        dtype_parsed = parse_dtype(dtype)
        chunks_parsed = parse_shapelike(chunks)

        compressor_parsed = parse_compressor(compressor)
        order_parsed = parse_indexing_order(order)
        dimension_separator_parsed = parse_separator(dimension_separator)
        filters_parsed = parse_filters(filters)
        fill_value_parsed = parse_fill_value(fill_value, dtype=dtype_parsed)
        attributes_parsed = parse_attributes(attributes)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "dtype", dtype_parsed)
        object.__setattr__(self, "chunks", chunks_parsed)
        object.__setattr__(self, "compressor", compressor_parsed)
        object.__setattr__(self, "order", order_parsed)
        object.__setattr__(self, "dimension_separator", dimension_separator_parsed)
        object.__setattr__(self, "filters", filters_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "attributes", attributes_parsed)

        # ensure that the metadata document is consistent
        _ = parse_metadata(self)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @cached_property
    def chunk_grid(self) -> RegularChunkGrid:
        return RegularChunkGrid(chunk_shape=self.chunks)

    @property
    def shards(self) -> ChunkCoords | None:
        return None

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        def _json_convert(
            o: Any,
        ) -> Any:
            if isinstance(o, np.dtype):
                if o.fields is None:
                    return o.str
                else:
                    return o.descr
            if isinstance(o, numcodecs.abc.Codec):
                codec_config = o.get_config()

                # Hotfix for https://github.com/zarr-developers/zarr-python/issues/2647
                if codec_config["id"] == "zstd" and not codec_config.get("checksum", False):
                    codec_config.pop("checksum", None)

                return codec_config
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
            raise TypeError

        zarray_dict = self.to_dict()
        zarray_dict["fill_value"] = _serialize_fill_value(self.fill_value, self.dtype)
        zattrs_dict = zarray_dict.pop("attributes", {})
        json_indent = config.get("json_indent")
        return {
            ZARRAY_JSON: prototype.buffer.from_bytes(
                json.dumps(
                    zarray_dict, default=_json_convert, indent=json_indent, allow_nan=False
                ).encode()
            ),
            ZATTRS_JSON: prototype.buffer.from_bytes(
                json.dumps(zattrs_dict, indent=json_indent, allow_nan=False).encode()
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArrayV2Metadata:
        # Make a copy to protect the original from modification.
        _data = data.copy()
        # Check that the zarr_format attribute is correct.
        _ = parse_zarr_format(_data.pop("zarr_format"))

        # zarr v2 allowed arbitrary keys in the metadata.
        # Filter the keys to only those expected by the constructor.
        expected = {x.name for x in fields(cls)}
        expected |= {"dtype", "chunks"}

        # check if `filters` is an empty sequence; if so use None instead and raise a warning
        filters = _data.get("filters")
        if (
            isinstance(filters, Sequence)
            and not isinstance(filters, (str, bytes))
            and len(filters) == 0
        ):
            msg = (
                "Found an empty list of filters in the array metadata document. "
                "This is contrary to the Zarr V2 specification, and will cause an error in the future. "
                "Use None (or Null in a JSON document) instead of an empty list of filters."
            )
            warnings.warn(msg, UserWarning, stacklevel=1)
            _data["filters"] = None

        _data = {k: v for k, v in _data.items() if k in expected}

        return cls(**_data)

    def to_dict(self) -> dict[str, JSON]:
        zarray_dict = super().to_dict()

        _ = zarray_dict.pop("dtype")
        dtype_json: JSON
        # In the case of zarr v2, the simplest i.e., '|VXX' dtype is represented as a string
        dtype_descr = self.dtype.descr
        if self.dtype.kind == "V" and dtype_descr[0][0] != "" and len(dtype_descr) != 0:
            dtype_json = tuple(self.dtype.descr)
        else:
            dtype_json = self.dtype.str
        zarray_dict["dtype"] = dtype_json

        return zarray_dict

    def get_chunk_spec(
        self, _chunk_coords: ChunkCoords, array_config: ArrayConfig, prototype: BufferPrototype
    ) -> ArraySpec:
        return ArraySpec(
            shape=self.chunks,
            dtype=self.dtype,
            fill_value=self.fill_value,
            config=array_config,
            prototype=prototype,
        )

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.dimension_separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier

    def update_shape(self, shape: ChunkCoords) -> Self:
        return replace(self, shape=shape)

    def update_attributes(self, attributes: dict[str, JSON]) -> Self:
        return replace(self, attributes=attributes)


def parse_dtype(data: npt.DTypeLike) -> np.dtype[Any]:
    if isinstance(data, list):  # this is a valid _VoidDTypeLike check
        data = [tuple(d) for d in data]
    return np.dtype(data)


def parse_zarr_format(data: object) -> Literal[2]:
    if data == 2:
        return 2
    raise ValueError(f"Invalid value. Expected 2. Got {data}.")


def parse_filters(data: object) -> tuple[numcodecs.abc.Codec, ...] | None:
    """
    Parse a potential tuple of filters
    """
    out: list[numcodecs.abc.Codec] = []

    if data is None:
        return data
    if isinstance(data, Iterable):
        for idx, val in enumerate(data):
            if isinstance(val, numcodecs.abc.Codec):
                out.append(val)
            elif isinstance(val, dict):
                out.append(numcodecs.get_codec(val))
            else:
                msg = f"Invalid filter at index {idx}. Expected a numcodecs.abc.Codec or a dict representation of numcodecs.abc.Codec. Got {type(val)} instead."
                raise TypeError(msg)
        if len(out) == 0:
            # Per the v2 spec, an empty tuple is not allowed -- use None to express "no filters"
            return None
        else:
            return tuple(out)
    # take a single codec instance and wrap it in a tuple
    if isinstance(data, numcodecs.abc.Codec):
        return (data,)
    msg = f"Invalid filters. Expected None, an iterable of numcodecs.abc.Codec or dict representations of numcodecs.abc.Codec. Got {type(data)} instead."
    raise TypeError(msg)


def parse_compressor(data: object) -> numcodecs.abc.Codec | None:
    """
    Parse a potential compressor.
    """
    if data is None or isinstance(data, numcodecs.abc.Codec):
        return data
    if isinstance(data, dict):
        return numcodecs.get_codec(data)
    msg = f"Invalid compressor. Expected None, a numcodecs.abc.Codec, or a dict representation of a numcodecs.abc.Codec. Got {type(data)} instead."
    raise ValueError(msg)


def parse_metadata(data: ArrayV2Metadata) -> ArrayV2Metadata:
    if (l_chunks := len(data.chunks)) != (l_shape := len(data.shape)):
        msg = (
            f"The `shape` and `chunks` attributes must have the same length. "
            f"`chunks` has length {l_chunks}, but `shape` has length {l_shape}."
        )
        raise ValueError(msg)
    return data


def _parse_structured_fill_value(fill_value: Any, dtype: np.dtype[Any]) -> Any:
    """Handle structured dtype/fill value pairs"""
    try:
        if isinstance(fill_value, list):
            return np.array([tuple(fill_value)], dtype=dtype)[0]
        elif isinstance(fill_value, tuple):
            return np.array([fill_value], dtype=dtype)[0]
        elif isinstance(fill_value, bytes):
            return np.frombuffer(fill_value, dtype=dtype)[0]
        elif isinstance(fill_value, str):
            decoded = base64.standard_b64decode(fill_value)
            return np.frombuffer(decoded, dtype=dtype)[0]
        else:
            return np.array(fill_value, dtype=dtype)[()]
    except Exception as e:
        raise ValueError(f"Fill_value {fill_value} is not valid for dtype {dtype}.") from e


def parse_fill_value(fill_value: Any, dtype: np.dtype[Any]) -> Any:
    """
    Parse a potential fill value into a value that is compatible with the provided dtype.

    Parameters
    ----------
    fill_value : Any
        A potential fill value.
    dtype : np.dtype[Any]
        A numpy dtype.

    Returns
    -------
        An instance of `dtype`, or `None`, or any python object (in the case of an object dtype)
    """

    if fill_value is None or dtype.hasobject:
        pass
    elif dtype.fields is not None:
        # the dtype is structured (has multiple fields), so the fill_value might be a
        # compound value (e.g., a tuple or dict) that needs field-wise processing.
        # We use parse_structured_fill_value to correctly convert each component.
        fill_value = _parse_structured_fill_value(fill_value, dtype)
    elif not isinstance(fill_value, np.void) and fill_value == 0:
        # this should be compatible across numpy versions for any array type, including
        # structured arrays
        fill_value = np.zeros((), dtype=dtype)[()]
    elif dtype.kind == "U":
        # special case unicode because of encoding issues on Windows if passed through numpy
        # https://github.com/alimanfoo/zarr/pull/172#issuecomment-343782713

        if not isinstance(fill_value, str):
            raise ValueError(
                f"fill_value {fill_value!r} is not valid for dtype {dtype}; must be a unicode string"
            )
    elif dtype.kind in "SV" and isinstance(fill_value, str):
        fill_value = base64.standard_b64decode(fill_value)
    elif dtype.kind == "c" and isinstance(fill_value, list) and len(fill_value) == 2:
        complex_val = complex(float(fill_value[0]), float(fill_value[1]))
        fill_value = np.array(complex_val, dtype=dtype)[()]
    else:
        try:
            if isinstance(fill_value, bytes) and dtype.kind == "V":
                # special case for numpy 1.14 compatibility
                fill_value = np.array(fill_value, dtype=dtype.str).view(dtype)[()]
            else:
                fill_value = np.array(fill_value, dtype=dtype)[()]

        except Exception as e:
            msg = f"Fill_value {fill_value} is not valid for dtype {dtype}."
            raise ValueError(msg) from e

    return fill_value


def _serialize_fill_value(fill_value: Any, dtype: np.dtype[Any]) -> JSON:
    serialized: JSON

    if fill_value is None:
        serialized = None
    elif dtype.kind in "SV":
        # There's a relationship between dtype and fill_value
        # that mypy isn't aware of. The fact that we have S or V dtype here
        # means we should have a bytes-type fill_value.
        serialized = base64.standard_b64encode(cast("bytes", fill_value)).decode("ascii")
    elif isinstance(fill_value, np.datetime64):
        serialized = np.datetime_as_string(fill_value)
    elif isinstance(fill_value, numbers.Integral):
        serialized = int(fill_value)
    elif isinstance(fill_value, numbers.Real):
        float_fv = float(fill_value)
        if np.isnan(float_fv):
            serialized = "NaN"
        elif np.isinf(float_fv):
            serialized = "Infinity" if float_fv > 0 else "-Infinity"
        else:
            serialized = float_fv
    elif isinstance(fill_value, numbers.Complex):
        serialized = [
            _serialize_fill_value(fill_value.real, dtype),
            _serialize_fill_value(fill_value.imag, dtype),
        ]
    else:
        serialized = fill_value

    return serialized


def _default_fill_value(dtype: np.dtype[Any]) -> Any:
    """
    Get the default fill value for a type.

    Notes
    -----
    This differs from :func:`parse_fill_value`, which parses a fill value
    stored in the Array metadata into an in-memory value. This only gives
    the default fill value for some type.

    This is useful for reading Zarr format 2 arrays, which allow the fill
    value to be unspecified.
    """
    if dtype.kind == "S":
        return b""
    elif dtype.kind in "UO":
        return ""
    elif dtype.kind in "Mm":
        return dtype.type("nat")
    elif dtype.kind == "V":
        if dtype.fields is not None:
            default = tuple(_default_fill_value(field[0]) for field in dtype.fields.values())
            return np.array([default], dtype=dtype)
        else:
            return np.zeros(1, dtype=dtype)
    else:
        return dtype.type(0)


def _default_compressor(
    dtype: np.dtype[Any],
) -> dict[str, JSON] | None:
    """Get the default filters and compressor for a dtype.

    https://numpy.org/doc/2.1/reference/generated/numpy.dtype.kind.html
    """
    default_compressor = config.get("array.v2_default_compressor")
    if dtype.kind in "biufcmM":
        dtype_key = "numeric"
    elif dtype.kind in "U":
        dtype_key = "string"
    elif dtype.kind in "OSV":
        dtype_key = "bytes"
    else:
        raise ValueError(f"Unsupported dtype kind {dtype.kind}")

    return cast("dict[str, JSON] | None", default_compressor.get(dtype_key, None))


def _default_filters(
    dtype: np.dtype[Any],
) -> list[dict[str, JSON]] | None:
    """Get the default filters and compressor for a dtype.

    https://numpy.org/doc/2.1/reference/generated/numpy.dtype.kind.html
    """
    default_filters = config.get("array.v2_default_filters")
    if dtype.kind in "biufcmM":
        dtype_key = "numeric"
    elif dtype.kind in "U":
        dtype_key = "string"
    elif dtype.kind in "OS":
        dtype_key = "bytes"
    elif dtype.kind == "V":
        dtype_key = "raw"
    else:
        raise ValueError(f"Unsupported dtype kind {dtype.kind}")

    return cast("list[dict[str, JSON]] | None", default_filters.get(dtype_key, None))
