from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import numcodecs.abc

from zarr.abc.metadata import Metadata
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.dtype.wrapper import TDType_co, TScalar_co, ZDType, _BaseDType, _BaseScalar

if TYPE_CHECKING:
    from typing import Literal, Self

    import numpy.typing as npt

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import ChunkCoords
    from zarr.core.dtype.wrapper import (
        TBaseDType,
        TBaseScalar,
        TDType_co,
        TScalar_co,
        ZDType,
    )

import json
from dataclasses import dataclass, field, fields, replace

import numcodecs
import numpy as np

from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.chunk_key_encodings import parse_separator
from zarr.core.common import (
    JSON,
    ZARRAY_JSON,
    ZATTRS_JSON,
    MemoryOrder,
    parse_shapelike,
)
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
    dtype: ZDType[_BaseDType, _BaseScalar]
    fill_value: int | float | str | bytes | None = 0
    order: MemoryOrder = "C"
    filters: tuple[numcodecs.abc.Codec, ...] | None = None
    dimension_separator: Literal[".", "/"] = "."
    compressor: numcodecs.abc.Codec | None
    attributes: dict[str, JSON] = field(default_factory=dict)
    zarr_format: Literal[2] = field(init=False, default=2)

    def __init__(
        self,
        *,
        shape: ChunkCoords,
        dtype: ZDType[TDType_co, TScalar_co],
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
        chunks_parsed = parse_shapelike(chunks)
        # TODO: remove this
        if not isinstance(dtype, ZDType):
            raise TypeError
        compressor_parsed = parse_compressor(compressor)
        order_parsed = parse_indexing_order(order)
        dimension_separator_parsed = parse_separator(dimension_separator)
        filters_parsed = parse_filters(filters)
        fill_value_parsed = parse_fill_value(fill_value, dtype=dtype.to_dtype())
        attributes_parsed = parse_attributes(attributes)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "dtype", dtype)
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

    @property
    def shards(self) -> ChunkCoords | None:
        return None

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        zarray_dict = self.to_dict()
        zattrs_dict = zarray_dict.pop("attributes", {})
        json_indent = config.get("json_indent")
        return {
            ZARRAY_JSON: prototype.buffer.from_bytes(
                json.dumps(zarray_dict, indent=json_indent, allow_nan=False).encode()
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
        dtype = get_data_type_from_native_dtype(_data["dtype"])
        _data["dtype"] = dtype
        if dtype.to_dtype().kind in "SV":
            fill_value_encoded = _data.get("fill_value")
            if fill_value_encoded is not None:
                fill_value = base64.standard_b64decode(fill_value_encoded)
                _data["fill_value"] = fill_value

        # zarr v2 allowed arbitrary keys here.
        # We don't want the ArrayV2Metadata constructor to fail just because someone put an
        # extra key in the metadata.
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
        if isinstance(zarray_dict["compressor"], numcodecs.abc.Codec):
            codec_config = zarray_dict["compressor"].get_config()
            # Hotfix for https://github.com/zarr-developers/zarr-python/issues/2647
            if codec_config["id"] == "zstd" and not codec_config.get("checksum", False):
                codec_config.pop("checksum")
            zarray_dict["compressor"] = codec_config

        if zarray_dict["filters"] is not None:
            raw_filters = zarray_dict["filters"]
            # TODO: remove this when we can stratically type the output JSON data structure
            # entirely
            if not isinstance(raw_filters, list | tuple):
                raise TypeError("Invalid type for filters. Expected a list or tuple.")
            new_filters = []
            for f in raw_filters:
                if isinstance(f, numcodecs.abc.Codec):
                    new_filters.append(f.get_config())
                else:
                    new_filters.append(f)
            zarray_dict["filters"] = new_filters

        # serialize the fill value after dtype-specific JSON encoding
        if self.fill_value is not None:
            fill_value = self.dtype.to_json_value(self.fill_value, zarr_format=2)
            zarray_dict["fill_value"] = fill_value

        # serialize the dtype after fill value-specific JSON encoding
        zarray_dict["dtype"] = self.dtype.to_json(zarr_format=2)

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


def parse_fill_value(fill_value: object, dtype: np.dtype[Any]) -> Any:
    """
    Inspect a sequence of codecs / filters for an "object codec", i.e. a codec
    that can serialize object arrays to contiguous bytes. Zarr python
    maintains a hard-coded set of object codec ids. If any element from the input
    has an id that matches one of the hard-coded object codec ids, that id
    is returned immediately.
    """

    if fill_value is None or dtype.hasobject:
        # no fill value
        pass
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
