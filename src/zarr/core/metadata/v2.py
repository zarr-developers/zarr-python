from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict, cast

from zarr.abc.codec import ArrayArrayCodec, Codec
from zarr.abc.metadata import Metadata
from zarr.abc.numcodec import Numcodec
from zarr.codecs._v2 import NumcodecsWrapper
from zarr.codecs.blosc import BloscCodec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.dtype import get_data_type_from_json
from zarr.core.dtype.common import OBJECT_CODEC_IDS
from zarr.errors import ZarrUserWarning
from zarr.registry import get_codec

if TYPE_CHECKING:
    from typing import Literal, Self

    import numpy.typing as npt

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.dtype.common import DTypeSpec_V2
    from zarr.core.dtype.wrapper import (
        TBaseDType,
        TBaseScalar,
        TDType_co,
        TScalar_co,
    )

import json
from dataclasses import dataclass, field, fields, replace

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
from zarr.core.dtype.wrapper import (
    ZDType,
)
from zarr.core.metadata.common import parse_attributes


class ArrayV2MetadataDict(TypedDict):
    """
    A typed dictionary model for Zarr format 2 metadata.
    """

    zarr_format: Literal[2]
    attributes: dict[str, JSON]


# Union of acceptable types for v2 compressors
CompressorLike_V2: TypeAlias = Mapping[str, JSON] | Numcodec | Codec


@dataclass(frozen=True, kw_only=True)
class ArrayV2Metadata(Metadata):
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: ZDType[TBaseDType, TBaseScalar]
    fill_value: int | float | str | bytes | None = None
    order: MemoryOrder = "C"
    filters: tuple[Codec, ...] | None = None
    dimension_separator: Literal[".", "/"] = "."
    compressor: Codec
    attributes: dict[str, JSON] = field(default_factory=dict)
    zarr_format: Literal[2] = field(init=False, default=2)

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TDType_co, TScalar_co],
        chunks: tuple[int, ...],
        fill_value: Any,
        order: MemoryOrder,
        dimension_separator: Literal[".", "/"] = ".",
        compressor: CompressorLike_V2 | None = None,
        filters: Iterable[CompressorLike_V2] | None = None,
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
        compressor_parsed = parse_compressor(compressor, dtype)
        order_parsed = parse_indexing_order(order)
        dimension_separator_parsed = parse_separator(dimension_separator)
        filters_parsed = parse_filters(filters, dtype)
        fill_value_parsed: TBaseScalar | None
        if fill_value is not None:
            fill_value_parsed = dtype.cast_scalar(fill_value)
        else:
            fill_value_parsed = fill_value

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

    @cached_property
    def chunk_grid(self) -> RegularChunkGrid:
        return RegularChunkGrid(chunk_shape=self.chunks)

    @property
    def shards(self) -> tuple[int, ...] | None:
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

        # To resolve a numpy object dtype array, we need to search for an object codec,
        # which could be in filters or as a compressor.
        # we will reference a hard-coded collection of object codec ids for this search.

        _filters, _compressor = (data.get("filters"), data.get("compressor"))
        if _filters is not None:
            _filters = cast("tuple[dict[str, JSON], ...]", _filters)
            object_codec_id = get_object_codec_id(tuple(_filters) + (_compressor,))
        else:
            object_codec_id = get_object_codec_id((_compressor,))
        # we add a layer of indirection here around the dtype attribute of the array metadata
        # because we also need to know the object codec id, if any, to resolve the data type

        dtype_spec: DTypeSpec_V2 = {
            "name": data["dtype"],
            "object_codec_id": object_codec_id,
        }
        dtype = get_data_type_from_json(dtype_spec, zarr_format=2)

        _data["dtype"] = dtype
        fill_value_encoded = _data.get("fill_value")
        if fill_value_encoded is not None:
            fill_value = dtype.from_json_scalar(fill_value_encoded, zarr_format=2)
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
            warnings.warn(msg, ZarrUserWarning, stacklevel=1)
            _data["filters"] = None

        _data = {k: v for k, v in _data.items() if k in expected}

        return cls(**_data)

    def to_dict(self) -> dict[str, JSON]:
        zarray_dict = super().to_dict()
        if self.compressor is not None:
            zarray_dict["compressor"] = self.compressor.to_json(zarr_format=2)
        else:
            zarray_dict["compressor"] = None
        new_filters = []
        if zarray_dict["filters"] is not None:
            new_filters.extend([f.to_json(zarr_format=2) for f in self.filters])
        else:
            new_filters = None
        zarray_dict["filters"] = new_filters

        # serialize the fill value after dtype-specific JSON encoding
        if self.fill_value is not None:
            fill_value = self.dtype.to_json_scalar(self.fill_value, zarr_format=2)
            zarray_dict["fill_value"] = fill_value

        # serialize the dtype after fill value-specific JSON encoding
        zarray_dict["dtype"] = self.dtype.to_json(zarr_format=2)["name"]

        return zarray_dict

    def get_chunk_spec(
        self, _chunk_coords: tuple[int, ...], array_config: ArrayConfig, prototype: BufferPrototype
    ) -> ArraySpec:
        return ArraySpec(
            shape=self.chunks,
            dtype=self.dtype,
            fill_value=self.fill_value,
            config=array_config,
            prototype=prototype,
        )

    def encode_chunk_key(self, chunk_coords: tuple[int, ...]) -> str:
        chunk_identifier = self.dimension_separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier

    def update_shape(self, shape: tuple[int, ...]) -> Self:
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


def _parse_codec(data: object, dtype: ZDType[Any, Any]) -> Codec | NumcodecsWrapper:
    """
    Resolve a potential codec.
    """
    if isinstance(data, (Codec, NumcodecsWrapper)):
        # TERRIBLE HACK
        if isinstance(data, BloscCodec):
            return data.evolve_from_array_spec(
                ArraySpec(
                    shape=(1,),
                    dtype=dtype,
                    fill_value=None,
                    config=ArrayConfig.from_dict({}),  # TODO: config is not needed here.
                    prototype=default_buffer_prototype(),  # TODO: prototype is not needed here.
                )
            )
        return data

    if isinstance(data, Numcodec):
        try:
            # attempt to get a v3-api compatible version of this codec from the registry
            return get_codec(data.get_config())
        except KeyError:
            # if we could not find a v3-api compatible version of this codec, wrap it
            # in a NumcodecsWrapper
            return NumcodecsWrapper(codec=data)

    if isinstance(data, Mapping):
        return get_codec(data)

    raise TypeError(
        f"Invalid compressor. Expected None, a numcodecs.abc.Codec, or a dict representation of a numcodecs.abc.Codec. Got {type(data)} instead."
    )


def parse_filters(
    data: object, dtype: ZDType[Any, Any]
) -> tuple[ArrayArrayCodec | NumcodecsWrapper, ...] | None:
    """
    Parse a potential tuple of filters
    """
    out: list[Codec | NumcodecsWrapper] = []

    if data is None:
        return data
    if not isinstance(data, Iterable):
        return (_parse_codec(data, dtype),)
    out = [(_parse_codec(val, dtype)) for val in data]
    if len(out) == 0:
        # Per the v2 spec, an empty tuple is not allowed -- use None to express "no filters"
        return None
    else:
        return tuple(out)


def parse_compressor(data: object, dtype: ZDType[Any, Any]) -> Codec | NumcodecsWrapper | None:
    """
    Parse a potential compressor.
    """
    # TODO: only validate the compressor in one place. currently we do it twice, once in init_array
    # and again when constructing metadata
    if data is None:
        return data
    return _parse_codec(data, dtype)


def parse_metadata(data: ArrayV2Metadata) -> ArrayV2Metadata:
    if (l_chunks := len(data.chunks)) != (l_shape := len(data.shape)):
        msg = (
            f"The `shape` and `chunks` attributes must have the same length. "
            f"`chunks` has length {l_chunks}, but `shape` has length {l_shape}."
        )
        raise ValueError(msg)
    return data


def is_object_codec(codec: JSON) -> bool:
    return codec.get("id") in OBJECT_CODEC_IDS


def get_object_codec_id(maybe_object_codecs: Sequence[Mapping[str, object]]) -> str | None:
    """
    Inspect a sequence of codecs / filters for an "object codec", i.e. a codec
    that can serialize object arrays to contiguous bytes. Zarr python
    maintains a hard-coded set of object codec ids. If any element from the input
    has an id that matches one of the hard-coded object codec ids, that id
    is returned immediately.
    """
    object_codec_id = None
    for maybe_object_codec in maybe_object_codecs:
        if (
            isinstance(maybe_object_codec, dict)
            and maybe_object_codec.get("id") in OBJECT_CODEC_IDS
        ):
            return cast("str", maybe_object_codec["id"])
    return object_codec_id
