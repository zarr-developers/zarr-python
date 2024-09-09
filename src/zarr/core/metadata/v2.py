from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal

    import numpy.typing as npt
    from typing_extensions import Self

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import JSON, ChunkCoords

import json
from dataclasses import dataclass, field, replace

import numpy as np

from zarr.core.array_spec import ArraySpec
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.chunk_key_encodings import parse_separator
from zarr.core.common import ZARRAY_JSON, ZATTRS_JSON, parse_dtype, parse_shapelike
from zarr.core.config import config, parse_indexing_order
from zarr.core.metadata.common import ArrayMetadata, parse_attributes


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
        fill_value_parsed = parse_fill_value(fill_value, dtype=data_type_parsed)
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
        _ = parse_metadata(self)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data_type

    @property
    def chunks(self) -> ChunkCoords:
        return self.chunk_grid.chunk_shape

    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        def _json_convert(
            o: Any,
        ) -> Any:
            if isinstance(o, np.dtype):
                if o.fields is None:
                    return o.str
                else:
                    return o.descr
            if np.isscalar(o):
                # convert numpy scalar to python type, and pass
                # python types through
                return getattr(o, "item", lambda: o)()
            raise TypeError

        zarray_dict = self.to_dict()
        assert isinstance(zarray_dict, dict)
        zattrs_dict = zarray_dict.pop("attributes", {})
        assert isinstance(zattrs_dict, dict)
        json_indent = config.get("json_indent")
        return {
            ZARRAY_JSON: prototype.buffer.from_bytes(
                json.dumps(zarray_dict, default=_json_convert, indent=json_indent).encode()
            ),
            ZATTRS_JSON: prototype.buffer.from_bytes(
                json.dumps(zattrs_dict, indent=json_indent).encode()
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArrayV2Metadata:
        # make a copy to protect the original from modification
        _data = data.copy()
        # check that the zarr_format attribute is correct
        _ = parse_zarr_format_v2(_data.pop("zarr_format"))
        return cls(**_data)

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


def parse_zarr_format_v2(data: Literal[2]) -> Literal[2]:
    if data == 2:
        return data
    raise ValueError(f"Invalid value. Expected 2. Got {data}.")


def parse_filters(data: list[dict[str, JSON]] | None) -> list[dict[str, JSON]] | None:
    return data


def parse_compressor(data: dict[str, JSON] | None) -> dict[str, JSON] | None:
    return data


def parse_metadata(data: ArrayV2Metadata) -> ArrayV2Metadata:
    if (l_chunks := len(data.chunks)) != (l_shape := len(data.shape)):
        msg = (
            f"The `shape` and `chunks` attributes must have the same length. "
            f"`chunks` has length {l_chunks}, but `shape` has length {l_shape}."
        )
        raise ValueError(msg)
    return data


def parse_fill_value(fill_value: Any, dtype: np.dtype[Any]) -> Any:
    """
    Parse a potential fill value into a value that is compatible with the provided dtype.

    This is a light wrapper around zarr.v2.util.normalize_fill_value.

    Parameters
    ----------
    fill_value: Any
        A potential fill value.
    dtype: np.dtype[Any]
        A numpy dtype.

    Returns
        An instance of `dtype`, or `None`, or any python object (in the case of an object dtype)
    """
    from zarr.v2.util import normalize_fill_value

    return normalize_fill_value(fill_value=fill_value, dtype=dtype)
