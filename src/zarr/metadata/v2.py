from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import Self

    from zarr.common import JSON, ChunkCoords

import json
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from zarr.buffer import Buffer, BufferPrototype
from zarr.chunk_key_encodings import parse_separator
from zarr.common import ZARRAY_JSON, ZATTRS_JSON
from zarr.config import config, parse_indexing_order
from zarr.metadata.common import (
    ArrayMetadataBase,
    ArraySpec,
    RegularChunkGrid,
    parse_attributes,
    parse_dtype,
    parse_shapelike,
)
from zarr.metadata.v3 import parse_fill_value


def parse_zarr_format(data: Any) -> Literal[2]:
    if data == 2:
        return 2
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


@dataclass(frozen=True, kw_only=True)
class ArrayV2Metadata(ArrayMetadataBase):
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

    def to_buffer_dict(self) -> dict[str, Buffer]:
        def _json_convert(
            o: Any,
        ) -> str | list[tuple[str, str] | tuple[str, str, tuple[int, ...]]]:
            if isinstance(o, np.dtype):
                if o.fields is None:
                    return o.str
                else:
                    return o.descr
            if np.isscalar(o):
                return str(o)
            raise TypeError

        zarray_dict = self.to_dict()
        assert isinstance(zarray_dict, dict)
        zattrs_dict = zarray_dict.pop("attributes", {})
        assert isinstance(zattrs_dict, dict)
        json_indent = config.get("json_indent")
        return {
            ZARRAY_JSON: Buffer.from_bytes(
                json.dumps(zarray_dict, default=_json_convert, indent=json_indent).encode()
            ),
            ZATTRS_JSON: Buffer.from_bytes(json.dumps(zattrs_dict, indent=json_indent).encode()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArrayV2Metadata:
        # check that the zarr_format attribute is correct
        _ = parse_zarr_format(data.pop("zarr_format"))
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
