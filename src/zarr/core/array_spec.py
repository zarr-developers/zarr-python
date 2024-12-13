from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.core.common import (
    MemoryOrder,
    parse_fill_value,
    parse_order,
    parse_shapelike,
    parse_write_empty_chunks,
)
from zarr.core.config import config as zarr_config

if TYPE_CHECKING:
    from zarr.core.buffer import BufferPrototype
    from zarr.core.common import ChunkCoords


@dataclass(frozen=True)
class ArrayConfig:
    order: MemoryOrder
    write_empty_chunks: bool

    def __init__(
        self, *, order: MemoryOrder | None = None, write_empty_chunks: bool | None = None
    ) -> None:
        order_parsed = parse_order(order) if order is not None else zarr_config.get("array.order")
        write_empty_chunks_parsed = (
            parse_write_empty_chunks(write_empty_chunks)
            if write_empty_chunks is not None
            else zarr_config.get("array.write_empty_chunks")
        )
        object.__setattr__(self, "order", order_parsed)
        object.__setattr__(self, "write_empty_chunks", write_empty_chunks_parsed)


@dataclass(frozen=True)
class ArraySpec:
    shape: ChunkCoords
    dtype: np.dtype[Any]
    fill_value: Any
    config: ArrayConfig
    prototype: BufferPrototype

    def __init__(
        self,
        shape: ChunkCoords,
        dtype: np.dtype[Any],
        fill_value: Any,
        config: ArrayConfig,
        prototype: BufferPrototype,
    ) -> None:
        shape_parsed = parse_shapelike(shape)
        dtype_parsed = np.dtype(dtype)
        fill_value_parsed = parse_fill_value(fill_value)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "dtype", dtype_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "prototype", prototype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def order(self) -> MemoryOrder:
        return self.config.order
