from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from zarr.common import ChunkCoords, parse_dtype, parse_fill_value, parse_order, parse_shapelike

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ArraySpec:
    shape: ChunkCoords
    dtype: np.dtype[Any]
    fill_value: Any
    order: Literal["C", "F"]

    def __init__(
        self, shape: ChunkCoords, dtype: np.dtype[Any], fill_value: Any, order: Literal["C", "F"]
    ) -> None:
        shape_parsed = parse_shapelike(shape)
        dtype_parsed = parse_dtype(dtype)
        fill_value_parsed = parse_fill_value(fill_value)
        order_parsed = parse_order(order)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "dtype", dtype_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "order", order_parsed)

    @property
    def ndim(self) -> int:
        return len(self.shape)
