from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal

    from typing_extensions import Self

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from zarr.abc.metadata import Metadata
from zarr.array_spec import ArraySpec
from zarr.buffer import Buffer, BufferPrototype
from zarr.chunk_grids import ChunkGrid
from zarr.common import JSON, ChunkCoords, ZarrFormat


@dataclass(frozen=True, kw_only=True)
class ArrayMetadata(Metadata, ABC):
    shape: ChunkCoords
    fill_value: Any
    chunk_grid: ChunkGrid
    attributes: dict[str, JSON]
    zarr_format: ZarrFormat

    @property
    @abstractmethod
    def dtype(self) -> np.dtype[Any]:
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @abstractmethod
    def get_chunk_spec(
        self, _chunk_coords: ChunkCoords, order: Literal["C", "F"], prototype: BufferPrototype
    ) -> ArraySpec:
        pass

    @abstractmethod
    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        pass

    @abstractmethod
    def to_buffer_dict(self, prototype: BufferPrototype) -> dict[str, Buffer]:
        pass

    @abstractmethod
    def update_shape(self, shape: ChunkCoords) -> Self:
        pass

    @abstractmethod
    def update_attributes(self, attributes: dict[str, JSON]) -> Self:
        pass


def parse_attributes(data: None | dict[str, JSON]) -> dict[str, JSON]:
    """
    Normalize `None` to an empty dict. All other values pass through.
    """
    if data is None:
        return {}

    return data
