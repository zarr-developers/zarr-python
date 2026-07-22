"""Structural typing for the chunk-grid surface used by chunk resolution.

`chunk_resolution` needs only a narrow slice of a chunk grid: the per-dimension
mapping between storage indices and chunk coordinates. Rather than import
zarr's concrete `ChunkGrid`, we type against these Protocols. zarr's
`ChunkGrid` / `DimensionGrid` satisfy them structurally, so no zarr import is
needed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt


class DimensionGridLike(Protocol):
    """The per-dimension chunk-mapping surface consumed by chunk resolution."""

    def index_to_chunk(self, idx: int) -> int: ...
    def chunk_offset(self, chunk_ix: int) -> int: ...
    def chunk_size(self, chunk_ix: int) -> int: ...
    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]: ...


class ChunkGridLike(Protocol):
    """A chunk grid exposing its per-dimension grids via `_dimensions`.

    Typed as a read-only, covariant `Sequence` so a concrete grid whose
    `_dimensions` is a `tuple` of a more specific dimension type structurally
    satisfies this Protocol.
    """

    @property
    def _dimensions(self) -> Sequence[DimensionGridLike]: ...
