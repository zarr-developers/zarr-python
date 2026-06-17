# DRAFT — public chunk-structure introspection (design/chunk-layout.md).
"""Public, typed chunk-structure introspection for zarr arrays.

``ChunkLayout`` distills the *declared* chunk structure of an array -- its chunk
grid metadata together with the chunk-structuring codecs in its pipeline -- into
the form a consumer can feed back into :func:`zarr.create_array`. See
``design/chunk-layout.md`` for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.core.metadata.v2 import ArrayV2Metadata
    from zarr.core.metadata.v3 import ArrayV3Metadata


@dataclass(frozen=True, kw_only=True)
class ChunkLayout:
    """Declared chunk structure of an array.

    A distillation of the chunk grid metadata and sharding codec
    configuration. Extent-free: sizes are as declared, never clipped
    to the array shape. Canonical: a dimension whose declared edge
    lengths are all equal is normalized to the bare uniform size, so
    layouts of the same abstract grid compare equal regardless of the
    metadata that declared them.
    """

    chunks: tuple[int | tuple[int, ...], ...]
    """Per-dimension chunk spec at this level: a bare int (uniform size)
    or explicit edge lengths -- the same union ``create_array`` accepts."""

    inner: ChunkLayout | None = None
    """Structure within each chunk of this level, or None if chunks are
    opaque (no sharding)."""

    def __post_init__(self) -> None:
        # Reuse the metadata-layer validator (positive ints; non-empty edge
        # tuples), which also normalizes inner sequences to tuples and is
        # precisely typed as ``Sequence[int | Sequence[int]]``.
        from zarr.core.metadata.v3 import _validate_chunk_shapes

        validated = _validate_chunk_shapes(self.chunks)
        # canonicalization: a uniform edge tuple collapses to its int
        object.__setattr__(
            self,
            "chunks",
            tuple(c[0] if isinstance(c, tuple) and len(set(c)) == 1 else c for c in validated),
        )
        if self.inner is not None and self.inner.ndim != self.ndim:
            raise ValueError(f"inner layout has {self.inner.ndim} dimensions, expected {self.ndim}")

    @property
    def ndim(self) -> int:
        return len(self.chunks)

    @property
    def is_regular(self) -> bool:
        """True if every dimension at this level has one uniform chunk size."""
        return all(isinstance(c, int) for c in self.chunks)

    @property
    def is_sharded(self) -> bool:
        """True if chunks at this level have internal sub-chunk structure."""
        return self.inner is not None

    @property
    def flattened_levels(self) -> tuple[ChunkLayout, ...]:
        """All nesting levels, outermost (storage granularity) to innermost."""
        return (self, *(self.inner.flattened_levels if self.inner is not None else ()))

    @property
    def innermost(self) -> ChunkLayout:
        """The innermost level of declared subdivision.

        Whether this unit is independently decodable depends on the
        full codec pipeline, not on the declared structure alone.
        """
        return self.inner.innermost if self.inner is not None else self

    @classmethod
    def from_metadata(cls, metadata: ArrayV2Metadata | ArrayV3Metadata) -> ChunkLayout:
        """Derive a :class:`ChunkLayout` from array metadata.

        Raises ``TypeError`` for a chunk grid kind this version cannot distill.
        """
        # Imported lazily to avoid an import cycle with the metadata modules.
        from zarr.core.metadata.v2 import ArrayV2Metadata
        from zarr.core.metadata.v3 import (
            RectilinearChunkGridMetadata,
            RegularChunkGridMetadata,
        )

        if isinstance(metadata, ArrayV2Metadata):
            return cls(chunks=tuple(metadata.chunks))

        inner: ChunkLayout | None = None
        if metadata.codecs:
            inner = metadata.codecs[0].inner_chunk_layout()

        grid = metadata.chunk_grid
        if isinstance(grid, RegularChunkGridMetadata):
            return cls(chunks=tuple(grid.chunk_shape), inner=inner)
        if isinstance(grid, RectilinearChunkGridMetadata):
            return cls(
                chunks=tuple(tuple(s) if not isinstance(s, int) else s for s in grid.chunk_shapes),
                inner=inner,
            )
        raise TypeError(f"Cannot derive a ChunkLayout from chunk grid {type(grid).__name__}")
