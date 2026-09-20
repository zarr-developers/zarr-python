"""Which entities are in scope when raw JSON is coerced.

The only registry the package needs. A validator reading a document has
to decide, for each extension point, which identifier maps to which
entity — and that decision is the *scope* it validates against, not a
property of the entities themselves.

Two scopes, because the question "is this document valid?" has two useful
answers. `CORE` is what the Zarr v3 specification itself defines, so a
document validating against it uses nothing an implementation could
refuse for being optional. `CORE_AND_EXTENSIONS` adds what
`zarr-extensions` registers and this package models. A name in neither is
not rejected — extension openness — it is simply not judged.

Identifiers are the `name` the metadata carries, with one exception. Every
`r<N>` spelling is one data-type family, so the family registers under an
invented identifier that no real name can collide with; `canonical_name`
folds a spelling onto it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from zarr_metadata.v3._entity import Context
from zarr_metadata.v3._extension_points import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
)
from zarr_metadata.v3.chunk_grid.rectilinear import RectilinearChunkGrid
from zarr_metadata.v3.chunk_grid.regular import RegularChunkGrid
from zarr_metadata.v3.chunk_key_encoding.default import DefaultChunkKeyEncoding
from zarr_metadata.v3.chunk_key_encoding.v2 import V2ChunkKeyEncoding
from zarr_metadata.v3.codec.blosc import BloscCodec
from zarr_metadata.v3.codec.bytes import BytesCodec
from zarr_metadata.v3.codec.crc32c import Crc32cCodec
from zarr_metadata.v3.codec.gzip import GzipCodec
from zarr_metadata.v3.codec.scale_offset import ScaleOffsetCodec
from zarr_metadata.v3.codec.transpose import TransposeCodec
from zarr_metadata.v3.codec.zstd import ZstdCodec

if TYPE_CHECKING:
    from zarr_metadata.v3._entity import MetadataEntity

_CORE_CODECS: Final[dict[str, type[MetadataEntity]]] = {
    BloscCodec.identifier: BloscCodec,
    BytesCodec.identifier: BytesCodec,
    Crc32cCodec.identifier: Crc32cCodec,
    GzipCodec.identifier: GzipCodec,
    TransposeCodec.identifier: TransposeCodec,
}
_EXTENSION_CODECS: Final[dict[str, type[MetadataEntity]]] = {
    ScaleOffsetCodec.identifier: ScaleOffsetCodec,
    ZstdCodec.identifier: ZstdCodec,
}

_CORE_DATA_TYPES: Final[dict[str, type[MetadataEntity]]] = {}
_EXTENSION_DATA_TYPES: Final[dict[str, type[MetadataEntity]]] = {}

_CORE_CHUNK_GRIDS: Final[dict[str, type[MetadataEntity]]] = {
    RegularChunkGrid.identifier: RegularChunkGrid,
}
_EXTENSION_CHUNK_GRIDS: Final[dict[str, type[MetadataEntity]]] = {
    RectilinearChunkGrid.identifier: RectilinearChunkGrid,
}

_CORE_CHUNK_KEY_ENCODINGS: Final[dict[str, type[MetadataEntity]]] = {
    DefaultChunkKeyEncoding.identifier: DefaultChunkKeyEncoding,
    V2ChunkKeyEncoding.identifier: V2ChunkKeyEncoding,
}


CORE: Final = Context(
    {
        CODECS: _CORE_CODECS,
        DATA_TYPE: _CORE_DATA_TYPES,
        CHUNK_GRID: _CORE_CHUNK_GRIDS,
        CHUNK_KEY_ENCODING: _CORE_CHUNK_KEY_ENCODINGS,
    }
)
"""Only what the Zarr v3 specification defines."""

CORE_AND_EXTENSIONS: Final = Context(
    {
        CODECS: {**_CORE_CODECS, **_EXTENSION_CODECS},
        DATA_TYPE: {**_CORE_DATA_TYPES, **_EXTENSION_DATA_TYPES},
        CHUNK_GRID: {**_CORE_CHUNK_GRIDS, **_EXTENSION_CHUNK_GRIDS},
        CHUNK_KEY_ENCODING: _CORE_CHUNK_KEY_ENCODINGS,
    }
)
"""What the specification defines, plus what `zarr-extensions` registers."""


__all__ = [
    "CORE",
    "CORE_AND_EXTENSIONS",
]
