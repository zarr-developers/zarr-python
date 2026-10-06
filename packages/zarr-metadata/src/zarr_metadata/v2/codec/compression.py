"""The v2 compressors numcodecs 0.16 configures: `zlib`, `gzip`, `bz2`, `lzma`, `blosc`, `zstd` and `lz4`.

Each parameter numcodecs defaults is optional; what it writes with
`get_config()` is the full object.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, NotRequired

from annotated_types import Ge, Interval
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._common import (
    JSONValue,  # noqa: TC001 - a TypedDict's annotations are evaluated at run time
)
from zarr_metadata.v2._definition import ZarrV2CodecDefinition

ZarrV2CompressionLevel = Annotated[int, Interval(ge=0, le=9)]
"""A zlib-style compression level, 0 to 9."""


class ZarrV2ZlibParameters(TypedDict, closed=True):
    """`numcodecs.Zlib(level=1)`."""

    level: NotRequired[ReadOnly[ZarrV2CompressionLevel]]


class ZarrV2GzipParameters(TypedDict, closed=True):
    """`numcodecs.GZip(level=1)`."""

    level: NotRequired[ReadOnly[ZarrV2CompressionLevel]]


class ZarrV2Bz2Parameters(TypedDict, closed=True):
    """`numcodecs.BZ2(level=1)`: 1 to 9."""

    level: NotRequired[ReadOnly[Annotated[int, Interval(ge=1, le=9)]]]


class ZarrV2LzmaParameters(TypedDict, closed=True):
    """`numcodecs.LZMA(format=1, check=-1, preset=None, filters=None)`, as `lzma` takes them."""

    format: NotRequired[ReadOnly[Annotated[int, Interval(ge=0, le=3)]]]
    check: NotRequired[ReadOnly[int]]
    preset: NotRequired[ReadOnly[Annotated[int, Ge(0)] | None]]
    filters: NotRequired[ReadOnly[tuple[JSONValue, ...] | None]]


BloscCName = Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
"""The compressors inside Blosc."""


class ZarrV2BloscParameters(TypedDict, closed=True):
    """`numcodecs.Blosc(cname='lz4', clevel=5, shuffle=1, blocksize=0, typesize=None)`: shuffle -1 is automatic, 0 none, 1 byte, 2 bit."""

    cname: NotRequired[ReadOnly[BloscCName]]
    clevel: NotRequired[ReadOnly[ZarrV2CompressionLevel]]
    shuffle: NotRequired[ReadOnly[Literal[-1, 0, 1, 2]]]
    blocksize: NotRequired[ReadOnly[Annotated[int, Ge(0)]]]
    typesize: NotRequired[ReadOnly[Annotated[int, Ge(1)] | None]]


class ZarrV2ZstdParameters(TypedDict, closed=True):
    """`numcodecs.Zstd(level=0, checksum=False)`: 0 is zstd's default level, and negative levels trade ratio for speed."""

    level: NotRequired[ReadOnly[Annotated[int, Interval(ge=-131072, le=22)]]]
    checksum: NotRequired[ReadOnly[bool]]


class ZarrV2Lz4Parameters(TypedDict, closed=True):
    """`numcodecs.LZ4(acceleration=1)`."""

    acceleration: NotRequired[ReadOnly[int]]


ZLIB_V2: Final = ZarrV2CodecDefinition(name="zlib", configuration=ZarrV2ZlibParameters)
"""`numcodecs.Zlib`."""
GZIP_V2: Final = ZarrV2CodecDefinition(name="gzip", configuration=ZarrV2GzipParameters)
"""`numcodecs.GZip`."""
BZ2_V2: Final = ZarrV2CodecDefinition(name="bz2", configuration=ZarrV2Bz2Parameters)
"""`numcodecs.BZ2`."""
LZMA_V2: Final = ZarrV2CodecDefinition(name="lzma", configuration=ZarrV2LzmaParameters)
"""`numcodecs.LZMA`."""
BLOSC_V2: Final = ZarrV2CodecDefinition(name="blosc", configuration=ZarrV2BloscParameters)
"""`numcodecs.Blosc`."""
ZSTD_V2: Final = ZarrV2CodecDefinition(name="zstd", configuration=ZarrV2ZstdParameters)
"""`numcodecs.Zstd`."""
LZ4_V2: Final = ZarrV2CodecDefinition(name="lz4", configuration=ZarrV2Lz4Parameters)
"""`numcodecs.LZ4`."""

__all__ = [
    "BLOSC_V2",
    "BZ2_V2",
    "GZIP_V2",
    "LZ4_V2",
    "LZMA_V2",
    "ZLIB_V2",
    "ZSTD_V2",
    "BloscCName",
    "ZarrV2BloscParameters",
    "ZarrV2Bz2Parameters",
    "ZarrV2CompressionLevel",
    "ZarrV2GzipParameters",
    "ZarrV2Lz4Parameters",
    "ZarrV2LzmaParameters",
    "ZarrV2ZlibParameters",
    "ZarrV2ZstdParameters",
]
