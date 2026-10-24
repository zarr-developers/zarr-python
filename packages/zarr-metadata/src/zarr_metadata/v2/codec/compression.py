"""The v2 compressors that numcodecs 0.16 configures: `zlib`, `gzip`, `bz2`, `lzma`, `blosc`, `zstd` and `lz4`.

Each parameter that numcodecs gives a default is optional. numcodecs
writes the full object with `get_config()`. The canonical form of each
codec fills in those defaults, so a configuration that leaves out a
parameter and one that writes its default are one codec.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, NotRequired

from annotated_types import Ge, Interval
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._common import (
    JSONValue,
)
from zarr_metadata.v2._definition import ZarrV2CodecDefinition

ZarrV2CompressionLevel = Annotated[int, Interval(ge=-1, le=9)]
"""A zlib-style compression level: 0 to 9, or -1 for zlib's default. numcodecs writes the level as given."""

_CInt = Annotated[int, Interval(ge=-(2**31), le=2**31 - 1)]
"""An integer that numcodecs passes to C as an `int`: one that fits in 32 bits, with a sign. numcodecs raises `OverflowError` for any other."""


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
    """`numcodecs.LZMA(format=1, check=-1, preset=None, filters=None)`, with the parameters as the `lzma` module accepts them."""

    format: NotRequired[ReadOnly[Annotated[int, Interval(ge=0, le=3)]]]
    check: NotRequired[ReadOnly[int]]
    preset: NotRequired[ReadOnly[Annotated[int, Ge(0)] | None]]
    filters: NotRequired[ReadOnly[tuple[JSONValue, ...] | None]]


ZarrV2BloscCName = Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
"""The compressors inside Blosc."""


class ZarrV2BloscParameters(TypedDict, closed=True):
    """`numcodecs.Blosc(cname='lz4', clevel=5, shuffle=1, blocksize=0, typesize=None)`.

    For `shuffle`, -1 is automatic, 0 is none, 1 is byte shuffle, and 2 is
    bit shuffle. numcodecs writes a `shuffle` given as `True` or `False`
    as it was given, and reads it as 1 or 0. A `blocksize` of 0 lets Blosc
    choose the block size. numcodecs reads any `blocksize` that fits in a
    C `int`, negative ones included.
    """

    cname: NotRequired[ReadOnly[ZarrV2BloscCName]]
    clevel: NotRequired[ReadOnly[Annotated[int, Interval(ge=0, le=9)]]]
    shuffle: NotRequired[ReadOnly[Literal[-1, 0, 1, 2] | bool]]
    blocksize: NotRequired[ReadOnly[_CInt]]
    typesize: NotRequired[ReadOnly[Annotated[int, Ge(1)] | None]]


class ZarrV2ZstdParameters(TypedDict, closed=True):
    """`numcodecs.Zstd(level=0, checksum=False)`. Level 0 is zstd's default level, and negative levels give up compression ratio for speed. zstd clamps a level outside its own range, -131072 to 22, to that range, so numcodecs reads any level that fits in a C `int`."""

    level: NotRequired[ReadOnly[_CInt]]
    checksum: NotRequired[ReadOnly[bool]]


class ZarrV2Lz4Parameters(TypedDict, closed=True):
    """`numcodecs.LZ4(acceleration=1)`."""

    acceleration: NotRequired[ReadOnly[int]]


def _zlib_canonical(configuration: ZarrV2ZlibParameters) -> ZarrV2ZlibParameters:
    """Fill in the default `level`, 1, as numcodecs writes it."""
    return {"level": 1, **configuration}


def _gzip_canonical(configuration: ZarrV2GzipParameters) -> ZarrV2GzipParameters:
    """Fill in the default `level`, 1, as numcodecs writes it."""
    return {"level": 1, **configuration}


def _bz2_canonical(configuration: ZarrV2Bz2Parameters) -> ZarrV2Bz2Parameters:
    """Fill in the default `level`, 1, as numcodecs writes it."""
    return {"level": 1, **configuration}


def _lzma_canonical(configuration: ZarrV2LzmaParameters) -> ZarrV2LzmaParameters:
    """Fill in the defaults, as numcodecs writes them: `format` 1 (`lzma.FORMAT_XZ`), `check` -1 (the format's own check), and null `preset` and `filters`."""
    return {"format": 1, "check": -1, "preset": None, "filters": None, **configuration}


def _blosc_canonical(configuration: ZarrV2BloscParameters) -> ZarrV2BloscParameters:
    """Fill in the defaults, as numcodecs writes them, and write a `shuffle` of `true` or `false` as 1 or 0, which numcodecs reads it as.

    numcodecs 0.16 reads `typesize` but never writes it. A null `typesize`
    is the default, and is dropped. Under `shuffle` 0 nothing is shuffled,
    so `typesize` is dropped there too, as the v3 blosc codec drops it
    under `noshuffle`. Under any other `shuffle`, it sets the element size
    that Blosc shuffles by, and is kept.
    """
    shuffle = configuration.get("shuffle", 1)
    filled: ZarrV2BloscParameters = {
        "cname": configuration.get("cname", "lz4"),
        "clevel": configuration.get("clevel", 5),
        "shuffle": (1 if shuffle else 0) if isinstance(shuffle, bool) else shuffle,
        "blocksize": configuration.get("blocksize", 0),
    }
    typesize = configuration.get("typesize")
    if typesize is None or filled["shuffle"] == 0:
        return filled
    return {**filled, "typesize": typesize}


def _zstd_canonical(configuration: ZarrV2ZstdParameters) -> ZarrV2ZstdParameters:
    """Fill in the defaults, `level` 0 and `checksum` false, as numcodecs writes them."""
    return {"level": 0, "checksum": False, **configuration}


def _lz4_canonical(configuration: ZarrV2Lz4Parameters) -> ZarrV2Lz4Parameters:
    """Fill in the default `acceleration`, 1, as numcodecs writes it."""
    return {"acceleration": 1, **configuration}


ZLIB_V2: Final = ZarrV2CodecDefinition(
    name="zlib", configuration=ZarrV2ZlibParameters, canonical=_zlib_canonical
)
"""`numcodecs.Zlib`."""
GZIP_V2: Final = ZarrV2CodecDefinition(
    name="gzip", configuration=ZarrV2GzipParameters, canonical=_gzip_canonical
)
"""`numcodecs.GZip`."""
BZ2_V2: Final = ZarrV2CodecDefinition(
    name="bz2", configuration=ZarrV2Bz2Parameters, canonical=_bz2_canonical
)
"""`numcodecs.BZ2`."""
LZMA_V2: Final = ZarrV2CodecDefinition(
    name="lzma", configuration=ZarrV2LzmaParameters, canonical=_lzma_canonical
)
"""`numcodecs.LZMA`."""
BLOSC_V2: Final = ZarrV2CodecDefinition(
    name="blosc", configuration=ZarrV2BloscParameters, canonical=_blosc_canonical
)
"""`numcodecs.Blosc`."""
ZSTD_V2: Final = ZarrV2CodecDefinition(
    name="zstd", configuration=ZarrV2ZstdParameters, canonical=_zstd_canonical
)
"""`numcodecs.Zstd`."""
LZ4_V2: Final = ZarrV2CodecDefinition(
    name="lz4", configuration=ZarrV2Lz4Parameters, canonical=_lz4_canonical
)
"""`numcodecs.LZ4`."""

__all__ = [
    "BLOSC_V2",
    "BZ2_V2",
    "GZIP_V2",
    "LZ4_V2",
    "LZMA_V2",
    "ZLIB_V2",
    "ZSTD_V2",
    "ZarrV2BloscCName",
    "ZarrV2BloscParameters",
    "ZarrV2Bz2Parameters",
    "ZarrV2CompressionLevel",
    "ZarrV2GzipParameters",
    "ZarrV2Lz4Parameters",
    "ZarrV2LzmaParameters",
    "ZarrV2ZlibParameters",
    "ZarrV2ZstdParameters",
]
