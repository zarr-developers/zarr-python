"""The v2 compressors that numcodecs 0.16 configures: `zlib`, `gzip`, `bz2`, `lzma`, `blosc`, `zstd` and `lz4`.

Each parameter that numcodecs gives a default is optional. numcodecs
writes the full object with `get_config()`. The canonical form of each
codec fills in those defaults, so a configuration that leaves out a
parameter and one that writes its default are one codec.
"""

from __future__ import annotations

from typing import Final

from zarr_metadata.v2._definition import ZarrV2CodecDefinition
from zarr_metadata.v2._types.codec import (
    ZarrV2BloscCName,
    ZarrV2BloscParameters,
    ZarrV2Bz2Parameters,
    ZarrV2CompressionLevel,
    ZarrV2GzipParameters,
    ZarrV2Lz4Parameters,
    ZarrV2LzmaParameters,
    ZarrV2ZlibParameters,
    ZarrV2ZstdParameters,
)


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
    is the default, and is dropped. Any other `typesize` is kept, whatever
    the `shuffle`: Blosc writes it into each chunk's header and splits
    blocks by it, so it changes the encoded chunk even when nothing is
    shuffled.
    """
    shuffle = configuration.get("shuffle", 1)
    filled: ZarrV2BloscParameters = {
        "cname": configuration.get("cname", "lz4"),
        "clevel": configuration.get("clevel", 5),
        "shuffle": (1 if shuffle else 0) if isinstance(shuffle, bool) else shuffle,
        "blocksize": configuration.get("blocksize", 0),
    }
    typesize = configuration.get("typesize")
    if typesize is None:
        return filled
    return {**filled, "typesize": typesize}


def _zstd_canonical(configuration: ZarrV2ZstdParameters) -> ZarrV2ZstdParameters:
    """Fill in the defaults, `level` 0 and `checksum` false, as numcodecs writes them."""
    return {"level": 0, "checksum": False, **configuration}


def _lz4_canonical(configuration: ZarrV2Lz4Parameters) -> ZarrV2Lz4Parameters:
    """Fill in the default `acceleration`, 1, as numcodecs writes it."""
    return {"acceleration": 1, **configuration}


ZLIB_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="zlib", configuration=ZarrV2ZlibParameters, canonical=_zlib_canonical
)
"""`numcodecs.Zlib`."""
GZIP_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="gzip", configuration=ZarrV2GzipParameters, canonical=_gzip_canonical
)
"""`numcodecs.GZip`."""
BZ2_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="bz2", configuration=ZarrV2Bz2Parameters, canonical=_bz2_canonical
)
"""`numcodecs.BZ2`."""
LZMA_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="lzma", configuration=ZarrV2LzmaParameters, canonical=_lzma_canonical
)
"""`numcodecs.LZMA`."""
BLOSC_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="blosc", configuration=ZarrV2BloscParameters, canonical=_blosc_canonical
)
"""`numcodecs.Blosc`."""
ZSTD_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="zstd", configuration=ZarrV2ZstdParameters, canonical=_zstd_canonical
)
"""`numcodecs.Zstd`."""
LZ4_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="lz4", configuration=ZarrV2Lz4Parameters, canonical=_lz4_canonical
)
"""`numcodecs.LZ4`."""

__all__ = [
    "BLOSC_CODEC_V2",
    "BZ2_CODEC_V2",
    "GZIP_CODEC_V2",
    "LZ4_CODEC_V2",
    "LZMA_CODEC_V2",
    "ZLIB_CODEC_V2",
    "ZSTD_CODEC_V2",
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
