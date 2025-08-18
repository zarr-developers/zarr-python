from __future__ import annotations

from zarr.codecs.numcodecs._codecs import (
    BZ2,
    CRC32,
    CRC32C,
    LZ4,
    LZMA,
    ZFPY,
    Adler32,
    AsType,
    BitRound,
    Blosc,
    Delta,
    FixedScaleOffset,
    Fletcher32,
    GZip,
    JenkinsLookup3,
    PackBits,
    PCodec,
    Quantize,
    Shuffle,
    Zlib,
    Zstd,
    _NumcodecsArrayArrayCodec,
    _NumcodecsArrayBytesCodec,
    _NumcodecsBytesBytesCodec,
    _NumcodecsCodec,
)
from zarr.registry import register_codec

register_codec("numcodecs.bz2", BZ2)
register_codec("numcodecs.crc32", CRC32)
register_codec("numcodecs.crc32c", CRC32C)
register_codec("numcodecs.lz4", LZ4)
register_codec("numcodecs.lzma", LZMA)
register_codec("numcodecs.zfpy", ZFPY)
register_codec("numcodecs.adler32", Adler32)
register_codec("numcodecs.astype", AsType)
register_codec("numcodecs.bitround", BitRound)
register_codec("numcodecs.blosc", Blosc)
register_codec("numcodecs.delta", Delta)
register_codec("numcodecs.fixedscaleoffset", FixedScaleOffset)
register_codec("numcodecs.fletcher32", Fletcher32)
register_codec("numcodecs.gzip", GZip)
register_codec("numcodecs.jenkins_lookup3", JenkinsLookup3)
register_codec("numcodecs.pcodec", PCodec)
register_codec("numcodecs.packbits", PackBits)
register_codec("numcodecs.quantize", Quantize)
register_codec("numcodecs.shuffle", Shuffle)
register_codec("numcodecs.zlib", Zlib)
register_codec("numcodecs.zstd", Zstd)

__all__ = [
    "BZ2",
    "CRC32",
    "CRC32C",
    "LZ4",
    "LZMA",
    "ZFPY",
    "Adler32",
    "AsType",
    "BitRound",
    "Blosc",
    "Delta",
    "FixedScaleOffset",
    "Fletcher32",
    "GZip",
    "JenkinsLookup3",
    "PCodec",
    "PackBits",
    "Quantize",
    "Shuffle",
    "Zlib",
    "Zstd",
    "_NumcodecsArrayArrayCodec",
    "_NumcodecsArrayBytesCodec",
    "_NumcodecsBytesBytesCodec",
    "_NumcodecsCodec",
]
