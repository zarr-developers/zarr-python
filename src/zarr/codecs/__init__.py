from __future__ import annotations

from zarr.codecs.blosc import BloscCodec
from zarr.codecs.bytes import BytesCodec, Endian
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.numcodecs import (
    BZ2,
    CRC32,
    LZ4,
    LZMA,
    ZFPY,
    Adler32,
    AsType,
    BitRound,
    FixedScaleOffset,
    Fletcher32,
    JenkinsLookup3,
    PackBits,
    PCodec,
    Quantize,
    Shuffle,
    Zlib,
)
from zarr.codecs.numcodecs.delta import Delta
from zarr.codecs.sharding import ShardingCodec, ShardingCodecIndexLocation
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.vlen_utf8 import VLenBytesCodec, VLenUTF8Codec
from zarr.codecs.zstd import ZstdCodec
from zarr.registry import register_codec

__all__ = [
    "BloscCodec",
    "BytesCodec",
    "Crc32cCodec",
    "Endian",
    "GzipCodec",
    "ShardingCodec",
    "ShardingCodecIndexLocation",
    "TransposeCodec",
    "VLenBytesCodec",
    "VLenUTF8Codec",
    "ZstdCodec",
]

register_codec("blosc", BloscCodec)
register_codec("bytes", BytesCodec)

# compatibility with earlier versions of ZEP1
register_codec("endian", BytesCodec)
register_codec("crc32c", Crc32cCodec)
register_codec("gzip", GzipCodec)
register_codec("sharding_indexed", ShardingCodec)
register_codec("zstd", ZstdCodec)
register_codec("vlen-utf8", VLenUTF8Codec)
register_codec("vlen-bytes", VLenBytesCodec)
register_codec("transpose", TransposeCodec)

# Register all the codecs formerly contained in numcodecs.zarr3 except
# for the codecs that have Zarr V3 specific implementations,
# namely, Blosc, CRC32C, Gzip

register_codec("numcodecs.bz2", BZ2, qualname="zarr.codecs.numcodecs.BZ2")
register_codec("bz2", BZ2, qualname="zarr.codecs.numcodecs.BZ2")

register_codec("numcodecs.crc32", CRC32, qualname="zarr.codecs.numcodecs.CRC32")
register_codec("crc32", CRC32, qualname="zarr.codecs.numcodecs.CRC32")

register_codec("numcodecs.lz4", LZ4, qualname="zarr.codecs.numcodecs.LZ4")
register_codec("lz4", LZ4, qualname="zarr.codecs.numcodecs.LZ4")

register_codec("numcodecs.lzma", LZMA, qualname="zarr.codecs.numcodecs.LZMA")
register_codec("lzma", LZMA, qualname="zarr.codecs.numcodecs.LZMA")

register_codec("numcodecs.zfpy", ZFPY, qualname="zarr.codecs.numcodecs.ZFPY")
register_codec("zfpy", ZFPY, qualname="zarr.codecs.numcodecs.ZFPY")

register_codec("numcodecs.adler32", Adler32, qualname="zarr.codecs.numcodecs.Adler32")
register_codec("adler32", Adler32, qualname="zarr.codecs.numcodecs.Adler32")

register_codec("numcodecs.astype", AsType, qualname="zarr.codecs.numcodecs.AsType")
register_codec("astype", AsType, qualname="zarr.codecs.numcodecs.AsType")

register_codec("numcodecs.bitround", BitRound, qualname="zarr.codecs.numcodecs.BitRound")
register_codec("bitround", BitRound, qualname="zarr.codecs.numcodecs.BitRound")

register_codec("numcodecs.delta", Delta, qualname="zarr.codecs.numcodecs.Delta")
register_codec("delta", Delta, qualname="zarr.codecs.numcodecs.Delta")

register_codec(
    "numcodecs.fixedscaleoffset",
    FixedScaleOffset,
    qualname="zarr.codecs.numcodecs.FixedScaleOffset",
)
register_codec(
    "fixedscaleoffset",
    FixedScaleOffset,
    qualname="zarr.codecs.numcodecs.FixedScaleOffset",
)

register_codec("numcodecs.fletcher32", Fletcher32, qualname="zarr.codecs.numcodecs.Fletcher32")
register_codec("fletcher32", Fletcher32, qualname="zarr.codecs.numcodecs.Fletcher32")

register_codec(
    "numcodecs.jenkins_lookup3", JenkinsLookup3, qualname="zarr.codecs.numcodecs.JenkinsLookup3"
)
register_codec("jenkins_lookup3", JenkinsLookup3, qualname="zarr.codecs.numcodecs.JenkinsLookup3")

register_codec("pcodec", PCodec, qualname="zarr.codecs.numcodecs.PCodec")
register_codec("numcodecs.pcodec", PCodec, qualname="zarr.codecs.numcodecs.PCodec")

register_codec("numcodecs.packbits", PackBits, qualname="zarr.codecs.numcodecs.PackBits")
register_codec("packbits", PackBits, qualname="zarr.codecs.numcodecs.PackBits")

register_codec("numcodecs.quantize", Quantize, qualname="zarr.codecs.numcodecs.Quantize")
register_codec("quantize", Quantize, qualname="zarr.codecs.numcodecs.Quantize")

register_codec("numcodecs.shuffle", Shuffle, qualname="zarr.codecs.numcodecs.Shuffle")
register_codec("shuffle", Shuffle, qualname="zarr.codecs.numcodecs.Shuffle")

register_codec("numcodecs.zlib", Zlib, qualname="zarr.codecs.numcodecs.Zlib")
register_codec("zlib", Zlib, qualname="zarr.codecs.numcodecs.Zlib")
