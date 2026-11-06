from __future__ import annotations

from zarr.codecs.blosc import BloscCname, BloscCodec, BloscShuffle
from zarr.codecs.bytes import BytesCodec, Endian
from zarr.codecs.cast_value import CastValue
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.gzip import GzipCodec

# Keep the numcodecs wrappers importable from this module for compatibility.
from zarr.codecs.numcodecs import (  # noqa: F401
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
)
from zarr.codecs.scale_offset import ScaleOffset
from zarr.codecs.sharding import ShardingCodec, ShardingCodecIndexLocation, SubchunkWriteOrder
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.vlen_utf8 import VLenBytesCodec, VLenUTF8Codec
from zarr.codecs.zstd import ZstdCodec

__all__ = [
    "BloscCname",
    "BloscCodec",
    "BloscShuffle",
    "BytesCodec",
    "CastValue",
    "Crc32cCodec",
    "Endian",
    "GzipCodec",
    "ScaleOffset",
    "ShardingCodec",
    "ShardingCodecIndexLocation",
    "SubchunkWriteOrder",
    "TransposeCodec",
    "VLenBytesCodec",
    "VLenUTF8Codec",
    "ZstdCodec",
]
