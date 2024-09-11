from __future__ import annotations

from zarr.codecs._blosc import BloscCname, BloscCodec, BloscShuffle
from zarr.codecs._bytes import BytesCodec, Endian
from zarr.codecs._crc32c_ import Crc32cCodec
from zarr.codecs._gzip import GzipCodec
from zarr.codecs._pipeline import BatchedCodecPipeline
from zarr.codecs._sharding import ShardingCodec, ShardingCodecIndexLocation
from zarr.codecs._transpose import TransposeCodec
from zarr.codecs._zstd import ZstdCodec
from zarr.registry import register_codec, register_pipeline

__all__ = [
    "BatchedCodecPipeline",
    "BloscCname",
    "BloscCodec",
    "BloscShuffle",
    "BytesCodec",
    "Crc32cCodec",
    "Endian",
    "GzipCodec",
    "ShardingCodec",
    "ShardingCodecIndexLocation",
    "TransposeCodec",
    "ZstdCodec",
]


register_codec("blosc", BloscCodec)
register_codec("bytes", BytesCodec)
# compatibility with earlier versions of ZEP1
register_codec("endian", BytesCodec)
register_codec("crc32c", Crc32cCodec)
register_codec("gzip", GzipCodec)
register_pipeline(BatchedCodecPipeline)
register_codec("sharding_indexed", ShardingCodec)
register_codec("transpose", TransposeCodec)
register_codec("zstd", ZstdCodec)
