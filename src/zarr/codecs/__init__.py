from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from zarr.codecs.blosc import BloscCname, BloscCodec, BloscShuffle
from zarr.codecs.bytes import BytesCodec, Endian
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.pipeline import BatchedCodecPipeline
from zarr.codecs.sharding import ShardingCodec, ShardingCodecIndexLocation
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.vlen_utf8 import VLenBytesCodec, VLenUTF8Codec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.metadata.v3 import DataType

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
    "VLenBytesCodec",
    "VLenUTF8Codec",
    "ZstdCodec",
]


def _get_default_array_bytes_codec(
    np_dtype: np.dtype[Any],
) -> BytesCodec | VLenUTF8Codec | VLenBytesCodec:
    dtype = DataType.from_numpy(np_dtype)
    if dtype == DataType.string:
        return VLenUTF8Codec()
    elif dtype == DataType.bytes:
        return VLenBytesCodec()
    else:
        return BytesCodec()
