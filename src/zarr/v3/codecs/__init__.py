from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from zarr.v3.common import NamedConfig

if TYPE_CHECKING:
    from zarr.v3.codecs.sharding import ShardingCodecMetadata
    from zarr.v3.codecs.blosc import BloscCodecMetadata
    from zarr.v3.codecs.bytes import BytesCodecMetadata
    from zarr.v3.codecs.transpose import TransposeCodecMetadata
    from zarr.v3.codecs.gzip import GzipCodecMetadata
    from zarr.v3.codecs.zstd import ZstdCodecMetadata
    from zarr.v3.codecs.crc32c_ import Crc32cCodecMetadata


ShardingCodecIndexLocation = Literal["start", "end"]


def blosc_codec(
    typesize: int,
    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"] = "zstd",
    clevel: int = 5,
    shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] = "noshuffle",
    blocksize: int = 0,
) -> "BloscCodecMetadata":
    from zarr.v3.codecs.blosc import BloscCodecMetadata, BloscCodecConfigurationMetadata

    return BloscCodecMetadata(
        configuration=BloscCodecConfigurationMetadata(
            cname=cname,
            clevel=clevel,
            shuffle=shuffle,
            blocksize=blocksize,
            typesize=typesize,
        )
    )


def bytes_codec(endian: Optional[Literal["big", "little"]] = "little") -> "BytesCodecMetadata":
    from zarr.v3.codecs.bytes import BytesCodecMetadata, BytesCodecConfigurationMetadata

    return BytesCodecMetadata(configuration=BytesCodecConfigurationMetadata(endian))


def transpose_codec(
    order: Union[Tuple[int, ...], Literal["C", "F"]], ndim: Optional[int] = None
) -> "TransposeCodecMetadata":
    from zarr.v3.codecs.transpose import TransposeCodecMetadata, TransposeCodecConfigurationMetadata

    if order == "C" or order == "F":
        assert (
            isinstance(ndim, int) and ndim > 0
        ), 'When using "C" or "F" the `ndim` argument needs to be provided.'
        if order == "C":
            order = tuple(range(ndim))
        if order == "F":
            order = tuple(ndim - i - 1 for i in range(ndim))

    return TransposeCodecMetadata(configuration=TransposeCodecConfigurationMetadata(order))


def gzip_codec(level: int = 5) -> "GzipCodecMetadata":
    from zarr.v3.codecs.gzip import GzipCodecMetadata, GzipCodecConfigurationMetadata

    return GzipCodecMetadata(configuration=GzipCodecConfigurationMetadata(level))


def zstd_codec(level: int = 0, checksum: bool = False) -> "ZstdCodecMetadata":
    from zarr.v3.codecs.zstd import ZstdCodecMetadata, ZstdCodecConfigurationMetadata

    return ZstdCodecMetadata(configuration=ZstdCodecConfigurationMetadata(level, checksum))


def crc32c_codec() -> "Crc32cCodecMetadata":
    from zarr.v3.codecs.crc32c_ import Crc32cCodecMetadata

    return Crc32cCodecMetadata()


def sharding_codec(
    chunk_shape: Tuple[int, ...],
    codecs: Optional[List[NamedConfig]] = None,
    index_codecs: Optional[List[NamedConfig]] = None,
    index_location: ShardingCodecIndexLocation = "end",
) -> "ShardingCodecMetadata":
    from zarr.v3.codecs.sharding import ShardingCodecMetadata, ShardingCodecConfigurationMetadata

    codecs = codecs or [bytes_codec()]
    index_codecs = index_codecs or [bytes_codec(), crc32c_codec()]
    return ShardingCodecMetadata(
        configuration=ShardingCodecConfigurationMetadata(
            chunk_shape, codecs, index_codecs, index_location
        )
    )
