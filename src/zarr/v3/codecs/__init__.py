from __future__ import annotations

from functools import reduce
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from warnings import warn

import numpy as np
from attr import frozen

from zarr.v3.abc.codec import Codec, ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.v3.common import BytesLike
from zarr.v3.metadata import CodecMetadata, ShardingCodecIndexLocation
from zarr.v3.codecs.registry import get_codec_class

if TYPE_CHECKING:
    from zarr.v3.metadata import CoreArrayMetadata
    from zarr.v3.codecs.sharding import ShardingCodecMetadata
    from zarr.v3.codecs.blosc import BloscCodecMetadata
    from zarr.v3.codecs.bytes import BytesCodecMetadata
    from zarr.v3.codecs.transpose import TransposeCodecMetadata
    from zarr.v3.codecs.gzip import GzipCodecMetadata
    from zarr.v3.codecs.zstd import ZstdCodecMetadata
    from zarr.v3.codecs.crc32c_ import Crc32cCodecMetadata


@frozen
class CodecPipeline:
    codecs: List[Codec]

    @classmethod
    def from_metadata(
        cls,
        codecs_metadata: Iterable[CodecMetadata],
        array_metadata: CoreArrayMetadata,
    ) -> CodecPipeline:
        out: List[Codec] = []
        for codec_metadata in codecs_metadata or []:
            codec_cls = get_codec_class(codec_metadata.name)
            codec = codec_cls.from_metadata(codec_metadata, array_metadata)
            out.append(codec)
            array_metadata = codec.resolve_metadata()
        CodecPipeline._validate_codecs(out, array_metadata)
        return cls(out)

    @staticmethod
    def _validate_codecs(codecs: List[Codec], array_metadata: CoreArrayMetadata) -> None:
        from zarr.v3.codecs.sharding import ShardingCodec

        assert any(
            isinstance(codec, ArrayBytesCodec) for codec in codecs
        ), "Exactly one array-to-bytes codec is required."

        prev_codec: Optional[Codec] = None
        for codec in codecs:
            if prev_codec is not None:
                assert not isinstance(codec, ArrayBytesCodec) or not isinstance(
                    prev_codec, ArrayBytesCodec
                ), (
                    f"ArrayBytesCodec '{type(codec)}' cannot follow after "
                    + f"ArrayBytesCodec '{type(prev_codec)}' because exactly "
                    + "1 ArrayBytesCodec is allowed."
                )
                assert not isinstance(codec, ArrayBytesCodec) or not isinstance(
                    prev_codec, BytesBytesCodec
                ), (
                    f"ArrayBytesCodec '{type(codec)}' cannot follow after "
                    + f"BytesBytesCodec '{type(prev_codec)}'."
                )
                assert not isinstance(codec, ArrayArrayCodec) or not isinstance(
                    prev_codec, ArrayBytesCodec
                ), (
                    f"ArrayArrayCodec '{type(codec)}' cannot follow after "
                    + f"ArrayBytesCodec '{type(prev_codec)}'."
                )
                assert not isinstance(codec, ArrayArrayCodec) or not isinstance(
                    prev_codec, BytesBytesCodec
                ), (
                    f"ArrayArrayCodec '{type(codec)}' cannot follow after "
                    + f"BytesBytesCodec '{type(prev_codec)}'."
                )

            if isinstance(codec, ShardingCodec):
                assert len(codec.configuration.chunk_shape) == len(array_metadata.shape), (
                    "The shard's `chunk_shape` and array's `shape` need to have the "
                    + "same number of dimensions."
                )
                assert all(
                    s % c == 0
                    for s, c in zip(
                        array_metadata.chunk_shape,
                        codec.configuration.chunk_shape,
                    )
                ), (
                    "The array's `chunk_shape` needs to be divisible by the "
                    + "shard's inner `chunk_shape`."
                )
            prev_codec = codec

        if any(isinstance(codec, ShardingCodec) for codec in codecs) and len(codecs) > 1:
            warn(
                "Combining a `sharding_indexed` codec disables partial reads and "
                + "writes, which may lead to inefficient performance."
            )

    def _array_array_codecs(self) -> List[ArrayArrayCodec]:
        return [codec for codec in self.codecs if isinstance(codec, ArrayArrayCodec)]

    def _array_bytes_codec(self) -> ArrayBytesCodec:
        return next(codec for codec in self.codecs if isinstance(codec, ArrayBytesCodec))

    def _bytes_bytes_codecs(self) -> List[BytesBytesCodec]:
        return [codec for codec in self.codecs if isinstance(codec, BytesBytesCodec)]

    async def decode(self, chunk_bytes: BytesLike) -> np.ndarray:
        for bb_codec in self._bytes_bytes_codecs()[::-1]:
            chunk_bytes = await bb_codec.decode(chunk_bytes)

        chunk_array = await self._array_bytes_codec().decode(chunk_bytes)

        for aa_codec in self._array_array_codecs()[::-1]:
            chunk_array = await aa_codec.decode(chunk_array)

        return chunk_array

    async def encode(self, chunk_array: np.ndarray) -> Optional[BytesLike]:
        for aa_codec in self._array_array_codecs():
            chunk_array_maybe = await aa_codec.encode(chunk_array)
            if chunk_array_maybe is None:
                return None
            chunk_array = chunk_array_maybe

        chunk_bytes_maybe = await self._array_bytes_codec().encode(chunk_array)
        if chunk_bytes_maybe is None:
            return None
        chunk_bytes = chunk_bytes_maybe

        for bb_codec in self._bytes_bytes_codecs():
            chunk_bytes_maybe = await bb_codec.encode(chunk_bytes)
            if chunk_bytes_maybe is None:
                return None
            chunk_bytes = chunk_bytes_maybe

        return chunk_bytes

    def compute_encoded_size(self, byte_length: int) -> int:
        return reduce(lambda acc, codec: codec.compute_encoded_size(acc), self.codecs, byte_length)


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
    codecs: Optional[List[CodecMetadata]] = None,
    index_codecs: Optional[List[CodecMetadata]] = None,
    index_location: ShardingCodecIndexLocation = ShardingCodecIndexLocation.end,
) -> "ShardingCodecMetadata":
    from zarr.v3.codecs.sharding import ShardingCodecMetadata, ShardingCodecConfigurationMetadata

    codecs = codecs or [bytes_codec()]
    index_codecs = index_codecs or [bytes_codec(), crc32c_codec()]
    return ShardingCodecMetadata(
        configuration=ShardingCodecConfigurationMetadata(
            chunk_shape, codecs, index_codecs, index_location
        )
    )
