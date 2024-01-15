from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from warnings import warn
from attr import frozen

import numpy as np

from zarr.v3.abc.codec import Codec, ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.v3.common import BytesLike
from zarr.v3.metadata import CodecMetadata, ShardingCodecIndexLocation, RuntimeConfiguration

if TYPE_CHECKING:
    from zarr.v3.metadata import ArrayMetadata, ChunkMetadata
    from zarr.v3.codecs.sharding import ShardingCodecMetadata
    from zarr.v3.codecs.blosc import BloscCodecMetadata
    from zarr.v3.codecs.bytes import BytesCodecMetadata
    from zarr.v3.codecs.transpose import TransposeCodecMetadata
    from zarr.v3.codecs.gzip import GzipCodecMetadata
    from zarr.v3.codecs.zstd import ZstdCodecMetadata
    from zarr.v3.codecs.crc32c_ import Crc32cCodecMetadata


def _find_array_bytes_codec(
    codecs: Iterable[Tuple[Codec, ChunkMetadata]]
) -> Tuple[ArrayBytesCodec, ChunkMetadata]:
    for codec, chunk_metadata in codecs:
        if isinstance(codec, ArrayBytesCodec):
            return (codec, chunk_metadata)
    raise KeyError


@frozen
class CodecPipeline:
    codecs: List[Codec]

    def validate(self, array_metadata: ArrayMetadata) -> None:
        from zarr.v3.codecs.sharding import ShardingCodec

        assert any(
            isinstance(codec, ArrayBytesCodec) for codec in self.codecs
        ), "Exactly one array-to-bytes codec is required."

        prev_codec: Optional[Codec] = None
        for codec in self.codecs:
            codec.validate(array_metadata)
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
            prev_codec = codec

        if any(isinstance(codec, ShardingCodec) for codec in self.codecs) and len(self.codecs) > 1:
            warn(
                "Combining a `sharding_indexed` codec disables partial reads and "
                + "writes, which may lead to inefficient performance."
            )

    def _codecs_with_resolved_metadata(
        self, chunk_metadata: ChunkMetadata
    ) -> Iterator[Tuple[Codec, ChunkMetadata]]:
        for codec in self.codecs:
            yield (codec, chunk_metadata)
            chunk_metadata = codec.resolve_metadata(chunk_metadata)

    async def decode(
        self,
        chunk_bytes: BytesLike,
        chunk_metadata: ChunkMetadata,
        runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        codecs = list(self._codecs_with_resolved_metadata(chunk_metadata))[::-1]

        for bb_codec, chunk_metadata in codecs:
            if isinstance(bb_codec, BytesBytesCodec):
                chunk_bytes = await bb_codec.decode(
                    chunk_bytes, chunk_metadata, runtime_configuration
                )

        ab_codec, chunk_metadata = _find_array_bytes_codec(codecs)
        chunk_array = await ab_codec.decode(chunk_bytes, chunk_metadata, runtime_configuration)

        for aa_codec, chunk_metadata in codecs:
            if isinstance(aa_codec, ArrayArrayCodec):
                chunk_array = await aa_codec.decode(
                    chunk_array, chunk_metadata, runtime_configuration
                )

        return chunk_array

    async def encode(
        self,
        chunk_array: np.ndarray,
        chunk_metadata: ChunkMetadata,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        codecs = list(self._codecs_with_resolved_metadata(chunk_metadata))

        for aa_codec, chunk_metadata in codecs:
            if isinstance(aa_codec, ArrayArrayCodec):
                chunk_array_maybe = await aa_codec.encode(
                    chunk_array, chunk_metadata, runtime_configuration
                )
                if chunk_array_maybe is None:
                    return None
                chunk_array = chunk_array_maybe

        ab_codec, chunk_metadata = _find_array_bytes_codec(codecs)
        chunk_bytes_maybe = await ab_codec.encode(
            chunk_array, chunk_metadata, runtime_configuration
        )
        if chunk_bytes_maybe is None:
            return None
        chunk_bytes = chunk_bytes_maybe

        for bb_codec, chunk_metadata in codecs:
            if isinstance(bb_codec, BytesBytesCodec):
                chunk_bytes_maybe = await bb_codec.encode(
                    chunk_bytes, chunk_metadata, runtime_configuration
                )
                if chunk_bytes_maybe is None:
                    return None
                chunk_bytes = chunk_bytes_maybe

        return chunk_bytes

    def compute_encoded_size(self, byte_length: int, chunk_metadata: ChunkMetadata) -> int:
        for codec in self.codecs:
            byte_length = codec.compute_encoded_size(byte_length, chunk_metadata)
            chunk_metadata = codec.resolve_metadata(chunk_metadata)
        return byte_length


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
    codecs: Optional[Iterable[CodecMetadata]] = None,
    index_codecs: Optional[Iterable[CodecMetadata]] = None,
    index_location: ShardingCodecIndexLocation = ShardingCodecIndexLocation.end,
) -> "ShardingCodecMetadata":
    from zarr.v3.codecs.sharding import ShardingCodecMetadata, ShardingCodecConfigurationMetadata

    codecs = tuple(codecs) if codecs is not None else (bytes_codec(),)
    index_codecs = (
        tuple(index_codecs) if index_codecs is not None else (bytes_codec(), crc32c_codec())
    )
    return ShardingCodecMetadata(
        configuration=ShardingCodecConfigurationMetadata(
            chunk_shape, codecs, index_codecs, index_location
        )
    )
