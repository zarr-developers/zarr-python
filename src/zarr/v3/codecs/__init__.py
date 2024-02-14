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

from zarr.v3.abc.codec import (
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    Codec,
    ArrayArrayCodec,
    ArrayBytesCodec,
    BytesBytesCodec,
)
from zarr.v3.common import BytesLike, SliceSelection
from zarr.v3.metadata import CodecMetadata, ShardingCodecIndexLocation, RuntimeConfiguration
from zarr.v3.store import StorePath

if TYPE_CHECKING:
    from zarr.v3.metadata import ArrayMetadata, ArraySpec
    from zarr.v3.codecs.sharding import ShardingCodecMetadata
    from zarr.v3.codecs.blosc import BloscCodecMetadata
    from zarr.v3.codecs.bytes import BytesCodecMetadata
    from zarr.v3.codecs.transpose import TransposeCodecMetadata
    from zarr.v3.codecs.gzip import GzipCodecMetadata
    from zarr.v3.codecs.zstd import ZstdCodecMetadata
    from zarr.v3.codecs.crc32c_ import Crc32cCodecMetadata


def _find_array_bytes_codec(
    codecs: Iterable[Tuple[Codec, ArraySpec]],
) -> Tuple[ArrayBytesCodec, ArraySpec]:
    for codec, array_spec in codecs:
        if isinstance(codec, ArrayBytesCodec):
            return (codec, array_spec)
    raise KeyError


@frozen
class CodecPipeline:
    array_array_codecs: List[ArrayArrayCodec]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: List[BytesBytesCodec]

    @classmethod
    def create(cls, codecs: List[Codec]) -> CodecPipeline:
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
            prev_codec = codec

        if any(isinstance(codec, ShardingCodec) for codec in codecs) and len(codecs) > 1:
            warn(
                "Combining a `sharding_indexed` codec disables partial reads and "
                + "writes, which may lead to inefficient performance."
            )

        return CodecPipeline(
            array_array_codecs=[codec for codec in codecs if isinstance(codec, ArrayArrayCodec)],
            array_bytes_codec=[codec for codec in codecs if isinstance(codec, ArrayBytesCodec)][0],
            bytes_bytes_codecs=[codec for codec in codecs if isinstance(codec, BytesBytesCodec)],
        )

    @property
    def supports_partial_decode(self) -> bool:
        return (len(self.array_array_codecs) + len(self.bytes_bytes_codecs)) == 0 and isinstance(
            self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin
        )

    @property
    def supports_partial_encode(self) -> bool:
        return (len(self.array_array_codecs) + len(self.bytes_bytes_codecs)) == 0 and isinstance(
            self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin
        )

    def __iter__(self) -> Iterator[Codec]:
        for aa_codec in self.array_array_codecs:
            yield aa_codec

        yield self.array_bytes_codec

        for bb_codec in self.bytes_bytes_codecs:
            yield bb_codec

    def validate(self, array_metadata: ArrayMetadata) -> None:
        for codec in self:
            codec.validate(array_metadata)

    def _codecs_with_resolved_metadata(
        self, array_spec: ArraySpec
    ) -> Tuple[
        List[Tuple[ArrayArrayCodec, ArraySpec]],
        Tuple[ArrayBytesCodec, ArraySpec],
        List[Tuple[BytesBytesCodec, ArraySpec]],
    ]:
        aa_codecs_with_spec: List[Tuple[ArrayArrayCodec, ArraySpec]] = []
        for aa_codec in self.array_array_codecs:
            aa_codecs_with_spec.append((aa_codec, array_spec))
            array_spec = aa_codec.resolve_metadata(array_spec)

        ab_codec_with_spec = (self.array_bytes_codec, array_spec)
        array_spec = self.array_bytes_codec.resolve_metadata(array_spec)

        bb_codecs_with_spec: List[Tuple[BytesBytesCodec, ArraySpec]] = []
        for bb_codec in self.bytes_bytes_codecs:
            bb_codecs_with_spec.append((bb_codec, array_spec))
            array_spec = bb_codec.resolve_metadata(array_spec)

        return (aa_codecs_with_spec, ab_codec_with_spec, bb_codecs_with_spec)

    async def decode(
        self,
        chunk_bytes: BytesLike,
        array_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        (
            aa_codecs_with_spec,
            ab_codec_with_spec,
            bb_codecs_with_spec,
        ) = self._codecs_with_resolved_metadata(array_spec)

        for bb_codec, array_spec in bb_codecs_with_spec[::-1]:
            chunk_bytes = await bb_codec.decode(chunk_bytes, array_spec, runtime_configuration)

        ab_codec, array_spec = ab_codec_with_spec
        chunk_array = await ab_codec.decode(chunk_bytes, array_spec, runtime_configuration)

        for aa_codec, array_spec in aa_codecs_with_spec[::-1]:
            chunk_array = await aa_codec.decode(chunk_array, array_spec, runtime_configuration)

        return chunk_array

    async def decode_partial(
        self,
        store_path: StorePath,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[np.ndarray]:
        assert self.supports_partial_decode
        assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin)
        return await self.array_bytes_codec.decode_partial(
            store_path, selection, chunk_spec, runtime_configuration
        )

    async def encode(
        self,
        chunk_array: np.ndarray,
        array_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        (
            aa_codecs_with_spec,
            ab_codec_with_spec,
            bb_codecs_with_spec,
        ) = self._codecs_with_resolved_metadata(array_spec)

        for aa_codec, array_spec in aa_codecs_with_spec:
            chunk_array_maybe = await aa_codec.encode(
                chunk_array, array_spec, runtime_configuration
            )
            if chunk_array_maybe is None:
                return None
            chunk_array = chunk_array_maybe

        ab_codec, array_spec = ab_codec_with_spec
        chunk_bytes_maybe = await ab_codec.encode(chunk_array, array_spec, runtime_configuration)
        if chunk_bytes_maybe is None:
            return None
        chunk_bytes = chunk_bytes_maybe

        for bb_codec, array_spec in bb_codecs_with_spec:
            chunk_bytes_maybe = await bb_codec.encode(
                chunk_bytes, array_spec, runtime_configuration
            )
            if chunk_bytes_maybe is None:
                return None
            chunk_bytes = chunk_bytes_maybe

        return chunk_bytes

    async def encode_partial(
        self,
        store_path: StorePath,
        chunk_array: np.ndarray,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        assert self.supports_partial_encode
        assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin)
        await self.array_bytes_codec.encode_partial(
            store_path, chunk_array, selection, chunk_spec, runtime_configuration
        )

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        for codec in self:
            byte_length = codec.compute_encoded_size(byte_length, array_spec)
            array_spec = codec.resolve_metadata(array_spec)
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
