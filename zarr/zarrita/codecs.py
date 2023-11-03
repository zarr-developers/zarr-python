from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Tuple, Union
from warnings import warn

import numcodecs
import numpy as np
from attr import asdict, evolve, frozen
from crc32c import crc32c
from numcodecs.blosc import Blosc
from numcodecs.gzip import GZip
from zstandard import ZstdCompressor, ZstdDecompressor

from zarrita.common import BytesLike, to_thread
from zarrita.metadata import (
    BloscCodecConfigurationMetadata,
    BloscCodecMetadata,
    BytesCodecConfigurationMetadata,
    BytesCodecMetadata,
    CodecMetadata,
    Crc32cCodecMetadata,
    GzipCodecConfigurationMetadata,
    GzipCodecMetadata,
    ShardingCodecConfigurationMetadata,
    ShardingCodecMetadata,
    TransposeCodecConfigurationMetadata,
    TransposeCodecMetadata,
    ZstdCodecConfigurationMetadata,
    ZstdCodecMetadata,
)

if TYPE_CHECKING:
    from zarrita.metadata import CoreArrayMetadata

# See https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
numcodecs.blosc.use_threads = False


class Codec(ABC):
    supports_partial_decode: bool
    supports_partial_encode: bool
    is_fixed_size: bool
    array_metadata: CoreArrayMetadata

    @abstractmethod
    def compute_encoded_size(self, input_byte_length: int) -> int:
        pass

    def resolve_metadata(self) -> CoreArrayMetadata:
        return self.array_metadata


class ArrayArrayCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
    ) -> Optional[np.ndarray]:
        pass


class ArrayBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: BytesLike,
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
    ) -> Optional[BytesLike]:
        pass


class BytesBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: BytesLike,
    ) -> BytesLike:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: BytesLike,
    ) -> Optional[BytesLike]:
        pass


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
            if codec_metadata.name == "endian":
                codec_metadata = evolve(codec_metadata, name="bytes")  # type: ignore

            codec: Codec
            if codec_metadata.name == "blosc":
                codec = BloscCodec.from_metadata(codec_metadata, array_metadata)
            elif codec_metadata.name == "gzip":
                codec = GzipCodec.from_metadata(codec_metadata, array_metadata)
            elif codec_metadata.name == "zstd":
                codec = ZstdCodec.from_metadata(codec_metadata, array_metadata)
            elif codec_metadata.name == "transpose":
                codec = TransposeCodec.from_metadata(codec_metadata, array_metadata)
            elif codec_metadata.name == "bytes":
                codec = BytesCodec.from_metadata(codec_metadata, array_metadata)
            elif codec_metadata.name == "crc32c":
                codec = Crc32cCodec.from_metadata(codec_metadata, array_metadata)
            elif codec_metadata.name == "sharding_indexed":
                from zarrita.sharding import ShardingCodec

                codec = ShardingCodec.from_metadata(codec_metadata, array_metadata)
            else:
                raise RuntimeError(f"Unsupported codec: {codec_metadata}")

            out.append(codec)
            array_metadata = codec.resolve_metadata()
        CodecPipeline._validate_codecs(out, array_metadata)
        return cls(out)

    @staticmethod
    def _validate_codecs(
        codecs: List[Codec], array_metadata: CoreArrayMetadata
    ) -> None:
        from zarrita.sharding import ShardingCodec

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
                assert len(codec.configuration.chunk_shape) == len(
                    array_metadata.shape
                ), (
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

        if (
            any(isinstance(codec, ShardingCodec) for codec in codecs)
            and len(codecs) > 1
        ):
            warn(
                "Combining a `sharding_indexed` codec disables partial reads and "
                + "writes, which may lead to inefficient performance."
            )

    def _array_array_codecs(self) -> List[ArrayArrayCodec]:
        return [codec for codec in self.codecs if isinstance(codec, ArrayArrayCodec)]

    def _array_bytes_codec(self) -> ArrayBytesCodec:
        return next(
            codec for codec in self.codecs if isinstance(codec, ArrayBytesCodec)
        )

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
        return reduce(
            lambda acc, codec: codec.compute_encoded_size(acc), self.codecs, byte_length
        )


@frozen
class BloscCodec(BytesBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: BloscCodecConfigurationMetadata
    blosc_codec: Blosc
    is_fixed_size = False

    @classmethod
    def from_metadata(
        cls, codec_metadata: BloscCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> BloscCodec:
        configuration = codec_metadata.configuration
        if configuration.typesize == 0:
            configuration = evolve(
                configuration, typesize=array_metadata.data_type.byte_count
            )
        config_dict = asdict(codec_metadata.configuration)
        config_dict.pop("typesize", None)
        map_shuffle_str_to_int = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
        config_dict["shuffle"] = map_shuffle_str_to_int[config_dict["shuffle"]]
        return cls(
            array_metadata=array_metadata,
            configuration=configuration,
            blosc_codec=Blosc.from_config(config_dict),
        )

    async def decode(
        self,
        chunk_bytes: bytes,
    ) -> BytesLike:
        return await to_thread(self.blosc_codec.decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
    ) -> Optional[BytesLike]:
        chunk_array = np.frombuffer(chunk_bytes, dtype=self.array_metadata.dtype)
        return await to_thread(self.blosc_codec.encode, chunk_array)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


@frozen
class BytesCodec(ArrayBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: BytesCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: BytesCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> BytesCodec:
        assert (
            array_metadata.dtype.itemsize == 1
            or codec_metadata.configuration.endian is not None
        ), "The `endian` configuration needs to be specified for multi-byte data types."
        return cls(
            array_metadata=array_metadata,
            configuration=codec_metadata.configuration,
        )

    def _get_byteorder(self, array: np.ndarray) -> Literal["big", "little"]:
        if array.dtype.byteorder == "<":
            return "little"
        elif array.dtype.byteorder == ">":
            return "big"
        else:
            import sys

            return sys.byteorder

    async def decode(
        self,
        chunk_bytes: BytesLike,
    ) -> np.ndarray:
        if self.array_metadata.dtype.itemsize > 0:
            if self.configuration.endian == "little":
                prefix = "<"
            else:
                prefix = ">"
            dtype = np.dtype(
                f"{prefix}{self.array_metadata.data_type.to_numpy_shortname()}"
            )
        else:
            dtype = np.dtype(f"|{self.array_metadata.data_type.to_numpy_shortname()}")
        chunk_array = np.frombuffer(chunk_bytes, dtype)

        # ensure correct chunk shape
        if chunk_array.shape != self.array_metadata.chunk_shape:
            chunk_array = chunk_array.reshape(
                self.array_metadata.chunk_shape,
            )
        return chunk_array

    async def encode(
        self,
        chunk_array: np.ndarray,
    ) -> Optional[BytesLike]:
        if chunk_array.dtype.itemsize > 1:
            byteorder = self._get_byteorder(chunk_array)
            if self.configuration.endian != byteorder:
                new_dtype = chunk_array.dtype.newbyteorder(self.configuration.endian)
                chunk_array = chunk_array.astype(new_dtype)
        return chunk_array.tobytes()

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length


@frozen
class TransposeCodec(ArrayArrayCodec):
    array_metadata: CoreArrayMetadata
    order: Tuple[int, ...]
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: TransposeCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> TransposeCodec:
        configuration = codec_metadata.configuration
        if configuration.order == "F":
            order = tuple(
                array_metadata.ndim - x - 1 for x in range(array_metadata.ndim)
            )

        elif configuration.order == "C":
            order = tuple(range(array_metadata.ndim))

        else:
            assert len(configuration.order) == array_metadata.ndim, (
                "The `order` tuple needs have as many entries as "
                + f"there are dimensions in the array. Got: {configuration.order}"
            )
            assert len(configuration.order) == len(set(configuration.order)), (
                "There must not be duplicates in the `order` tuple. "
                + f"Got: {configuration.order}"
            )
            assert all(0 <= x < array_metadata.ndim for x in configuration.order), (
                "All entries in the `order` tuple must be between 0 and "
                + f"the number of dimensions in the array. Got: {configuration.order}"
            )
            order = tuple(configuration.order)

        return cls(
            array_metadata=array_metadata,
            order=order,
        )

    def resolve_metadata(self) -> CoreArrayMetadata:
        from zarrita.metadata import CoreArrayMetadata

        return CoreArrayMetadata(
            shape=tuple(
                self.array_metadata.shape[self.order[i]]
                for i in range(self.array_metadata.ndim)
            ),
            chunk_shape=tuple(
                self.array_metadata.chunk_shape[self.order[i]]
                for i in range(self.array_metadata.ndim)
            ),
            data_type=self.array_metadata.data_type,
            fill_value=self.array_metadata.fill_value,
            runtime_configuration=self.array_metadata.runtime_configuration,
        )

    async def decode(
        self,
        chunk_array: np.ndarray,
    ) -> np.ndarray:
        inverse_order = [0 for _ in range(self.array_metadata.ndim)]
        for x, i in enumerate(self.order):
            inverse_order[x] = i
        chunk_array = chunk_array.transpose(inverse_order)
        return chunk_array

    async def encode(
        self,
        chunk_array: np.ndarray,
    ) -> Optional[np.ndarray]:
        chunk_array = chunk_array.transpose(self.order)
        return chunk_array

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length


@frozen
class GzipCodec(BytesBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: GzipCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: GzipCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> GzipCodec:
        return cls(
            array_metadata=array_metadata,
            configuration=codec_metadata.configuration,
        )

    async def decode(
        self,
        chunk_bytes: bytes,
    ) -> BytesLike:
        return await to_thread(GZip(self.configuration.level).decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
    ) -> Optional[BytesLike]:
        return await to_thread(GZip(self.configuration.level).encode, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


@frozen
class ZstdCodec(BytesBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: ZstdCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: ZstdCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> ZstdCodec:
        return cls(
            array_metadata=array_metadata,
            configuration=codec_metadata.configuration,
        )

    def _compress(self, data: bytes) -> bytes:
        ctx = ZstdCompressor(
            level=self.configuration.level, write_checksum=self.configuration.checksum
        )
        return ctx.compress(data)

    def _decompress(self, data: bytes) -> bytes:
        ctx = ZstdDecompressor()
        return ctx.decompress(data)

    async def decode(
        self,
        chunk_bytes: bytes,
    ) -> BytesLike:
        return await to_thread(self._decompress, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
    ) -> Optional[BytesLike]:
        return await to_thread(self._compress, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


@frozen
class Crc32cCodec(BytesBytesCodec):
    array_metadata: CoreArrayMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: Crc32cCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> Crc32cCodec:
        return cls(array_metadata=array_metadata)

    async def decode(
        self,
        chunk_bytes: bytes,
    ) -> BytesLike:
        crc32_bytes = chunk_bytes[-4:]
        inner_bytes = chunk_bytes[:-4]

        assert np.uint32(crc32c(inner_bytes)).tobytes() == bytes(crc32_bytes)
        return inner_bytes

    async def encode(
        self,
        chunk_bytes: bytes,
    ) -> Optional[BytesLike]:
        return chunk_bytes + np.uint32(crc32c(chunk_bytes)).tobytes()

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length + 4


def blosc_codec(
    typesize: int,
    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"] = "zstd",
    clevel: int = 5,
    shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] = "noshuffle",
    blocksize: int = 0,
) -> BloscCodecMetadata:
    return BloscCodecMetadata(
        configuration=BloscCodecConfigurationMetadata(
            cname=cname,
            clevel=clevel,
            shuffle=shuffle,
            blocksize=blocksize,
            typesize=typesize,
        )
    )


def bytes_codec(
    endian: Optional[Literal["big", "little"]] = "little"
) -> BytesCodecMetadata:
    return BytesCodecMetadata(configuration=BytesCodecConfigurationMetadata(endian))


def transpose_codec(
    order: Union[Tuple[int, ...], Literal["C", "F"]]
) -> TransposeCodecMetadata:
    return TransposeCodecMetadata(
        configuration=TransposeCodecConfigurationMetadata(order)
    )


def gzip_codec(level: int = 5) -> GzipCodecMetadata:
    return GzipCodecMetadata(configuration=GzipCodecConfigurationMetadata(level))


def zstd_codec(level: int = 0, checksum: bool = False) -> ZstdCodecMetadata:
    return ZstdCodecMetadata(
        configuration=ZstdCodecConfigurationMetadata(level, checksum)
    )


def crc32c_codec() -> Crc32cCodecMetadata:
    return Crc32cCodecMetadata()


def sharding_codec(
    chunk_shape: Tuple[int, ...],
    codecs: Optional[List[CodecMetadata]] = None,
    index_codecs: Optional[List[CodecMetadata]] = None,
) -> ShardingCodecMetadata:
    codecs = codecs or [bytes_codec()]
    index_codecs = index_codecs or [bytes_codec(), crc32c_codec()]
    return ShardingCodecMetadata(
        configuration=ShardingCodecConfigurationMetadata(
            chunk_shape, codecs, index_codecs
        )
    )
