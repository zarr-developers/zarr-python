from __future__ import annotations
from zarr.v3.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec, Codec
from zarr.v3.codecs.registry import get_codec_class
from zarr.v3.common import BytesLike, NamedConfig, RuntimeConfiguration
from zarr.v3.metadata import ArraySpec


import numpy as np


from dataclasses import dataclass
from functools import reduce
from typing import Iterable, List, Optional
from warnings import warn


@dataclass(frozen=True)
class CodecPipeline:
    codecs: List[Codec]

    @classmethod
    def from_metadata(
        cls,
        codecs_metadata: Iterable[NamedConfig],
        array_metadata: ArraySpec,
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
    def _validate_codecs(codecs: List[Codec], array_metadata: ArraySpec) -> None:
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


async def decode(
    codecs: List[Codec], chunk_bytes: BytesLike, runtime_configuration: RuntimeConfiguration
) -> np.ndarray:
    # todo: increase the arity of the function signature with positions for array_array, array_bytes, and bytes_bytes
    # codices (codexes?)
    _array_array_codecs = [codec for codec in codecs if isinstance(codec, ArrayArrayCodec)]
    _array_bytes_codec = next(codec for codec in codecs if isinstance(codec, ArrayBytesCodec))
    _bytes_bytes_codecs = [codec for codec in codecs if isinstance(codec, BytesBytesCodec)]

    for bb_codec in _bytes_bytes_codecs[::-1]:
        chunk_bytes = await bb_codec.decode(
            chunk_bytes, runtime_configuration=runtime_configuration
        )

    chunk_array = await _array_bytes_codec.decode(
        chunk_bytes, runtime_configuration=runtime_configuration
    )

    for aa_codec in _array_array_codecs[::-1]:
        chunk_array = await aa_codec.decode(
            chunk_array, runtime_configuration=runtime_configuration
        )

    return chunk_array


async def encode(
    codecs: List[Codec], chunk_array: np.ndarray, runtime_configuration: RuntimeConfiguration
) -> Optional[BytesLike]:
    # todo: increase the arity of the function signature with positions for array_array, array_bytes, and bytes_bytes
    # codices (codexes?)
    _array_array_codecs = [codec for codec in codecs if isinstance(codec, ArrayArrayCodec)]
    _array_bytes_codec = next(codec for codec in codecs if isinstance(codec, ArrayBytesCodec))
    _bytes_bytes_codecs = [codec for codec in codecs if isinstance(codec, BytesBytesCodec)]

    for aa_codec in _array_array_codecs:
        chunk_array_maybe = await aa_codec.encode(
            chunk_array, runtime_configuration=runtime_configuration
        )
        if chunk_array_maybe is None:
            return None
        chunk_array = chunk_array_maybe

    chunk_bytes_maybe = await _array_bytes_codec.encode(
        chunk_array, runtime_configuration=runtime_configuration
    )
    if chunk_bytes_maybe is None:
        return None
    chunk_bytes = chunk_bytes_maybe

    for bb_codec in _bytes_bytes_codecs:
        chunk_bytes_maybe = await bb_codec.encode(
            chunk_bytes, runtime_configuration=runtime_configuration
        )
        if chunk_bytes_maybe is None:
            return None
        chunk_bytes = chunk_bytes_maybe

    return chunk_bytes
