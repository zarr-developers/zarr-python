from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from dataclasses import dataclass
from warnings import warn

from zarr.v3.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    BytesBytesCodec,
    Codec,
)
from zarr.v3.abc.metadata import Metadata
from zarr.v3.codecs.registry import get_codec_class

if TYPE_CHECKING:
    from typing import Dict, Iterator, List, Optional, Tuple, Union
    from typing_extensions import Self
    from zarr.v3.store import StorePath
    from zarr.v3.metadata import ArrayMetadata, RuntimeConfiguration
    from zarr.v3.common import JSON, ArraySpec, BytesLike, SliceSelection


@dataclass(frozen=True)
class CodecPipeline(Metadata):
    array_array_codecs: Tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: Tuple[BytesBytesCodec, ...]

    @classmethod
    def from_dict(cls, data: Union[JSON, List[Codec]]) -> Self:
        out: List[Codec] = []
        for c in data:
            if isinstance(c, Codec):
                out.append(c)
            else:
                out.append(get_codec_class(c["name"]).from_dict(c))
        return CodecPipeline.from_list(out)

    def to_dict(self) -> Dict[str, JSON]:
        return [c.to_dict() for c in self]

    def evolve(self, array_spec: ArraySpec) -> Self:
        return CodecPipeline.from_list([c.evolve(array_spec) for c in self])

    @classmethod
    def from_list(cls, codecs: List[Codec]) -> CodecPipeline:
        from zarr.v3.codecs.sharding import ShardingCodec

        if not any(isinstance(codec, ArrayBytesCodec) for codec in codecs):
            raise ValueError("Exactly one array-to-bytes codec is required.")

        prev_codec: Optional[Codec] = None
        for codec in codecs:
            if prev_codec is not None:
                if isinstance(codec, ArrayBytesCodec) and isinstance(prev_codec, ArrayBytesCodec):
                    raise ValueError(
                        f"ArrayBytesCodec '{type(codec)}' cannot follow after "
                        + f"ArrayBytesCodec '{type(prev_codec)}' because exactly "
                        + "1 ArrayBytesCodec is allowed."
                    )
                if isinstance(codec, ArrayBytesCodec) and isinstance(prev_codec, BytesBytesCodec):
                    raise ValueError(
                        f"ArrayBytesCodec '{type(codec)}' cannot follow after "
                        + f"BytesBytesCodec '{type(prev_codec)}'."
                    )
                if isinstance(codec, ArrayArrayCodec) and isinstance(prev_codec, ArrayBytesCodec):
                    raise ValueError(
                        f"ArrayArrayCodec '{type(codec)}' cannot follow after "
                        + f"ArrayBytesCodec '{type(prev_codec)}'."
                    )
                if isinstance(codec, ArrayArrayCodec) and isinstance(prev_codec, BytesBytesCodec):
                    raise ValueError(
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
            array_array_codecs=tuple(
                codec for codec in codecs if isinstance(codec, ArrayArrayCodec)
            ),
            array_bytes_codec=[codec for codec in codecs if isinstance(codec, ArrayBytesCodec)][0],
            bytes_bytes_codecs=tuple(
                codec for codec in codecs if isinstance(codec, BytesBytesCodec)
            ),
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
