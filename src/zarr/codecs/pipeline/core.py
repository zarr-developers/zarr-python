from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable
import numpy as np
from dataclasses import dataclass
from warnings import warn

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    BytesBytesCodec,
    Codec,
)
from zarr.abc.store import ByteGetter, ByteSetter
from zarr.abc.metadata import Metadata
from zarr.codecs.registry import get_codec_class
from zarr.common import parse_named_configuration

if TYPE_CHECKING:
    from typing import Iterator, List, Optional, Tuple, Union
    from typing_extensions import Self
    from zarr.metadata import ArrayMetadata
    from zarr.config import RuntimeConfiguration
    from zarr.common import JSON, ArraySpec, BytesLike, SliceSelection


@dataclass(frozen=True)
class CodecPipeline(Metadata, ABC):
    array_array_codecs: Tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: Tuple[BytesBytesCodec, ...]

    @classmethod
    def from_dict(cls, data: Iterable[Union[JSON, Codec]]) -> Self:
        out: List[Codec] = []
        if not isinstance(data, Iterable):
            raise TypeError(f"Expected iterable, got {type(data)}")

        for c in data:
            if isinstance(c, Codec):
                out.append(c)
            else:
                name_parsed, _ = parse_named_configuration(c, require_configuration=False)
                out.append(get_codec_class(name_parsed).from_dict(c))  # type: ignore[arg-type]
        return cls.from_list(out)

    def to_dict(self) -> JSON:
        return [c.to_dict() for c in self]

    def evolve(self, array_spec: ArraySpec) -> Self:
        return type(self).from_list([c.evolve(array_spec) for c in self])

    @staticmethod
    def codecs_from_list(
        codecs: List[Codec],
    ) -> Tuple[Tuple[ArrayArrayCodec, ...], ArrayBytesCodec, Tuple[BytesBytesCodec, ...]]:
        from zarr.codecs.sharding import ShardingCodec

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

        return (
            tuple(codec for codec in codecs if isinstance(codec, ArrayArrayCodec)),
            [codec for codec in codecs if isinstance(codec, ArrayBytesCodec)][0],
            tuple(codec for codec in codecs if isinstance(codec, BytesBytesCodec)),
        )

    @classmethod
    def from_list(cls, codecs: List[Codec]) -> Self:
        array_array_codecs, array_bytes_codec, bytes_bytes_codecs = cls.codecs_from_list(codecs)

        return cls(
            array_array_codecs=array_array_codecs,
            array_bytes_codec=array_bytes_codec,
            bytes_bytes_codecs=bytes_bytes_codecs,
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

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        for codec in self:
            byte_length = codec.compute_encoded_size(byte_length, array_spec)
            array_spec = codec.resolve_metadata(array_spec)
        return byte_length

    @abstractmethod
    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[Tuple[Optional[BytesLike], ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[np.ndarray]]:
        pass

    @abstractmethod
    async def decode_partial(
        self,
        batch_info: Iterable[Tuple[ByteGetter, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[np.ndarray]]:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[Tuple[Optional[np.ndarray], ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[BytesLike]]:
        pass

    @abstractmethod
    async def encode_partial(
        self,
        batch_info: Iterable[Tuple[ByteSetter, np.ndarray, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        pass

    @abstractmethod
    async def read_batch(
        self,
        batch_info: Iterable[Tuple[ByteGetter, ArraySpec, SliceSelection, SliceSelection]],
        out: np.ndarray,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        pass

    @abstractmethod
    async def write_batch(
        self,
        batch_info: Iterable[Tuple[ByteSetter, ArraySpec, SliceSelection, SliceSelection]],
        value: np.ndarray,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        pass
