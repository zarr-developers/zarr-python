from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING
import numpy as np
from dataclasses import dataclass

from zarr.abc.codec import (
    ByteGetter,
    ByteSetter,
    Codec,
)
from zarr.codecs.pipeline.batched import BatchedCodecPipeline
from zarr.codecs.pipeline.core import CodecPipeline
from zarr.common import concurrent_map

if TYPE_CHECKING:
    from typing import List, Optional, Tuple, Iterable
    from typing_extensions import Self
    from zarr.config import RuntimeConfiguration
    from zarr.common import ArraySpec, BytesLike, SliceSelection

DEFAULT_BATCH_SIZE = 1000


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


@dataclass(frozen=True)
class HybridCodecPipeline(CodecPipeline):
    batch_size: int  # TODO: There needs to be a way of specifying this from the user code
    batched_codec_pipeline: BatchedCodecPipeline

    @classmethod
    def from_list(cls, codecs: List[Codec], *, batch_size: Optional[int] = None) -> Self:
        array_array_codecs, array_bytes_codec, bytes_bytes_codecs = cls.codecs_from_list(codecs)

        return cls(
            array_array_codecs=array_array_codecs,
            array_bytes_codec=array_bytes_codec,
            bytes_bytes_codecs=bytes_bytes_codecs,
            batch_size=batch_size or DEFAULT_BATCH_SIZE,
            batched_codec_pipeline=BatchedCodecPipeline(
                array_array_codecs=array_array_codecs,
                array_bytes_codec=array_bytes_codec,
                bytes_bytes_codecs=bytes_bytes_codecs,
            ),
        )

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[Tuple[Optional[BytesLike], ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[np.ndarray]]:
        output: list[Optional[np.ndarray]] = []
        for batch_info in batched(chunk_bytes_and_specs, self.batch_size):
            output.extend(
                await self.batched_codec_pipeline.decode(batch_info, runtime_configuration)
            )
        return output

    async def decode_partial(
        self,
        batch_info: Iterable[Tuple[ByteGetter, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[np.ndarray]]:
        output: list[Optional[np.ndarray]] = []
        for single_batch_info in batched(batch_info, self.batch_size):
            output.extend(
                await self.batched_codec_pipeline.decode_partial(
                    single_batch_info, runtime_configuration
                )
            )
        return output

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[Tuple[Optional[np.ndarray], ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[BytesLike]]:
        output: list[Optional[BytesLike]] = []
        for single_batch_info in batched(chunk_arrays_and_specs, self.batch_size):
            output.extend(
                await self.batched_codec_pipeline.encode(single_batch_info, runtime_configuration)
            )
        return output

    async def encode_partial(
        self,
        batch_info: Iterable[Tuple[ByteSetter, np.ndarray, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        for single_batch_info in batched(batch_info, self.batch_size):
            await self.batched_codec_pipeline.encode_partial(
                single_batch_info, runtime_configuration
            )

    async def read_batch(
        self,
        batch_info: Iterable[Tuple[ByteGetter, ArraySpec, SliceSelection, SliceSelection]],
        out: np.ndarray,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        await concurrent_map(
            [
                (single_batch_info, out, runtime_configuration)
                for single_batch_info in batched(batch_info, self.batch_size)
            ],
            self.batched_codec_pipeline.read_batch,
            runtime_configuration.concurrency,
        )

    async def write_batch(
        self,
        batch_info: Iterable[Tuple[ByteSetter, ArraySpec, SliceSelection, SliceSelection]],
        value: np.ndarray,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        await concurrent_map(
            [
                (
                    single_batch_info,
                    value,
                    runtime_configuration,
                )
                for single_batch_info in batched(batch_info, self.batch_size)
            ],
            self.batched_codec_pipeline.write_batch,
            runtime_configuration.concurrency,
        )
