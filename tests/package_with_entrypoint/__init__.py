from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

import zarr.core.buffer
from zarr.abc.codec import ArrayBytesCodec, CodecInput, CodecPipeline
from zarr.codecs import BytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer


class TestEntrypointCodec(ArrayBytesCodec):
    is_fixed_size = True

    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[CodecInput | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        return [None]

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[CodecInput | None, ArraySpec]],
    ) -> npt.NDArray[Any]:
        return np.array(1)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length


class TestEntrypointCodecPipeline(CodecPipeline):
    def __init__(self, batch_size: int = 1) -> None:
        pass

    async def encode(
        self, chunks_and_specs: Iterable[tuple[CodecInput | None, ArraySpec]]
    ) -> Iterable[Buffer | None]:
        return [None]

    async def decode(
        self, chunks_and_specs: Iterable[tuple[CodecInput | None, ArraySpec]]
    ) -> Iterable[NDBuffer | None]:
        return np.array(1)


class TestEntrypointBuffer(Buffer):
    pass


class TestEntrypointNDBuffer(NDBuffer):
    pass


class TestEntrypointGroup:
    class Codec(BytesCodec):
        pass

    class Buffer(zarr.core.buffer.Buffer):
        pass

    class NDBuffer(zarr.core.buffer.NDBuffer):
        pass

    class Pipeline(CodecPipeline):
        pass
