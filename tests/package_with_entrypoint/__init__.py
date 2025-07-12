from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import zarr.core.buffer
from zarr.abc.codec import ArrayBytesCodec, CodecInput, CodecPipeline
from zarr.codecs import BytesCodec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.dtype.common import DataTypeValidationError, DTypeJSON, DTypeSpec_V2
from zarr.core.dtype.npy.bool import Bool

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, ClassVar, Literal, Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import ZarrFormat


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


class TestDataType(Bool):
    """
    This is a "data type" that serializes to "test"
    """

    _zarr_v3_name: ClassVar[Literal["test"]] = "test"  # type: ignore[assignment]

    @classmethod
    def from_json(cls, data: DTypeJSON, *, zarr_format: Literal[2, 3]) -> Self:
        if zarr_format == 2 and data == {"name": cls._zarr_v3_name, "object_codec_id": None}:
            return cls()
        if zarr_format == 3 and data == cls._zarr_v3_name:
            return cls()
        raise DataTypeValidationError(
            f"Invalid JSON representation of {cls.__name__}. Got {data!r}"
        )

    def to_json(self, zarr_format: ZarrFormat) -> str | DTypeSpec_V2:  # type: ignore[override]
        if zarr_format == 2:
            return {"name": self._zarr_v3_name, "object_codec_id": None}
        if zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError("zarr_format must be 2 or 3")
