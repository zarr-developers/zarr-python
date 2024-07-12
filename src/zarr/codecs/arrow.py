from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from zarr.abc.codec import ArrayBytesCodec
from zarr.array_spec import ArraySpec
from zarr.buffer import Buffer, NDBuffer
from zarr.codecs.registry import register_codec
from zarr.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from typing_extensions import Self

CHUNK_FIELD_NAME = "zarr_chunk"


@dataclass(frozen=True)
class ArrowRecordBatchCodec(ArrayBytesCodec):
    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "arrow", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "arrow"}

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        assert isinstance(chunk_bytes, Buffer)

        # TODO: make this compatible with buffer prototype
        arrow_buffer = pa.py_buffer(chunk_bytes.to_bytes())
        with pa.ipc.open_stream(arrow_buffer) as reader:
            batches = [b for b in reader]
        assert len(batches) == 1
        arrow_array = batches[0][CHUNK_FIELD_NAME]
        chunk_array = chunk_spec.prototype.nd_buffer.from_ndarray_like(
            arrow_array.to_numpy(zero_copy_only=False)
        )

        # ensure correct chunk shape
        if chunk_array.shape != chunk_spec.shape:
            chunk_array = chunk_array.reshape(
                chunk_spec.shape,
            )
        return chunk_array

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        assert isinstance(chunk_array, NDBuffer)
        arrow_array = pa.array(chunk_array.as_ndarray_like().ravel())
        rb = pa.record_batch([arrow_array], names=[CHUNK_FIELD_NAME])
        # TODO: allocate buffer differently
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, rb.schema) as writer:
            writer.write_batch(rb)
        return chunk_spec.prototype.buffer.from_bytes(memoryview(sink.getvalue()))

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise ValueError("Don't know how to compute encoded size!")


register_codec("arrow", ArrowRecordBatchCodec)
