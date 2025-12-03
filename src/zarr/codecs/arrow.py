from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

from arro3.core import Array, Table
from arro3.io import read_ipc_stream, write_ipc_stream

from zarr.abc.codec import ArrayBytesCodec
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer


@dataclass(frozen=True)
class ArrowIPCCodec(ArrayBytesCodec):
    """Arrow IPC codec"""

    column_name: str

    def __init__(self, *, column_name: str = "zarr_array") -> None:
        object.__setattr__(self, "column_name", column_name)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "arrow-ipc", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "arrow_ipc", "configuration": {"column_name": self.column_name}}

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        # TODO: possibly parse array dtype to configure codec
        return self

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        record_batch_reader = read_ipc_stream(io.BytesIO(chunk_bytes.as_buffer_like()))
        # Note: we only expect a single batch per chunk
        record_batch = record_batch_reader.read_next_batch()
        array = record_batch.column(self.column_name)
        numpy_array = array.to_numpy()
        # all arrow arrays are flat; reshape to chunk shape
        numpy_array.shape = chunk_spec.shape
        # make sure we got the right dtype out
        # assert numpy_array.dtype == chunk_spec.dtype.to_native_dtype(), (
        #     f"dtype mismatch, got {numpy_array.dtype}, expected {chunk_spec.dtype.to_native_dtype()}"
        # )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(numpy_array)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        # TODO: generalize flattening strategy to prevent memory copies
        numpy_array = chunk_array.as_ndarray_like().ravel(order="C")
        arrow_array = Array.from_numpy(numpy_array)
        table = Table.from_arrays(arrays=[arrow_array], names=[self.column_name])
        # TODO: figure out how to avoid copying the bytes to a new buffer!
        # Doh, this is the whole point of Arrow, right?
        buffer = io.BytesIO()
        write_ipc_stream(table, buffer)
        return chunk_spec.prototype.buffer.from_bytes(buffer.getvalue())

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
