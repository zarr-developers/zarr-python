from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import typing_extensions
from crc32c import crc32c

from zarr.abc.codec import BytesBytesCodec
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


@dataclass(frozen=True)
class Crc32cCodec(BytesBytesCodec):
    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        parse_named_configuration(data, "crc32c", require_configuration=False)
        return cls()

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "crc32c"}

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        data = chunk_bytes.as_numpy_array()
        crc32_bytes = data[-4:]
        inner_bytes = data[:-4]

        # Need to do a manual cast until https://github.com/numpy/numpy/issues/26783 is resolved
        computed_checksum = np.uint32(crc32c(cast(typing_extensions.Buffer, inner_bytes))).tobytes()
        stored_checksum = bytes(crc32_bytes)
        if computed_checksum != stored_checksum:
            raise ValueError(
                f"Stored and computed checksum do not match. Stored: {stored_checksum!r}. Computed: {computed_checksum!r}."
            )
        return chunk_spec.prototype.buffer.from_array_like(inner_bytes)

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        data = chunk_bytes.as_numpy_array()
        # Calculate the checksum and "cast" it to a numpy array
        checksum = np.array([crc32c(cast(typing_extensions.Buffer, data))], dtype=np.uint32)
        # Append the checksum (as bytes) to the data
        return chunk_spec.prototype.buffer.from_array_like(np.append(data, checksum.view("b")))

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4


register_codec("crc32c", Crc32cCodec)
