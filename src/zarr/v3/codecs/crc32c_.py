from __future__ import annotations
from dataclasses import dataclass

from typing import TYPE_CHECKING

import numpy as np

from crc32c import crc32c

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import parse_named_configuration
from zarr.v3.buffer import Buffer, as_buffer

if TYPE_CHECKING:
    from typing import Dict, Optional
    from typing_extensions import Self
    from zarr.v3.common import JSON, ArraySpec
    from zarr.v3.config import RuntimeConfiguration


@dataclass(frozen=True)
class Crc32cCodec(BytesBytesCodec):
    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        parse_named_configuration(data, "crc32c", require_configuration=False)
        return cls()

    def to_dict(self) -> Dict[str, JSON]:
        return {"name": "crc32c"}

    async def decode(
        self,
        chunk_bytes: Buffer,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Buffer:
        data = chunk_bytes.memoryview()
        crc32_bytes = data[-4:]
        inner_bytes = data[:-4]

        computed_checksum = np.uint32(crc32c(inner_bytes)).tobytes()
        stored_checksum = bytes(crc32_bytes)
        if computed_checksum != stored_checksum:
            raise ValueError(
                "Stored and computed checksum do not match. "
                + f"Stored: {stored_checksum!r}. Computed: {computed_checksum!r}."
            )
        return Buffer(inner_bytes)

    async def encode(
        self,
        chunk_bytes: Buffer,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[Buffer]:
        checksum = crc32c(chunk_bytes.memoryview())
        return as_buffer(chunk_bytes.to_bytes() + np.uint32(checksum).tobytes())

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4


register_codec("crc32c", Crc32cCodec)
