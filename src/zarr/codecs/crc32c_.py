from __future__ import annotations
from dataclasses import dataclass

from typing import TYPE_CHECKING

import numpy as np

from crc32c import crc32c

from zarr.codecs.mixins import BytesBytesCodecBatchMixin
from zarr.codecs.registry import register_codec
from zarr.common import parse_named_configuration

if TYPE_CHECKING:
    from typing import Dict, Optional
    from typing_extensions import Self
    from zarr.common import JSON, BytesLike, ArraySpec
    from zarr.config import RuntimeConfiguration


@dataclass(frozen=True)
class Crc32cCodec(BytesBytesCodecBatchMixin):
    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        parse_named_configuration(data, "crc32c", require_configuration=False)
        return cls()

    def to_dict(self) -> Dict[str, JSON]:
        return {"name": "crc32c"}

    async def decode_single(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        crc32_bytes = chunk_bytes[-4:]
        inner_bytes = chunk_bytes[:-4]

        computed_checksum = np.uint32(crc32c(inner_bytes)).tobytes()
        stored_checksum = bytes(crc32_bytes)
        if computed_checksum != stored_checksum:
            raise ValueError(
                "Stored and computed checksum do not match. "
                + f"Stored: {stored_checksum!r}. Computed: {computed_checksum!r}."
            )
        return inner_bytes

    async def encode_single(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        return chunk_bytes + np.uint32(crc32c(chunk_bytes)).tobytes()

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4


register_codec("crc32c", Crc32cCodec)
