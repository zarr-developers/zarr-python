from __future__ import annotations
from dataclasses import dataclass

from typing import TYPE_CHECKING

import numpy as np

from crc32c import crc32c

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import parse_name

if TYPE_CHECKING:
    from typing import Dict, Optional
    from typing_extensions import Self
    from zarr.v3.common import JSON, BytesLike, RuntimeConfiguration, ArraySpec


@dataclass(frozen=True)
class Crc32cCodec(BytesBytesCodec):
    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        parse_name(data["name"], "crc32c")
        return cls()

    def to_dict(self) -> Dict[str, JSON]:
        return {"name": "crc32c"}

    async def decode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        crc32_bytes = chunk_bytes[-4:]
        inner_bytes = chunk_bytes[:-4]

        assert np.uint32(crc32c(inner_bytes)).tobytes() == bytes(crc32_bytes)
        return inner_bytes

    async def encode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        return chunk_bytes + np.uint32(crc32c(chunk_bytes)).tobytes()

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4


register_codec("crc32c", Crc32cCodec)
