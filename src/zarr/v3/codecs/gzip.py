from __future__ import annotations
from dataclasses import dataclass, field

from typing import TYPE_CHECKING, Dict

from zarr.v3.abc.metadata import Metadata
from numcodecs.gzip import GZip
from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import JSON, ArraySpec, parse_name, to_thread

if TYPE_CHECKING:
    from zarr.v3.metadata import RuntimeConfiguration
    from zarr.v3.common import BytesLike
    from typing_extensions import Self
    from typing import Optional, Dict, Literal


def parse_gzip_level(data: JSON) -> int:
    if data not in range(0, 10):
        raise ValueError(
            f"Expected an integer from the inclusive range (0, 9). Got {data} instead."
        )
    return data


@dataclass(frozen=True)
class GzipCodec(BytesBytesCodec):
    is_fixed_size = False

    level: int = 5

    def __init__(self, *, level=5) -> None:
        level_parsed = parse_gzip_level(level)

        object.__setattr__(self, "level", level_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        parse_name(data["name"], "gzip")
        return cls(**data["configuration"])

    def to_dict(self) -> Dict[str, JSON]:
        return {"name": "gzip", "configuration": {"level": self.level}}

    async def decode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        return await to_thread(GZip(self.level).decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        return await to_thread(GZip(self.level).encode, chunk_bytes)

    def compute_encoded_size(
        self,
        _input_byte_length: int,
        _chunk_spec: ArraySpec,
    ) -> int:
        raise NotImplementedError


register_codec("gzip", GzipCodec)
