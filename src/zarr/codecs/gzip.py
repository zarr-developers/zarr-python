from __future__ import annotations
from dataclasses import dataclass

from typing import TYPE_CHECKING

from numcodecs.gzip import GZip
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs.registry import register_codec
from zarr.common import parse_named_configuration, to_thread

if TYPE_CHECKING:
    from typing import Optional, Dict
    from typing_extensions import Self
    from zarr.common import JSON, ArraySpec, BytesLike
    from zarr.config import RuntimeConfiguration


def parse_gzip_level(data: JSON) -> int:
    if not isinstance(data, (int)):
        raise TypeError(f"Expected int, got {type(data)}")
    if data not in range(0, 10):
        raise ValueError(
            f"Expected an integer from the inclusive range (0, 9). Got {data} instead."
        )
    return data


@dataclass(frozen=True)
class GzipCodec(BytesBytesCodec):
    is_fixed_size = False

    level: int = 5

    def __init__(self, *, level: int = 5) -> None:
        level_parsed = parse_gzip_level(level)

        object.__setattr__(self, "level", level_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "gzip")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

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
