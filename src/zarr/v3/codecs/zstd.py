from __future__ import annotations
from typing import TYPE_CHECKING, Dict
from dataclasses import dataclass


from zstandard import ZstdCompressor, ZstdDecompressor

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import parse_name, to_thread, ArraySpec

if TYPE_CHECKING:
    from zarr.v3.metadata import RuntimeConfiguration
    from typing import Literal, Dict, Optional
    from typing_extensions import Self
    from zarr.v3.common import BytesLike, JSON


def parse_zstd_level(data: JSON) -> int:
    if isinstance(data, int):
        if data >= 23:
            msg = f"Value must be less than or equal to 22. Got {data} instead."
            raise ValueError(msg)
        return data
    msg = f"Got value with type {type(data)}, but expected an int"
    raise TypeError(msg)


def parse_checksum(data: JSON) -> bool:
    if isinstance(data, bool):
        return data
    msg = f"Expected bool, got {type(data)}"
    raise TypeError(msg)


@dataclass(frozen=True)
class ZstdCodec(BytesBytesCodec):
    is_fixed_size = True

    level: int = 0
    checksum: bool = False

    def __init__(self, *, level, checksum) -> None:
        level_parsed = parse_zstd_level(level)
        checksum_parsed = parse_checksum(checksum)

        object.__setattr__(self, "level", level_parsed)
        object.__setattr__(self, "checksum", checksum_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        parse_name(data["name"], "zstd")
        return cls(**data["configuration"])

    def to_dict(self) -> Dict[str, JSON]:
        return {"name": "zstd", "configuration": {"level": self.level, "checksum": self.checksum}}

    def _compress(self, data: bytes) -> bytes:
        ctx = ZstdCompressor(level=self.level, write_checksum=self.checksum)
        return ctx.compress(data)

    def _decompress(self, data: bytes) -> bytes:
        ctx = ZstdDecompressor()
        return ctx.decompress(data)

    async def decode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        return await to_thread(self._decompress, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        return await to_thread(self._compress, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


register_codec("zstd", ZstdCodec)
