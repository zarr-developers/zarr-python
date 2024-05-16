from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass


from zstandard import ZstdCompressor, ZstdDecompressor

from zarr.codecs.mixins import BytesBytesCodecBatchMixin
from zarr.buffer import Buffer, as_numpy_array_wrapper
from zarr.codecs.registry import register_codec
from zarr.common import parse_named_configuration, to_thread

if TYPE_CHECKING:
    from typing import Dict, Optional
    from typing_extensions import Self
    from zarr.common import JSON, ArraySpec


def parse_zstd_level(data: JSON) -> int:
    if isinstance(data, int):
        if data >= 23:
            raise ValueError(f"Value must be less than or equal to 22. Got {data} instead.")
        return data
    raise TypeError(f"Got value with type {type(data)}, but expected an int.")


def parse_checksum(data: JSON) -> bool:
    if isinstance(data, bool):
        return data
    raise TypeError(f"Expected bool. Got {type(data)}.")


@dataclass(frozen=True)
class ZstdCodec(BytesBytesCodecBatchMixin):
    is_fixed_size = True

    level: int = 0
    checksum: bool = False

    def __init__(self, *, level: int = 0, checksum: bool = False) -> None:
        level_parsed = parse_zstd_level(level)
        checksum_parsed = parse_checksum(checksum)

        object.__setattr__(self, "level", level_parsed)
        object.__setattr__(self, "checksum", checksum_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "zstd")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, JSON]:
        return {"name": "zstd", "configuration": {"level": self.level, "checksum": self.checksum}}

    def _compress(self, data: bytes) -> bytes:
        ctx = ZstdCompressor(level=self.level, write_checksum=self.checksum)
        return ctx.compress(data)

    def _decompress(self, data: bytes) -> bytes:
        ctx = ZstdDecompressor()
        return ctx.decompress(data)

    async def decode_single(
        self,
        chunk_bytes: Buffer,
        _chunk_spec: ArraySpec,
    ) -> Buffer:
        return await to_thread(as_numpy_array_wrapper, self._decompress, chunk_bytes)

    async def encode_single(
        self,
        chunk_bytes: Buffer,
        _chunk_spec: ArraySpec,
    ) -> Optional[Buffer]:
        return await to_thread(as_numpy_array_wrapper, self._compress, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


register_codec("zstd", ZstdCodec)
