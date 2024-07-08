from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy.typing as npt
from zstandard import ZstdCompressor, ZstdDecompressor

from zarr.abc.codec import BytesBytesCodec
from zarr.array_spec import ArraySpec
from zarr.buffer import Buffer
from zarr.buffer.cpu import as_numpy_array_wrapper
from zarr.codecs.registry import register_codec
from zarr.common import JSON, parse_named_configuration, to_thread

if TYPE_CHECKING:
    from typing_extensions import Self


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
class ZstdCodec(BytesBytesCodec):
    is_fixed_size = True

    level: int = 0
    checksum: bool = False

    def __init__(self, *, level: int = 0, checksum: bool = False) -> None:
        level_parsed = parse_zstd_level(level)
        checksum_parsed = parse_checksum(checksum)

        object.__setattr__(self, "level", level_parsed)
        object.__setattr__(self, "checksum", checksum_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "zstd")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "zstd", "configuration": {"level": self.level, "checksum": self.checksum}}

    def _compress(self, data: npt.NDArray[Any]) -> bytes:
        ctx = ZstdCompressor(level=self.level, write_checksum=self.checksum)
        return ctx.compress(data.tobytes())

    def _decompress(self, data: npt.NDArray[Any]) -> bytes:
        ctx = ZstdDecompressor()
        return ctx.decompress(data.tobytes())

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await to_thread(
            as_numpy_array_wrapper, self._decompress, chunk_bytes, chunk_spec.prototype
        )

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return await to_thread(
            as_numpy_array_wrapper, self._compress, chunk_bytes, chunk_spec.prototype
        )

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


register_codec("zstd", ZstdCodec)
