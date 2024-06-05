from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from numcodecs.gzip import GZip

from zarr.abc.codec import BytesBytesCodec
from zarr.array_spec import ArraySpec
from zarr.buffer import Buffer, as_numpy_array_wrapper
from zarr.codecs.registry import register_codec
from zarr.common import JSON, parse_named_configuration, to_thread

if TYPE_CHECKING:
    from typing_extensions import Self


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
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "gzip")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "gzip", "configuration": {"level": self.level}}

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await to_thread(
            as_numpy_array_wrapper, GZip(self.level).decode, chunk_bytes, chunk_spec.prototype
        )

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return await to_thread(
            as_numpy_array_wrapper, GZip(self.level).encode, chunk_bytes, chunk_spec.prototype
        )

    def compute_encoded_size(
        self,
        _input_byte_length: int,
        _chunk_spec: ArraySpec,
    ) -> int:
        raise NotImplementedError


register_codec("gzip", GzipCodec)
