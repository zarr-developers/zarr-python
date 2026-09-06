from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from numcodecs.gzip import GZip

from zarr.abc.codec import BytesBytesCodec
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON, parse_named_configuration
from zarr.core.json_parse import parse_field

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


def parse_gzip_level(data: JSON) -> int:
    parsed: int = parse_field(data, int, "level", error=TypeError)
    if parsed not in range(10):
        raise ValueError(
            f"Expected an integer from the inclusive range (0, 9). Got {parsed} instead."
        )
    return parsed


def _gzip_streams_equal_except_mtime(a: bytes, b: bytes) -> bool:
    """Compare two gzip streams, ignoring the MTIME field of the header.

    Per RFC 1952 the gzip header is [magic(2)][CM(1)][FLG(1)][MTIME(4)][XFL(1)][OS(1)],
    so bytes 4-8 are MTIME. The fixed offsets assume the standard 10-byte header
    with no FNAME/FEXTRA/FCOMMENT flags set, which holds here because numcodecs'
    ``GZip.encode`` wraps ``gzip.GzipFile`` without a filename.
    """
    if len(a) != len(b):
        return False

    return a[:4] == b[:4] and a[8:] == b[8:]


@dataclass(frozen=True)
class GzipCodec(BytesBytesCodec):
    """gzip codec"""

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

    @cached_property
    def _gzip_codec(self) -> GZip:
        return GZip(self.level)

    def _decode_sync(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return as_numpy_array_wrapper(self._gzip_codec.decode, chunk_bytes, chunk_spec.prototype)

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    def _encode_sync(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return as_numpy_array_wrapper(self._gzip_codec.encode, chunk_bytes, chunk_spec.prototype)

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return await asyncio.to_thread(self._encode_sync, chunk_bytes, chunk_spec)

    def compute_encoded_size(
        self,
        _input_byte_length: int,
        _chunk_spec: ArraySpec,
    ) -> int:
        raise NotImplementedError
