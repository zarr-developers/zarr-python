from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING
from dataclasses import dataclass


from numcodecs.zstd import Zstd

from zarr.abc.codec import BytesBytesCodec
from zarr.codecs.registry import register_codec
from zarr.common import parse_named_configuration, to_thread

if TYPE_CHECKING:
    from typing import Dict, Optional
    from typing_extensions import Self
    from zarr.config import RuntimeConfiguration
    from zarr.common import BytesLike, JSON, ArraySpec

DEFAULT_ZSTD_LEVEL = 3


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

    level: int = DEFAULT_ZSTD_LEVEL
    checksum: bool = False

    def __init__(self, *, level: int = DEFAULT_ZSTD_LEVEL, checksum: bool = False) -> None:
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

    @cached_property
    def _zstd_codec(self) -> Zstd:
        config_dict = {"level": self.level, "checksum": self.checksum}
        return Zstd.from_config(config_dict)

    async def decode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        return await to_thread(self._zstd_codec.decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        return await to_thread(self._zstd_codec.encode, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


register_codec("zstd", ZstdCodec)
