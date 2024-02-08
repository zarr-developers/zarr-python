from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

from zarr.v3.abc.metadata import Metadata

from zstandard import ZstdCompressor, ZstdDecompressor

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import to_thread, ArraySpec

if TYPE_CHECKING:
    from zarr.v3.metadata import RuntimeConfiguration
    from typing import Any, Literal, Dict, Type, Optional
    from typing_extensions import Self
    from zarr.v3.common import BytesLike, NamedConfig


def parse_zstd_level(data: Any) -> int:
    if isinstance(data, int):
        if data >= 23:
            msg = f"Value must be less than or equal to 22. Got {data} instead."
            raise ValueError(msg)
        return data
    msg = f"Got value with type {type(data)}, but expected an int"
    raise TypeError(msg)


def parse_checksum(data: Any) -> bool:
    if isinstance(data, bool):
        return data
    msg = f"Expected bool, got {type(data)}"
    raise TypeError(msg)


def parse_name(data: Any) -> Literal["zstd"]:
    if data == "zstd":
        return data
    msg = f"Expected 'zstd', got {data}"
    raise ValueError(msg)


@dataclass(frozen=True)
class ZstdCodecConfigurationMetadata(Metadata):
    level: int = 0
    checksum: bool = False

    def __init__(self, level: int, checksum: bool):
        level_parsed = parse_zstd_level(level)
        checksum_parsed = parse_checksum(checksum)
        object.__setattr__(self, "level", level_parsed)
        object.__setattr__(self, "checksum", checksum_parsed)


@dataclass(frozen=True)
class ZstdCodecMetadata(Metadata):
    configuration: ZstdCodecConfigurationMetadata
    name: Literal["zstd"] = field(default="zstd", init=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        _ = parse_name(data.pop("name"))
        return cls(**data)


@dataclass(frozen=True)
class ZstdCodec(BytesBytesCodec):
    configuration: ZstdCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(cls, codec_metadata: NamedConfig) -> ZstdCodec:
        assert isinstance(codec_metadata, ZstdCodecMetadata)
        return cls(configuration=codec_metadata.configuration)

    @classmethod
    def get_metadata_class(cls) -> Type[ZstdCodecMetadata]:
        return ZstdCodecMetadata

    def _compress(self, data: bytes) -> bytes:
        ctx = ZstdCompressor(level=self.metadata.level, write_checksum=self.metadata.checksum)
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
