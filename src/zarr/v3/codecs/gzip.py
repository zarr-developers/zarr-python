from __future__ import annotations
from dataclasses import dataclass, field

from typing import TYPE_CHECKING

from zarr.v3.abc.metadata import Metadata
from numcodecs.gzip import GZip
from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import ArraySpec, to_thread

if TYPE_CHECKING:
    from zarr.v3.metadata import RuntimeConfiguration
    from zarr.v3.common import BytesLike, NamedConfig
    from typing_extensions import Self
    from typing import Any, Optional, Dict, Literal, Type


def parse_gzip_level(data: Any) -> int:
    if data not in range(0, 10):
        msg = f"Expected an integer from the inclusive range (0, 9). Got {data} instead."
        raise ValueError(msg)
    return data


@dataclass(frozen=True)
class GzipCodecConfigurationMetadata(Metadata):
    level: int = 5

    def __init__(self, level: int):
        level_parsed = parse_gzip_level(level)
        object.__setattr__(self, "level", level_parsed)


@dataclass(frozen=True)
class GzipCodecMetadata(Metadata):
    configuration: GzipCodecConfigurationMetadata
    name: Literal["gzip"] = field(default="gzip", init=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(configuration=GzipCodecConfigurationMetadata.from_dict(data["configuration"]))


@dataclass(frozen=True)
class GzipCodec(BytesBytesCodec):
    configuration: GzipCodecConfigurationMetadata
    is_fixed_size: Literal[True] = field(default=True, init=False)

    @classmethod
    def from_metadata(cls, codec_metadata: NamedConfig) -> GzipCodec:
        assert isinstance(codec_metadata, GzipCodecMetadata)

        return cls(configuration=codec_metadata.configuration)

    @classmethod
    def get_metadata_class(cls) -> Type[GzipCodecMetadata]:
        return GzipCodecMetadata

    async def decode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        return await to_thread(GZip(self.configuration.level).decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        return await to_thread(GZip(self.configuration.level).encode, chunk_bytes)

    def compute_encoded_size(
        self,
        _input_byte_length: int,
        _chunk_spec: ArraySpec,
    ) -> int:
        raise NotImplementedError


register_codec("gzip", GzipCodec)
