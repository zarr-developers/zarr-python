from dataclasses import dataclass
import math
from typing import Type
from typing_extensions import Self
from warnings import warn

import numpy as np
from zarr.abc.codec import ArrayArrayCodec, BytesBytesCodec, Codec

import numcodecs
from zarr.codecs.registry import register_codec
from zarr.common import JSON, ArraySpec, BytesLike, parse_named_configuration, product, to_thread
from zarr.config import RuntimeConfiguration
from zarr.metadata import ArrayMetadata

CODEC_PREFIX = "https://zarr.dev/numcodecs/"


def parse_codec_configuration(name: str, expected_name_prefix: str) -> dict[str, JSON]:
    parsed_name, parsed_configuration = parse_named_configuration(name)
    if not parsed_name.startswith(expected_name_prefix):
        raise ValueError(
            f"Expected name to start with '{expected_name_prefix}'. Got {name} instead."
        )
    id = parsed_name[len(expected_name_prefix) :]
    return {"id": id, **parsed_configuration}


@dataclass(frozen=True)
class NumcodecsCodec(Codec):
    codec: numcodecs.abc.Codec
    codec_id: str

    def __init__(self, *, codec_config: dict[str, JSON]) -> None:
        if "id" not in codec_config:
            codec_config = {"id": self.codec_id, **codec_config}
        object.__setattr__(self, "codec", numcodecs.get_codec(codec_config))
        warn(
            "Numcodecs codecs are not in the Zarr version 3 specification and "
            "may not be supported by other zarr implementations.",
            category=UserWarning,
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        codec_config = parse_codec_configuration(data, CODEC_PREFIX)
        cls(codec_config)

    def to_dict(self) -> JSON:
        codec_config = self.codec.get_config()
        del codec_config["id"]
        return {
            "name": f"{CODEC_PREFIX}{self.codec.codec_id}",
            "configuration": codec_config,
        }


@dataclass(frozen=True)
class NumcodecsBytesBytesCodec(BytesBytesCodec, NumcodecsCodec):
    def __init__(self, *, codec_config: dict[str, JSON]) -> None:
        super().__init__(codec_config=codec_config)

    async def decode(
        self,
        chunk_bytes: BytesLike,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        return await to_thread(self.codec.decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: BytesLike,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        return await to_thread(self.codec.encode, chunk_bytes)


@dataclass(frozen=True)
class NumcodecsArrayArrayCodec(ArrayArrayCodec, NumcodecsCodec):
    def __init__(self, *, codec_config: dict[str, JSON]) -> None:
        super().__init__(codec_config=codec_config)

    async def decode(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        out = await to_thread(self.codec.decode, chunk_array)
        return out.reshape(chunk_spec.shape)

    async def encode(
        self,
        chunk_array: np.ndarray,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        return await to_thread(self.codec.encode, chunk_array)


def make_bytes_bytes_codec(codec_id: str) -> Type[NumcodecsBytesBytesCodec]:
    # rename for class scope
    _codec_id = codec_id

    @dataclass(frozen=True)
    class _Codec(NumcodecsBytesBytesCodec):
        codec_id = _codec_id

        def __init__(self, codec_config: dict[str, JSON] = {}) -> None:
            super().__init__(codec_config=codec_config)

    return _Codec


def make_array_array_codec(codec_id: str) -> Type[NumcodecsArrayArrayCodec]:
    # rename for class scope
    _codec_id = codec_id

    @dataclass(frozen=True)
    class _Codec(NumcodecsArrayArrayCodec):
        codec_id = _codec_id

        def __init__(self, codec_config: dict[str, JSON] = {}) -> None:
            if "id" not in codec_config:
                codec_config = {"id": self.codec_id, **codec_config}
            super().__init__(codec_config=codec_config)

    return _Codec


@dataclass(frozen=True)
class AsTypeCodec(NumcodecsArrayArrayCodec):
    codec_id = "astype"

    def __init__(self, codec_config: dict[str, JSON] = {}) -> None:
        super().__init__(codec_config=codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return ArraySpec(
            chunk_spec.shape,
            np.dtype(self.codec.get_config()["encode_dtype"]),
            chunk_spec.fill_value,
        )

    def evolve(self, array_spec: ArraySpec) -> Self:
        codec_config = self.codec.get_config()
        if str(array_spec.dtype) != codec_config["decode_dtype"]:
            return AsTypeCodec({**codec_config, "decode_dtype": str(array_spec.dtype)})


@dataclass(frozen=True)
class PackbitsCodec(NumcodecsArrayArrayCodec):
    codec_id = "packbits"

    def __init__(self, codec_config: dict[str, JSON] = {}) -> None:
        super().__init__(codec_config=codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return ArraySpec(
            (1 + math.ceil(product(chunk_spec.shape) / 8),),
            np.dtype("uint8"),
            chunk_spec.fill_value,
        )

    def validate(self, array_metadata: ArrayMetadata) -> None:
        if array_metadata.dtype != np.dtype("bool"):
            raise ValueError(f"Packbits filter requires bool dtype. Got {array_metadata.dtype}.")


register_codec(f"{CODEC_PREFIX}blosc", make_bytes_bytes_codec("blosc"))
register_codec(f"{CODEC_PREFIX}lz4", make_bytes_bytes_codec("lz4"))
register_codec(f"{CODEC_PREFIX}zstd", make_bytes_bytes_codec("zstd"))
register_codec(f"{CODEC_PREFIX}zlib", make_bytes_bytes_codec("zlib"))
register_codec(f"{CODEC_PREFIX}gzip", make_bytes_bytes_codec("gzip"))
register_codec(f"{CODEC_PREFIX}bz2", make_bytes_bytes_codec("bz2"))
register_codec(f"{CODEC_PREFIX}lzma", make_bytes_bytes_codec("lzma"))

register_codec(f"{CODEC_PREFIX}delta", make_array_array_codec("delta"))
register_codec(f"{CODEC_PREFIX}fixedscaleoffset", make_array_array_codec("fixedscaleoffset"))
register_codec(f"{CODEC_PREFIX}quantize", make_array_array_codec("quantize"))
register_codec(f"{CODEC_PREFIX}bitround", make_array_array_codec("bitround"))
register_codec(f"{CODEC_PREFIX}packbits", PackbitsCodec)
register_codec(f"{CODEC_PREFIX}astype", AsTypeCodec)
