from dataclasses import dataclass
from functools import cached_property
import math
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


def parse_codec_configuration(data: dict[str, JSON], expected_name_prefix: str) -> dict[str, JSON]:
    parsed_name, parsed_configuration = parse_named_configuration(data)
    if not parsed_name.startswith(expected_name_prefix):
        raise ValueError(
            f"Expected name to start with '{expected_name_prefix}'. Got {parsed_name} instead."
        )
    id = parsed_name[len(expected_name_prefix) :]
    return {"id": id, **parsed_configuration}


@dataclass(frozen=True)
class NumcodecsCodec(Codec):
    codec_config: dict[str, JSON]

    def __init__(self, *, codec_id: str | None = None, codec_config: dict[str, JSON]) -> None:
        if "id" not in codec_config:
            if not codec_id:
                raise ValueError(
                    "The codec id needs to be supplied either through the id attribute "
                    "of the codec_config or through the codec_id argument."
                )
            codec_config = {"id": codec_id, **codec_config}
        elif codec_id and codec_config["id"] != codec_id:
            raise ValueError(f"Codec id does not match {codec_id}. Got: {codec_config['id']}.")

        object.__setattr__(self, "codec_config", codec_config)
        warn(
            "Numcodecs codecs are not in the Zarr version 3 specification and "
            "may not be supported by other zarr implementations.",
            category=UserWarning,
        )

    @cached_property
    def _codec(self) -> numcodecs.abc.Codec:
        return numcodecs.get_codec(self.codec_config)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        codec_config = parse_codec_configuration(data, CODEC_PREFIX)
        assert isinstance(codec_config["id"], str)  # for mypy
        return cls(codec_config=codec_config)

    def to_dict(self) -> JSON:
        codec_config = self.codec_config.copy()
        codec_id = codec_config.pop("id")
        return {
            "name": f"{CODEC_PREFIX}{codec_id}",
            "configuration": codec_config,
        }


class NumcodecsBytesBytesCodec(NumcodecsCodec, BytesBytesCodec):
    def __init__(self, *, codec_id: str, codec_config: dict[str, JSON]) -> None:
        super().__init__(codec_id=codec_id, codec_config=codec_config)

    async def decode(
        self,
        chunk_bytes: BytesLike,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        return await to_thread(self._codec.decode, chunk_bytes)

    def _encode(self, chunk_bytes: BytesLike) -> BytesLike:
        encoded = self._codec.encode(chunk_bytes)
        if isinstance(encoded, np.ndarray):  # Required for checksum codecs
            return encoded.tobytes()
        return encoded

    async def encode(
        self,
        chunk_bytes: BytesLike,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        return await to_thread(self._encode, chunk_bytes)


class NumcodecsArrayArrayCodec(NumcodecsCodec, ArrayArrayCodec):
    def __init__(self, *, codec_id: str, codec_config: dict[str, JSON]) -> None:
        super().__init__(codec_id=codec_id, codec_config=codec_config)

    async def decode(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        out = await to_thread(self._codec.decode, chunk_array)
        return out.reshape(chunk_spec.shape)

    async def encode(
        self,
        chunk_array: np.ndarray,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        return await to_thread(self._codec.encode, chunk_array)


def make_bytes_bytes_codec(codec_id: str) -> type[NumcodecsBytesBytesCodec]:
    # rename for class scope
    _codec_id = codec_id

    class _Codec(NumcodecsBytesBytesCodec):
        def __init__(self, codec_config: dict[str, JSON] = {}) -> None:
            super().__init__(codec_id=_codec_id, codec_config=codec_config)

    return _Codec


def make_array_array_codec(codec_id: str) -> type[NumcodecsArrayArrayCodec]:
    # rename for class scope
    _codec_id = codec_id

    class _Codec(NumcodecsArrayArrayCodec):
        def __init__(self, codec_config: dict[str, JSON] = {}) -> None:
            super().__init__(codec_id=_codec_id, codec_config=codec_config)

    return _Codec


def make_checksum_codec(codec_id: str) -> type[NumcodecsBytesBytesCodec]:
    # rename for class scope
    _codec_id = codec_id

    class _ChecksumCodec(NumcodecsBytesBytesCodec):
        def __init__(self, codec_config: dict[str, JSON] = {}) -> None:
            super().__init__(codec_id=_codec_id, codec_config=codec_config)

        def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
            return input_byte_length + 4

    return _ChecksumCodec


class FixedScaleOffsetCodec(NumcodecsArrayArrayCodec):
    def __init__(self, codec_config: dict[str, JSON] = {}) -> None:
        super().__init__(codec_id="fixedscaleoffset", codec_config=codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            return ArraySpec(
                chunk_spec.shape,
                np.dtype(astype),
                chunk_spec.fill_value,
            )
        return chunk_spec

    def evolve(self, array_spec: ArraySpec) -> Self:
        if str(array_spec.dtype) != self.codec_config.get("dtype"):
            return self.__class__({**self.codec_config, "dtype": str(array_spec.dtype)})
        return self


class QuantizeCodec(NumcodecsArrayArrayCodec):
    def __init__(self, codec_config: dict[str, JSON] = {}) -> None:
        super().__init__(codec_id="quantize", codec_config=codec_config)

    def evolve(self, array_spec: ArraySpec) -> Self:
        if str(array_spec.dtype) != self.codec_config.get("dtype"):
            return self.__class__({**self.codec_config, "dtype": str(array_spec.dtype)})
        return self


class AsTypeCodec(NumcodecsArrayArrayCodec):
    def __init__(self, codec_config: dict[str, JSON] = {}) -> None:
        super().__init__(codec_id="astype", codec_config=codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return ArraySpec(
            chunk_spec.shape,
            np.dtype(self.codec_config["encode_dtype"]),
            chunk_spec.fill_value,
        )

    def evolve(self, array_spec: ArraySpec) -> Self:
        decode_dtype = self.codec_config.get("decode_dtype")
        if str(array_spec.dtype) != decode_dtype:
            return self.__class__({**self.codec_config, "decode_dtype": str(array_spec.dtype)})
        return self


class PackbitsCodec(NumcodecsArrayArrayCodec):
    def __init__(self, codec_config: dict[str, JSON] = {}) -> None:
        super().__init__(codec_id="packbits", codec_config=codec_config)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return ArraySpec(
            (1 + math.ceil(product(chunk_spec.shape) / 8),),
            np.dtype("uint8"),
            chunk_spec.fill_value,
        )

    def validate(self, array_metadata: ArrayMetadata) -> None:
        if array_metadata.dtype != np.dtype("bool"):
            raise ValueError(f"Packbits filter requires bool dtype. Got {array_metadata.dtype}.")


# bytes-to-bytes codecs
register_codec(f"{CODEC_PREFIX}blosc", make_bytes_bytes_codec("blosc"))
register_codec(f"{CODEC_PREFIX}lz4", make_bytes_bytes_codec("lz4"))
register_codec(f"{CODEC_PREFIX}zstd", make_bytes_bytes_codec("zstd"))
register_codec(f"{CODEC_PREFIX}zlib", make_bytes_bytes_codec("zlib"))
register_codec(f"{CODEC_PREFIX}gzip", make_bytes_bytes_codec("gzip"))
register_codec(f"{CODEC_PREFIX}bz2", make_bytes_bytes_codec("bz2"))
register_codec(f"{CODEC_PREFIX}lzma", make_bytes_bytes_codec("lzma"))

# array-to-array codecs ("filters")
register_codec(f"{CODEC_PREFIX}delta", make_array_array_codec("delta"))
register_codec(f"{CODEC_PREFIX}fixedscaleoffset", FixedScaleOffsetCodec)
register_codec(f"{CODEC_PREFIX}quantize", QuantizeCodec)
register_codec(f"{CODEC_PREFIX}bitround", make_array_array_codec("bitround"))
register_codec(f"{CODEC_PREFIX}packbits", PackbitsCodec)
register_codec(f"{CODEC_PREFIX}astype", AsTypeCodec)

# bytes-to-bytes checksum codecs
register_codec(f"{CODEC_PREFIX}crc32", make_checksum_codec("crc32"))
register_codec(f"{CODEC_PREFIX}adler32", make_checksum_codec("crc32"))
register_codec(f"{CODEC_PREFIX}fletcher32", make_checksum_codec("crc32"))
register_codec(f"{CODEC_PREFIX}jenkins_lookup3", make_checksum_codec("crc32"))
