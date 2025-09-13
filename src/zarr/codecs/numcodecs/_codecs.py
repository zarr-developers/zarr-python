"""
This module provides compatibility for :py:mod:`numcodecs` in Zarr version 3.

These codecs were previously defined in :py:mod:`numcodecs`, and have now been moved to `zarr`.

>>> import zarr
>>> import zarr.codecs.numcodecs as numcodecs
>>>
>>> array = zarr.create_array(
...   store="data.zarr",
...   shape=(1024, 1024),
...   chunks=(64, 64),
...   dtype="uint32",
...   filters=[numcodecs.zarr3.Delta()],
...   compressors=[numcodecs.zarr3.BZ2(level=5)])
>>> array[:] = np.arange(*array.shape).astype(array.dtype)

.. note::

    Please note that the codecs in :py:mod:`zarr.codecs.numcodecs` are not part of the Zarr version
    3 specification. Using these codecs might cause interoperability issues with other Zarr
    implementations.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, replace
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Self,
    TypedDict,
    cast,
    overload,
)
from warnings import warn

import numpy as np
from typing_extensions import ReadOnly

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    BytesBytesCodec,
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    _check_codecjson_v2,
)
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON, NamedConfig, NamedRequiredConfig, ZarrFormat, product
from zarr.dtype import UInt8, ZDType, parse_dtype
from zarr.errors import ZarrUserWarning
from zarr.registry import get_numcodec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr.abc.numcodec import Numcodec
    from zarr.codecs.blosc import BloscJSON_V2, BloscJSON_V3
    from zarr.codecs.crc32c_ import Crc32cConfig, Crc32cJSON_V2, Crc32cJSON_V3
    from zarr.codecs.gzip import GZipConfig, GZipJSON_V2, GZipJSON_V3
    from zarr.codecs.zstd import ZstdConfig_V3, ZstdJSON_V2, ZstdJSON_V3
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer


# TypedDict definitions for V2 and V3 JSON representations


# Configuration classes for codec parameters
class LZ4Config(TypedDict):
    acceleration: NotRequired[int]


class ZlibConfig(TypedDict):
    level: NotRequired[int]


class BZ2Config(TypedDict):
    level: NotRequired[int]


class LZMAConfig(TypedDict):
    format: NotRequired[int]
    check: NotRequired[int]
    preset: NotRequired[int]
    filters: NotRequired[list[dict[str, Any]]]


class ShuffleConfig(TypedDict):
    elementsize: NotRequired[int]


class LZ4JSON_V2(LZ4Config):
    """JSON representation of LZ4 codec for Zarr V2."""

    id: ReadOnly[Literal["lz4"]]


class LZ4JSON_V3(NamedRequiredConfig[Literal["lz4"], LZ4Config]):
    """JSON representation of LZ4 codec for Zarr V3."""


class ZlibJSON_V2(ZlibConfig):
    """JSON representation of Zlib codec for Zarr V2."""

    id: ReadOnly[Literal["zlib"]]


class ZlibJSON_V3(NamedRequiredConfig[Literal["zlib"], ZlibConfig]):
    """JSON representation of Zlib codec for Zarr V3."""


class BZ2JSON_V2(BZ2Config):
    """JSON representation of BZ2 codec for Zarr V2."""

    id: ReadOnly[Literal["bz2"]]


class BZ2JSON_V3(NamedRequiredConfig[Literal["bz2"], BZ2Config]):
    """JSON representation of BZ2 codec for Zarr V3."""


class LZMAJSON_V2(LZMAConfig):
    """JSON representation of LZMA codec for Zarr V2."""

    id: ReadOnly[Literal["lzma"]]


class LZMAJSON_V3(NamedRequiredConfig[Literal["lzma"], LZMAConfig]):
    """JSON representation of LZMA codec for Zarr V3."""


class ShuffleJSON_V2(ShuffleConfig):
    """JSON representation of Shuffle codec for Zarr V2."""

    id: ReadOnly[Literal["shuffle"]]


class ShuffleJSON_V3(NamedRequiredConfig[Literal["shuffle"], ShuffleConfig]):
    """JSON representation of Shuffle codec for Zarr V3."""


# Array-to-array codec configuration classes
class DeltaConfig(TypedDict):
    dtype: str
    astype: NotRequired[str]


class BitRoundConfig(TypedDict):
    keepbits: int


class FixedScaleOffsetConfig(TypedDict):
    dtype: NotRequired[str]
    scale: NotRequired[float]
    offset: NotRequired[float]
    astype: NotRequired[str]


class QuantizeConfig(TypedDict):
    digits: int
    dtype: NotRequired[str]


class PackBitsConfig(TypedDict):
    pass  # PackBits has no configuration parameters


class AsTypeConfig(TypedDict):
    encode_dtype: str
    decode_dtype: NotRequired[str]


# Array-to-array codec JSON representations
class DeltaJSON_V2(DeltaConfig):
    """JSON representation of Delta codec for Zarr V2."""

    id: ReadOnly[Literal["delta"]]


class DeltaJSON_V3(NamedRequiredConfig[Literal["delta"], DeltaConfig]):
    """JSON representation of Delta codec for Zarr V3."""


class BitRoundJSON_V2(BitRoundConfig):
    """JSON representation of BitRound codec for Zarr V2."""

    id: ReadOnly[Literal["bitround"]]


class BitRoundJSON_V3(NamedRequiredConfig[Literal["bitround"], BitRoundConfig]):
    """JSON representation of BitRound codec for Zarr V3."""


class FixedScaleOffsetJSON_V2(FixedScaleOffsetConfig):
    """JSON representation of FixedScaleOffset codec for Zarr V2."""

    id: ReadOnly[Literal["fixedscaleoffset"]]


class FixedScaleOffsetJSON_V3(
    NamedRequiredConfig[Literal["fixedscaleoffset"], FixedScaleOffsetConfig]
):
    """JSON representation of FixedScaleOffset codec for Zarr V3."""


class QuantizeJSON_V2(QuantizeConfig):
    """JSON representation of Quantize codec for Zarr V2."""

    id: ReadOnly[Literal["quantize"]]


class QuantizeJSON_V3(NamedRequiredConfig[Literal["quantize"], QuantizeConfig]):
    """JSON representation of Quantize codec for Zarr V3."""


class PackBitsJSON_V2(PackBitsConfig):
    """JSON representation of PackBits codec for Zarr V2."""

    id: ReadOnly[Literal["packbits"]]


class PackBitsJSON_V3(NamedRequiredConfig[Literal["packbits"], PackBitsConfig]):
    """JSON representation of PackBits codec for Zarr V3."""


class AsTypeJSON_V2(AsTypeConfig):
    """JSON representation of AsType codec for Zarr V2."""

    id: ReadOnly[Literal["astype"]]


class AsTypeJSON_V3(NamedRequiredConfig[Literal["astype"], AsTypeConfig]):
    """JSON representation of AsType codec for Zarr V3."""


# Checksum codec JSON representations
class Crc32Config(TypedDict):
    """Configuration parameters for CRC32 codec."""


class Crc32JSON_V2(Crc32Config):
    """JSON representation of CRC32 codec for Zarr V2."""

    id: ReadOnly[Literal["crc32"]]


class Crc32JSON_V3(NamedConfig[Literal["crc32"], Crc32Config]):
    """JSON representation of CRC32 codec for Zarr V3."""


class Adler32Config(TypedDict):
    """Configuration parameters for Adler32 codec."""


class Adler32JSON_V2(Adler32Config):
    """JSON representation of Adler32 codec for Zarr V2."""

    id: ReadOnly[Literal["adler32"]]


class Adler32JSON_V3(NamedConfig[Literal["adler32"], Adler32Config]):
    """JSON representation of Adler32 codec for Zarr V3."""


class Fletcher32Config(TypedDict):
    """Configuration parameters for Fletcher32 codec."""


class Fletcher32JSON_V2(Fletcher32Config):
    """JSON representation of Fletcher32 codec for Zarr V2."""

    id: ReadOnly[Literal["fletcher32"]]


class Fletcher32JSON_V3(NamedRequiredConfig[Literal["fletcher32"], Fletcher32Config]):
    """JSON representation of Fletcher32 codec for Zarr V3."""


class JenkinsLookup3Config(TypedDict):
    """Configuration parameters for JenkinsLookup3 codec."""


class JenkinsLookup3JSON_V2(JenkinsLookup3Config):
    """JSON representation of JenkinsLookup3 codec for Zarr V2."""

    id: ReadOnly[Literal["jenkins_lookup3"]]


class JenkinsLookup3JSON_V3(NamedRequiredConfig[Literal["jenkins_lookup3"], JenkinsLookup3Config]):
    """JSON representation of JenkinsLookup3 codec for Zarr V3."""


# Array-to-bytes codec JSON representations
class PCodecConfig(TypedDict):
    """Configuration parameters for PCodec codec."""

    level: NotRequired[int]
    delta_encoding_order: NotRequired[int]


class PCodecJSON_V2(PCodecConfig):
    """JSON representation of PCodec codec for Zarr V2."""

    id: ReadOnly[Literal["pcodec"]]


class PCodecJSON_V3(NamedRequiredConfig[Literal["pcodec"], PCodecConfig]):
    """JSON representation of PCodec codec for Zarr V3."""


class ZFPYConfig(TypedDict):
    """Configuration parameters for ZFPY codec."""

    mode: NotRequired[int]
    rate: NotRequired[float]
    precision: NotRequired[int]
    tolerance: NotRequired[float]


class ZFPYJSON_V2(ZFPYConfig):
    """JSON representation of ZFPY codec for Zarr V2."""

    id: ReadOnly[Literal["zfpy"]]


class ZFPYJSON_V3(NamedRequiredConfig[Literal["zfpy"], ZFPYConfig]):
    """JSON representation of ZFPY codec for Zarr V3."""


def _warn_unstable_specification(obj: _NumcodecsCodec) -> None:
    warn(
        f"The codec {type(obj)} does not have a stable specification. "
        "Data saved with this codec may not be supported by other Zarr implementations. "
        "The API for this codec may change in future versions of Zarr.",
        category=ZarrUserWarning,
        stacklevel=2,
    )


@dataclass(frozen=True)
class _NumcodecsCodec:
    codec_name: str
    _codec_id: ClassVar[str]
    codec_config: Mapping[str, Any]

    def __init__(self, **codec_config: Any) -> None:
        object.__setattr__(self, "codec_config", codec_config)

    def to_dict(self) -> dict[str, JSON]:
        return cast(dict[str, JSON], self.to_json(zarr_format=3))

    @cached_property
    def _codec(self) -> Numcodec:
        return get_numcodec({"id": self._codec_id, **self.codec_config})  # type: ignore[typeddict-item]

    @classmethod
    def _from_json_v2(cls, data: CodecJSON_V2) -> Self:
        return cls(**data)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON_V3) -> Self:
        if isinstance(data, str):
            return cls()
        return cls(**data.get("configuration", {}))

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if _check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> CodecJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> CodecJSON_V2 | CodecJSON_V3:
        codec_id = self._codec_id
        codec_config = {k: v for k, v in self.codec_config.items() if k != "id"}
        if zarr_format == 2:
            return {"id": codec_id, **codec_config}  # type: ignore[return-value, typeddict-item]
        else:
            return {"name": codec_id, "configuration": codec_config}


class _NumcodecsBytesBytesCodec(_NumcodecsCodec, BytesBytesCodec):
    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper,
            self._codec.decode,
            chunk_data,
            chunk_spec.prototype,
        )

    def _encode(self, chunk_data: Buffer, prototype: BufferPrototype) -> Buffer:
        encoded = self._codec.encode(chunk_data.as_array_like())
        if isinstance(encoded, np.ndarray):  # Required for checksum codecs
            return prototype.buffer.from_bytes(encoded.tobytes())
        return prototype.buffer.from_bytes(encoded)

    async def _encode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(self._encode, chunk_data, chunk_spec.prototype)


class _NumcodecsArrayArrayCodec(_NumcodecsCodec, ArrayArrayCodec):
    async def _decode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.decode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out)


class _NumcodecsArrayBytesCodec(_NumcodecsCodec, ArrayBytesCodec):
    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_bytes = chunk_data.to_bytes()
        out = await asyncio.to_thread(self._codec.decode, chunk_bytes)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.buffer.from_bytes(out)


# bytes-to-bytes codecs
class Blosc(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.blosc"
    _codec_id = "blosc"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> BloscJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> BloscJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> BloscJSON_V2 | BloscJSON_V3:
        return super().to_json(zarr_format)  # type: ignore[return-value]


class LZ4(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.lz4"
    _codec_id = "lz4"
    codec_config: LZ4Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> LZ4JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> LZ4JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> LZ4JSON_V2 | LZ4JSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


class Zstd(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.zstd"
    _codec_id = "zstd"
    codec_config: ZstdConfig_V3

    @overload
    def to_json(self, zarr_format: Literal[2]) -> ZstdJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> ZstdJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> ZstdJSON_V2 | ZstdJSON_V3:
        return super().to_json(zarr_format)  # type: ignore[return-value]


class Zlib(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.zlib"
    _codec_id = "zlib"
    codec_config: ZlibConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> ZlibJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> ZlibJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> ZlibJSON_V2 | ZlibJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]


class GZip(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.gzip"
    _codec_id = "gzip"
    codec_config: GZipConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> GZipJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> GZipJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> GZipJSON_V2 | GZipJSON_V3:
        return super().to_json(zarr_format)  # type: ignore[return-value]


class BZ2(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.bz2"
    _codec_id = "bz2"
    codec_config: BZ2Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> BZ2JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> BZ2JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> BZ2JSON_V2 | BZ2JSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]


class LZMA(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.lzma"
    _codec_id = "lzma"
    codec_config: LZMAConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> LZMAJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> LZMAJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> LZMAJSON_V2 | LZMAJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]


class Shuffle(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.shuffle"
    _codec_id = "shuffle"
    codec_config: ShuffleConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> ShuffleJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> ShuffleJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> ShuffleJSON_V2 | ShuffleJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if self.codec_config.get("elementsize") is None:
            dtype = array_spec.dtype.to_native_dtype()
            return type(self)(**{**self.codec_config, "elementsize": dtype.itemsize})
        return self  # pragma: no cover

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


# array-to-array codecs ("filters")
class Delta(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.delta"
    _codec_id = "delta"
    codec_config: DeltaConfig

    def __init__(self, **codec_config: Any) -> None:
        if "codec_config" in codec_config:
            raise ValueError("The argument 'codec_config' is not supported.")
        super().__init__(**codec_config)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> DeltaJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> DeltaJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> DeltaJSON_V2 | DeltaJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            dtype = parse_dtype(np.dtype(astype), zarr_format=3)
            return replace(chunk_spec, dtype=dtype)
        return chunk_spec

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


class BitRound(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.bitround"
    _codec_id = "bitround"
    codec_config: BitRoundConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> BitRoundJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> BitRoundJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> BitRoundJSON_V2 | BitRoundJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


class FixedScaleOffset(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.fixedscaleoffset"
    _codec_id = "fixedscaleoffset"
    codec_config: FixedScaleOffsetConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> FixedScaleOffsetJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> FixedScaleOffsetJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> FixedScaleOffsetJSON_V2 | FixedScaleOffsetJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            dtype = parse_dtype(np.dtype(astype), zarr_format=3)
            return replace(chunk_spec, dtype=dtype)
        return chunk_spec

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if self.codec_config.get("dtype") is None:
            dtype = array_spec.dtype.to_native_dtype()
            return type(self)(**{**self.codec_config, "dtype": str(dtype)})
        return self

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


class Quantize(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.quantize"
    _codec_id = "quantize"
    codec_config: QuantizeConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> QuantizeJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> QuantizeJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> QuantizeJSON_V2 | QuantizeJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if self.codec_config.get("dtype") is None:
            dtype = array_spec.dtype.to_native_dtype()
            return type(self)(**{**self.codec_config, "dtype": str(dtype)})
        return self

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


class PackBits(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.packbits"
    _codec_id = "packbits"
    codec_config: PackBitsConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> PackBitsJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> PackBitsJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> PackBitsJSON_V2 | PackBitsJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return replace(
            chunk_spec,
            shape=(1 + math.ceil(product(chunk_spec.shape) / 8),),
            dtype=UInt8(),
        )

    # todo: remove this type: ignore when this class can be defined w.r.t.
    # a single zarr dtype API
    def validate(self, *, dtype: ZDType[Any, Any], **_kwargs: Any) -> None:
        # this is bugged and will fail
        _dtype = dtype.to_native_dtype()
        if _dtype != np.dtype("bool"):
            raise ValueError(f"Packbits filter requires bool dtype. Got {dtype}.")

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


class AsType(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.astype"
    _codec_id = "astype"
    codec_config: AsTypeConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> AsTypeJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> AsTypeJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> AsTypeJSON_V2 | AsTypeJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        dtype = parse_dtype(np.dtype(self.codec_config["encode_dtype"]), zarr_format=3)
        return replace(chunk_spec, dtype=dtype)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> AsType:
        if self.codec_config.get("decode_dtype") is None:
            # TODO: remove these coverage exemptions the correct way, i.e. with tests
            dtype = array_spec.dtype.to_native_dtype()  # pragma: no cover
            return AsType(**{**self.codec_config, "decode_dtype": str(dtype)})  # pragma: no cover
        return self

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


# bytes-to-bytes checksum codecs
class _NumcodecsChecksumCodec(_NumcodecsBytesBytesCodec):
    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4  # pragma: no cover


class CRC32(_NumcodecsChecksumCodec):
    codec_name = "numcodecs.crc32"
    _codec_id = "crc32"
    codec_config: Crc32Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Crc32JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> Crc32JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> Crc32JSON_V2 | Crc32JSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]


class CRC32C(_NumcodecsChecksumCodec):
    codec_name = "numcodecs.crc32c"
    _codec_id = "crc32c"
    codec_config: Crc32cConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Crc32cJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> Crc32cJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> Crc32cJSON_V2 | Crc32cJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]


class Adler32(_NumcodecsChecksumCodec):
    codec_name = "numcodecs.adler32"
    _codec_id = "adler32"
    codec_config: Adler32Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Adler32JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> Adler32JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> Adler32JSON_V2 | Adler32JSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]


class Fletcher32(_NumcodecsChecksumCodec):
    codec_name = "numcodecs.fletcher32"
    _codec_id = "fletcher32"
    codec_config: Fletcher32Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Fletcher32JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> Fletcher32JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> Fletcher32JSON_V2 | Fletcher32JSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]


class JenkinsLookup3(_NumcodecsChecksumCodec):
    codec_name = "numcodecs.jenkins_lookup3"
    _codec_id = "jenkins_lookup3"
    codec_config: JenkinsLookup3Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> JenkinsLookup3JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> JenkinsLookup3JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> JenkinsLookup3JSON_V2 | JenkinsLookup3JSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]


# array-to-bytes codecs
class PCodec(_NumcodecsArrayBytesCodec):
    codec_name = "numcodecs.pcodec"
    _codec_id = "pcodec"
    codec_config: PCodecConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> PCodecJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> PCodecJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> PCodecJSON_V2 | PCodecJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]


class ZFPY(_NumcodecsArrayBytesCodec):
    codec_name = "numcodecs.zfpy"
    _codec_id = "zfpy"
    codec_config: ZFPYConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> ZFPYJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> ZFPYJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> ZFPYJSON_V2 | ZFPYJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]
