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
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, overload
from warnings import warn

import numpy as np

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    BytesBytesCodec,
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
)
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON, NamedConfig, ZarrFormat, product
from zarr.dtype import UInt8, ZDType, parse_dtype
from zarr.errors import ZarrUserWarning
from zarr.registry import get_numcodec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr.abc.numcodec import Numcodec
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer


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

    def to_dict(self) -> NamedConfig[str, Mapping[str, object]]:
        return {"name": self._codec_id, "configuration": self.codec_config}

    @cached_property
    def _codec(self) -> Numcodec:
        return get_numcodec({"id": self._codec_id, **self.codec_config})  # type: ignore[arg-type]

    @classmethod
    def _from_json_v2(cls, data: CodecJSON_V2[str]) -> Self:
        return cls(**data)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON_V3) -> Self:
        if isinstance(data, str):
            return cls()
        return cls(**data["configuration"])

    @classmethod
    def from_json(cls, data: CodecJSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls._from_json_v2(data)
        elif zarr_format == 3:
            return cls._from_json_v3(data)
        raise ValueError(
            f"Unsupported Zarr format {zarr_format}. Expected 2 or 3."
        )  # pragma: no cover

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...

    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        codec_id = self._codec_id
        codec_config = {k: v for k, v in self.codec_config.items() if k != "id"}
        if zarr_format == 2:
            return {"id": codec_id, **codec_config}
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


class LZ4(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.lz4"
    _codec_id = "lz4"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)


class Zstd(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.zstd"
    _codec_id = "zstd"


class Zlib(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.zlib"
    _codec_id = "zlib"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)


class GZip(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.gzip"
    _codec_id = "gzip"


class BZ2(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.bz2"
    _codec_id = "bz2"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)


class LZMA(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.lzma"
    _codec_id = "lzma"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)


class Shuffle(_NumcodecsBytesBytesCodec):
    codec_name = "numcodecs.shuffle"
    _codec_id = "shuffle"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Shuffle:
        if self.codec_config.get("elementsize") is None:
            dtype = array_spec.dtype.to_native_dtype()
            return Shuffle(**{**self.codec_config, "elementsize": dtype.itemsize})
        return self  # pragma: no cover


# array-to-array codecs ("filters")
class Delta(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.delta"
    _codec_id = "delta"

    def __init__(self, **codec_config: Any) -> None:
        if "codec_config" in codec_config:
            raise ValueError("The argument 'codec_config' is not supported.")
        super().__init__(**codec_config)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            dtype = parse_dtype(np.dtype(astype), zarr_format=3)  # type: ignore[call-overload]
            return replace(chunk_spec, dtype=dtype)
        return chunk_spec


class BitRound(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.bitround"
    _codec_id = "bitround"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)


class FixedScaleOffset(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.fixedscaleoffset"
    _codec_id = "fixedscaleoffset"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            dtype = parse_dtype(np.dtype(astype), zarr_format=3)  # type: ignore[call-overload]
            return replace(chunk_spec, dtype=dtype)
        return chunk_spec

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> FixedScaleOffset:
        if self.codec_config.get("dtype") is None:
            dtype = array_spec.dtype.to_native_dtype()
            return FixedScaleOffset(**{**self.codec_config, "dtype": str(dtype)})
        return self


class Quantize(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.quantize"
    _codec_id = "quantize"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if self.codec_config.get("dtype") is None:
            dtype = array_spec.dtype.to_native_dtype()
            return Quantize(**{**self.codec_config, "dtype": str(dtype)})
        return self


class PackBits(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.packbits"
    _codec_id = "packbits"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)

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


class AsType(_NumcodecsArrayArrayCodec):
    codec_name = "numcodecs.astype"
    _codec_id = "astype"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        dtype = parse_dtype(np.dtype(self.codec_config["encode_dtype"]), zarr_format=3)  # type: ignore[arg-type]
        return replace(chunk_spec, dtype=dtype)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> AsType:
        if self.codec_config.get("decode_dtype") is None:
            # TODO: remove these coverage exemptions the correct way, i.e. with tests
            dtype = array_spec.dtype.to_native_dtype()  # pragma: no cover
            return AsType(**{**self.codec_config, "decode_dtype": str(dtype)})  # pragma: no cover
        return self


# bytes-to-bytes checksum codecs
class _NumcodecsChecksumCodec(_NumcodecsBytesBytesCodec):
    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4  # pragma: no cover


class CRC32(_NumcodecsChecksumCodec):
    codec_name = "numcodecs.crc32"
    _codec_id = "crc32"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)


class CRC32C(_NumcodecsChecksumCodec):
    codec_name = "numcodecs.crc32c"
    _codec_id = "crc32c"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)


class Adler32(_NumcodecsChecksumCodec):
    codec_name = "numcodecs.adler32"
    _codec_id = "adler32"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)


class Fletcher32(_NumcodecsChecksumCodec):
    codec_name = "numcodecs.fletcher32"
    _codec_id = "fletcher32"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)


class JenkinsLookup3(_NumcodecsChecksumCodec):
    codec_name = "numcodecs.jenkins_lookup3"
    _codec_id = "jenkins_lookup3"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)


# array-to-bytes codecs
class PCodec(_NumcodecsArrayBytesCodec):
    codec_name = "numcodecs.pcodec"
    _codec_id = "pcodec"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)


class ZFPY(_NumcodecsArrayBytesCodec):
    codec_name = "numcodecs.zfpy"
    _codec_id = "zfpy"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, Mapping[str, object]]: ...
    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecJSON_V2[str] | NamedConfig[str, Mapping[str, object]]:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)
