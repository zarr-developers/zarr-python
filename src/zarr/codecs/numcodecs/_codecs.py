"""
This module provides compatibility for [numcodecs][] in Zarr version 3.

These codecs were previously defined in [numcodecs][], and have now been moved to `zarr`.

>>> import numpy as np
>>> import zarr
>>> import zarr.codecs.numcodecs as numcodecs
>>>
>>> array = zarr.create_array(
...   store="data.zarr",
...   shape=(1024, 1024),
...   chunks=(64, 64),
...   dtype="uint32",
...   filters=[numcodecs.Delta(dtype="uint32")],
...   compressors=[numcodecs.BZ2(level=5)])
>>> array[:] = np.arange(np.prod(array.shape), dtype=array.dtype).reshape(*array.shape)

!!! note
    Please note that the codecs in [zarr.codecs.numcodecs][] are not part of the Zarr version
    3 specification. Using these codecs might cause interoperability issues with other Zarr
    implementations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    cast,
    overload,
)
from warnings import warn

import numpy as np

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    BytesBytesCodec,
)
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import (
    JSON,
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    ZarrFormat,
    _check_codecjson_v2,
)
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
    _codec: Numcodec
    codec_config: Mapping[str, Any]

    def __init__(self, **codec_config: Any) -> None:
        object.__setattr__(
            self,
            "_codec",
            get_numcodec(
                {"id": self._codec_id, **{k: v for k, v in codec_config.items() if k != "id"}}
            ),
        )  # type: ignore[typeddict-item]
        object.__setattr__(self, "codec_config", self._codec.get_config())

    def to_dict(self) -> dict[str, JSON]:
        return cast(dict[str, JSON], self.to_json(zarr_format=3))

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

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


class _NumcodecsArrayArrayCodec(_NumcodecsCodec, ArrayArrayCodec):
    async def _decode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.decode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


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

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


# bytes-to-bytes checksum codecs
class _NumcodecsChecksumCodec(_NumcodecsBytesBytesCodec):
    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length + 4  # pragma: no cover
