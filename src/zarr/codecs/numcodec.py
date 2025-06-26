"""
Utilities for interfacing with the numcodecs library.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, overload

import numpy as np
from typing_extensions import Protocol, runtime_checkable

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec, CodecConfig_V2
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer.core import Buffer, BufferPrototype, NDArrayLike, NDBuffer
from zarr.core.buffer.cpu import as_numpy_array_wrapper

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import BaseConfig, NamedConfig, ZarrFormat

BufferOrNDArray = Buffer | np.ndarray[tuple[int, ...], np.dtype[np.generic]] | NDArrayLike


def resolve_numcodec(config: CodecConfig_V2[str]) -> Numcodec:
    import numcodecs

    return numcodecs.get_codec(config)  # type: ignore[no-any-return]


@runtime_checkable
class Numcodec(Protocol):
    """
    A protocol that models the ``numcodecs.abc.Codec`` interface.
    """

    codec_id: str

    def encode(self, buf: BufferOrNDArray) -> BufferOrNDArray: ...

    def decode(
        self, buf: BufferOrNDArray, out: BufferOrNDArray | None = None
    ) -> BufferOrNDArray: ...

    def get_config(self) -> CodecConfig_V2[str]: ...

    @classmethod
    def from_config(cls, config: CodecConfig_V2[str]) -> Self: ...


@dataclass(frozen=True, kw_only=True)
class NumcodecsAdapter:
    _codec: Numcodec

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecConfig_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, BaseConfig]: ...

    def to_json(
        self, zarr_format: ZarrFormat
    ) -> CodecConfig_V2[str] | NamedConfig[str, BaseConfig]:
        if zarr_format == 2:
            return self._codec.get_config()
        elif zarr_format == 3:
            config = self._codec.get_config()
            config_no_id = {k: v for k, v in config.items() if k != "id"}
            return {"name": config["id"], "configuration": config_no_id}
        raise ValueError(f"Unsupported zarr format: {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_v2(cls, data: Mapping[str, object]) -> Self:
        return cls(_codec=resolve_numcodec(data))  # type: ignore[arg-type]

    @classmethod
    def _from_json_v3(cls, data: Mapping[str, object]) -> Self:
        raise NotImplementedError(
            "This class does not support creating instances from JSON data for Zarr format 3."
        )

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


class NumcodecsBytesBytesCodec(NumcodecsAdapter, BytesBytesCodec):
    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper,
            self._codec.decode,
            chunk_data,
            chunk_spec.prototype,
        )

    def _encode(self, chunk_bytes: Buffer, prototype: BufferPrototype) -> Buffer:
        encoded = self._codec.encode(chunk_bytes.as_array_like())
        if isinstance(encoded, np.ndarray):  # Required for checksum codecs
            return prototype.buffer.from_bytes(encoded.tobytes())
        return prototype.buffer.from_bytes(encoded)

    async def _encode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(self._encode, chunk_data, chunk_spec.prototype)


@dataclass(kw_only=True, frozen=True)
class NumcodecsArrayCodec(NumcodecsAdapter, ArrayArrayCodec):
    async def _decode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.decode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))  # type: ignore[union-attr]

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out)  # type: ignore[arg-type]


@dataclass(kw_only=True, frozen=True)
class NumcodecsArrayBytesCodec(NumcodecsAdapter, ArrayBytesCodec):
    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_bytes = chunk_data.to_bytes()
        out = await asyncio.to_thread(self._codec.decode, chunk_bytes)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.buffer.from_bytes(out)
