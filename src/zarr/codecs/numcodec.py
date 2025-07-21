"""
Utilities for interfacing with the numcodecs library.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, TypeGuard, overload

import numpy as np
from typing_extensions import Protocol, runtime_checkable

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    BaseCodec,
    BytesBytesCodec,
    CodecJSON,
    CodecJSON_V2,
)
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer.core import Buffer, BufferPrototype, NDArrayLike, NDBuffer
from zarr.core.buffer.cpu import as_numpy_array_wrapper

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import BaseConfig, NamedConfig, ZarrFormat

BufferOrNDArray = Buffer | np.ndarray[tuple[int, ...], np.dtype[np.generic]] | NDArrayLike


def get_numcodec_class(name: str) -> type[Numcodec]:
    """Obtain a numcodec codec class by name.

    Parameters
    ----------
    name : str
        The name of the codec to get

    Returns
    -------
    codec : Codec

    Examples
    --------

    >>> import numcodecs as codecs
    >>> codec = codecs.get_codec('zlib')
    >>> codec
    Zlib(level=1)

    """
    import numcodecs.registry as numcodecs_registry

    cls = numcodecs_registry.codec_registry.get(name)
    if cls is None and name in numcodecs_registry.entries:
        cls = numcodecs_registry.entries[name].load()
        numcodecs_registry.register_codec(cls, codec_id=name)
    if cls is not None:
        return cls
    raise KeyError(name)


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

    def get_config(self) -> CodecJSON_V2[str]: ...

    @classmethod
    def from_config(cls, config: CodecJSON_V2[str]) -> Self: ...


def is_numcodec_cls(obj: object) -> TypeGuard[type[Numcodec]]:
    """
    Check if the given object implements the Numcodec protocol. Because the @runtime_checkable
    decorator does not allow issubclass checks for protocols with non-method members (i.e., attributes),
    we need to manually check for the presence of the required attributes and methods.
    """
    return (
        isinstance(obj, type)
        and hasattr(obj, "codec_id")
        and isinstance(obj.codec_id, str)
        and hasattr(obj, "encode")
        and callable(obj.encode)
        and hasattr(obj, "decode")
        and callable(obj.decode)
        and hasattr(obj, "get_config")
        and callable(obj.get_config)
        and hasattr(obj, "from_config")
        and callable(obj.from_config)
    )


@dataclass(frozen=True, kw_only=True)
class NumcodecsWrapper(BaseCodec[Buffer | NDBuffer, Buffer | NDBuffer]):
    codec: Numcodec

    @overload
    def to_json(self, zarr_format: Literal[2]) -> CodecJSON_V2[str]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> NamedConfig[str, BaseConfig]: ...

    def to_json(self, zarr_format: ZarrFormat) -> CodecJSON_V2[str] | NamedConfig[str, BaseConfig]:
        if zarr_format == 2:
            return self.codec.get_config()
        elif zarr_format == 3:
            config = self.codec.get_config()
            config_no_id = {k: v for k, v in config.items() if k != "id"}
            return {"name": config["id"], "configuration": config_no_id}
        raise ValueError(f"Unsupported zarr format: {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        raise NotADirectoryError(
            "This class does not support creating instances from JSON data for Zarr format 2."
            )

    @classmethod
    def _from_json_v3(cls, data: CodecJSON) -> Self:
        raise NotImplementedError(
            "This class does not support creating instances from JSON data for Zarr format 3."
        )

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError

    def to_array_array(self) -> NumcodecsArrayArrayCodec:
        """
        Use the ``_codec`` attribute to create a NumcodecsArrayArrayCodec.
        """
        return NumcodecsArrayArrayCodec(codec=self.codec)

    def to_bytes_bytes(self) -> NumcodecsBytesBytesCodec:
        """
        Use the ``_codec`` attribute to create a NumcodecsBytesBytesCodec.
        """
        return NumcodecsBytesBytesCodec(codec=self.codec)

    def to_array_bytes(self) -> NumcodecsArrayBytesCodec:
        """
        Use the ``_codec`` attribute to create a NumcodecsArrayBytesCodec.
        """
        return NumcodecsArrayBytesCodec(codec=self.codec)


class NumcodecsBytesBytesCodec(NumcodecsWrapper, BytesBytesCodec):
    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper,
            self.codec.decode,
            chunk_data,
            chunk_spec.prototype,
        )

    def _encode(self, chunk_bytes: Buffer, prototype: BufferPrototype) -> Buffer:
        encoded = self.codec.encode(chunk_bytes.as_array_like())
        if isinstance(encoded, np.ndarray):  # Required for checksum codecs
            return prototype.buffer.from_bytes(encoded.tobytes())
        return prototype.buffer.from_bytes(encoded)

    async def _encode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(self._encode, chunk_data, chunk_spec.prototype)


@dataclass(kw_only=True, frozen=True)
class NumcodecsArrayArrayCodec(NumcodecsWrapper, ArrayArrayCodec):
    async def _decode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self.codec.decode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))  # type: ignore[union-attr]

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self.codec.encode, chunk_ndarray)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out)  # type: ignore[arg-type]


@dataclass(kw_only=True, frozen=True)
class NumcodecsArrayBytesCodec(NumcodecsWrapper, ArrayBytesCodec):
    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        chunk_bytes = chunk_data.to_bytes()
        out = await asyncio.to_thread(self.codec.decode, chunk_bytes)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out.reshape(chunk_spec.shape))

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        out = await asyncio.to_thread(self.codec.encode, chunk_ndarray)
        return chunk_spec.prototype.buffer.from_bytes(out)
