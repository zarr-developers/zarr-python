from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, overload

import numpy as np
from numcodecs.compat import ensure_bytes, ensure_ndarray_like

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    BytesBytesCodec,
    CodecJSON,
    CodecJSON_V2,
)
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
from zarr.registry import get_ndbuffer_class

if TYPE_CHECKING:
    from zarr.abc.numcodec import Numcodec
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.buffer.core import BufferPrototype
    from zarr.core.common import BaseConfig, NamedConfig, ZarrFormat


@dataclass(frozen=True)
class V2Codec(ArrayBytesCodec):
    filters: tuple[Numcodec, ...] | None
    compressor: Numcodec | None

    is_fixed_size = False

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        cdata = chunk_bytes.as_array_like()
        # decompress
        if self.compressor:
            chunk = await asyncio.to_thread(self.compressor.decode, cdata)
        else:
            chunk = cdata

        # apply filters
        if self.filters:
            for f in reversed(self.filters):
                chunk = await asyncio.to_thread(f.decode, chunk)

        # view as numpy array with correct dtype
        chunk = ensure_ndarray_like(chunk)
        # special case object dtype, because incorrect handling can lead to
        # segfaults and other bad things happening
        if chunk_spec.dtype.dtype_cls is not np.dtypes.ObjectDType:
            try:
                chunk = chunk.view(chunk_spec.dtype.to_native_dtype())
            except TypeError:
                # this will happen if the dtype of the chunk
                # does not match the dtype of the array spec i.g. if
                # the dtype of the chunk_spec is a string dtype, but the chunk
                # is an object array. In this case, we need to convert the object
                # array to the correct dtype.

                chunk = np.array(chunk).astype(chunk_spec.dtype.to_native_dtype())

        elif chunk.dtype != object:
            # If we end up here, someone must have hacked around with the filters.
            # We cannot deal with object arrays unless there is an object
            # codec in the filter chain, i.e., a filter that converts from object
            # array to something else during encoding, and converts back to object
            # array during decoding.
            raise RuntimeError("cannot read object array without object codec")

        # ensure correct chunk shape
        chunk = chunk.reshape(-1, order="A")
        chunk = chunk.reshape(chunk_spec.shape, order=chunk_spec.order)

        return get_ndbuffer_class().from_ndarray_like(chunk)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        chunk = chunk_array.as_ndarray_like()

        # ensure contiguous and correct order
        chunk = chunk.astype(chunk_spec.dtype.to_native_dtype(), order=chunk_spec.order, copy=False)

        # apply filters
        if self.filters:
            for f in self.filters:
                chunk = await asyncio.to_thread(f.encode, chunk)
        # check object encoding
        if ensure_ndarray_like(chunk).dtype == object:
            raise RuntimeError("cannot write object array without object codec")

        # compress
        if self.compressor:
            cdata = await asyncio.to_thread(self.compressor.encode, chunk)
        else:
            cdata = chunk
        cdata = ensure_bytes(cdata)
        return chunk_spec.prototype.buffer.from_bytes(cdata)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class NumcodecsWrapper:
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

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        """Fills in codec configuration parameters that can be automatically
        inferred from the array metadata.

        Parameters
        ----------
        array_spec : ArraySpec

        Returns
        -------
        Self
        """
        return self

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        """Validates that the codec configuration is compatible with the array metadata.
        Raises errors when the codec configuration is not compatible.

        Parameters
        ----------
        shape : tuple[int, ...]
            The array shape
        dtype : np.dtype[Any]
            The array data type
        chunk_grid : ChunkGrid
            The array chunk grid
        """

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
        from zarr.core.buffer.cpu import as_numpy_array_wrapper

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
