from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeGuard

import numpy as np
from numcodecs.compat import ensure_bytes, ensure_ndarray_like

from zarr.abc.codec import ArrayBytesCodec, Numcodec
from zarr.registry import get_ndbuffer_class

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer


def _is_numcodec(obj: object) -> TypeGuard[Numcodec]:
    """
    Check if the given object implements the Numcodec protocol.

    The @runtime_checkable decorator does not allow issubclass checks for protocols with non-method
    members (i.e., attributes), so we use this function to manually check for the presence of the
    required attributes and methods on a given object.
    """
    return _is_numcodec_cls(type(obj))


def _is_numcodec_cls(obj: object) -> TypeGuard[type[Numcodec]]:
    """
    Check if the given object is a class implements the Numcodec protocol.

    The @runtime_checkable decorator does not allow issubclass checks for protocols with non-method
    members (i.e., attributes), so we use this function to manually check for the presence of the
    required attributes and methods on a given object.
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
            chunk = await asyncio.to_thread(self.compressor.decode, cdata)  # type: ignore[arg-type]
        else:
            chunk = cdata  # type: ignore[assignment]

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

                chunk = np.array(chunk).astype(chunk_spec.dtype.to_native_dtype())  # type: ignore[assignment]

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
                chunk = await asyncio.to_thread(f.encode, chunk)  # type: ignore[arg-type]

        # check object encoding
        if ensure_ndarray_like(chunk).dtype == object:
            raise RuntimeError("cannot write object array without object codec")

        # compress
        if self.compressor:
            cdata = await asyncio.to_thread(self.compressor.encode, chunk)  # type: ignore[arg-type]
        else:
            cdata = chunk  # type: ignore[assignment]

        cdata = ensure_bytes(cdata)
        return chunk_spec.prototype.buffer.from_bytes(cdata)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
