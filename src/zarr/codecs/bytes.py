from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import numpy as np

from zarr.abc.codec import ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin
from zarr.abc.store import RangeByteRequest
from zarr.codecs._deprecated_enum import _coerce_enum_input, _DeprecatedStrEnumMeta
from zarr.core.common import JSON, parse_named_configuration
from zarr.core.dtype.common import HasEndianness
from zarr.core.dtype.npy.structured import Struct

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.store import ByteGetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.indexing import Selector, SelectorTuple


EndianLiteral = Literal["little", "big"]
"""Byte order of multi-byte numeric data."""

ENDIAN: Final = ("little", "big")


class Endian(metaclass=_DeprecatedStrEnumMeta):
    """
    Deprecated. Pass a literal string (`"little"` or `"big"`) directly to
    `BytesCodec` instead.
    """

    _members: ClassVar[dict[str, str]] = {"little": "little", "big": "big"}


def _parse_endian(data: object) -> EndianLiteral:
    if isinstance(data, str) and data in ENDIAN:
        return data  # type: ignore[return-value]
    raise ValueError(f"endian must be one of {list(ENDIAN)!r}. Got {data!r}.")


@dataclass(frozen=True)
class BytesCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    """bytes codec"""

    is_fixed_size = True

    endian: EndianLiteral | None

    def __init__(self, *, endian: Endian | EndianLiteral | None = sys.byteorder) -> None:
        if endian is None:
            endian_parsed: EndianLiteral | None = None
        else:
            coerced = _coerce_enum_input(endian, "endian", "BytesCodec")
            endian_parsed = _parse_endian(coerced)

        object.__setattr__(self, "endian", endian_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "bytes", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        configuration_parsed.setdefault("endian", None)
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        if self.endian is None:
            return {"name": "bytes"}
        else:
            return {"name": "bytes", "configuration": {"endian": self.endian}}

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if isinstance(array_spec.dtype, Struct):
            if array_spec.dtype.has_multi_byte_fields():
                if self.endian is None:
                    warnings.warn(
                        "Missing 'endian' for structured dtype with multi-byte fields. "
                        "Assuming little-endian for legacy compatibility.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return replace(self, endian="little")
            else:
                if self.endian is not None:
                    return replace(self, endian=None)
        elif not isinstance(array_spec.dtype, HasEndianness):
            if self.endian is not None:
                return replace(self, endian=None)
        elif self.endian is None:
            raise ValueError(
                "The `endian` configuration needs to be specified for multi-byte data types."
            )
        return self

    def _decode_sync(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        endian_str = self.endian
        dtype = chunk_spec.dtype.to_native_dtype()
        # The byte order of the stored data is set by this codec's `endian`
        # configuration; the byte order of the decoded array is set by the array's
        # data type. The two are independent: the raw bytes are viewed with a dtype
        # in the stored byte order, then converted to the declared dtype if needed.
        if isinstance(chunk_spec.dtype, HasEndianness):
            view_dtype = replace(chunk_spec.dtype, endianness=endian_str).to_native_dtype()  # type: ignore[call-arg]
        elif isinstance(chunk_spec.dtype, Struct) and endian_str is not None:
            # Per the struct data type spec, all multi-byte fields are stored in the
            # byte order configured on this codec.
            view_dtype = dtype.newbyteorder(endian_str)
        else:
            view_dtype = dtype
        as_array_like = chunk_bytes.as_array_like()
        chunk_array = chunk_spec.prototype.nd_buffer.from_ndarray_like(
            as_array_like.view(dtype=view_dtype)  # type: ignore[attr-defined]
        )
        if view_dtype != dtype:
            # This byte-swapping conversion copies the chunk. The dtype inequality
            # guard keeps the common case, where the stored and declared byte orders
            # already match, on the zero-copy view path above.
            chunk_array = chunk_array.astype(dtype)

        # ensure correct chunk shape
        if chunk_array.shape != chunk_spec.shape:
            chunk_array = chunk_array.reshape(
                chunk_spec.shape,
            )
        return chunk_array

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._decode_sync(chunk_bytes, chunk_spec)

    async def _decode_partial_single(
        self,
        byte_getter: ByteGetter,
        selection: SelectorTuple,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        """Read only the part of an uncompressed chunk that a selection needs.

        The chunk is stored in C order, so each row along its first axis is a
        contiguous run of bytes. The rows from the first to the last one the
        selection touches are fetched with a single range request, and the
        selection is applied to them. When the selection touches a single row,
        the same is done within that row along the next axis, and so on, so
        that `arr[t, y0:y1, x0:x1]` fetches rows `y0` to `y1` of frame `t`. A
        selection that touches every row reads the whole chunk, as before.
        """
        window = _block_window(selection, chunk_spec.shape)
        chunk_bytes = await byte_getter.get(
            prototype=chunk_spec.prototype, byte_range=_block_byte_range(window, chunk_spec)
        )
        return self._decode_block(chunk_bytes, selection, window, chunk_spec)

    def _decode_partial_sync(
        self,
        byte_getter: Any,
        selection: SelectorTuple,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        """Sync equivalent of `_decode_partial_single`."""
        window = _block_window(selection, chunk_spec.shape)
        chunk_bytes = byte_getter.get_sync(
            prototype=chunk_spec.prototype, byte_range=_block_byte_range(window, chunk_spec)
        )
        return self._decode_block(chunk_bytes, selection, window, chunk_spec)

    def _decode_block(
        self,
        chunk_bytes: Buffer | None,
        selection: SelectorTuple,
        window: tuple[int, tuple[int, ...], SelectorTuple] | None,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        """Decode the bytes fetched for a block window and apply the selection to them."""
        if chunk_bytes is None:
            return None
        if window is None:
            return self._decode_sync(chunk_bytes, chunk_spec)[selection]
        offset, block_shape, block_selection = window
        itemsize = _itemsize(chunk_spec)
        start = offset * itemsize
        end = start + math.prod(block_shape) * itemsize
        chunk_size = math.prod(chunk_spec.shape) * itemsize
        # A store may not honor the byte range it was given. The length of what
        # it sent tells which bytes those are.
        if len(chunk_bytes) == end - start:
            pass
        elif len(chunk_bytes) == chunk_size:
            # The whole chunk, as an HTTP server that ignores the Range header sends.
            chunk_bytes = chunk_bytes[start:end]
        elif len(chunk_bytes) == chunk_size - start:
            # Everything from the start of the range to the end of the chunk.
            chunk_bytes = chunk_bytes[: end - start]
        else:
            raise ValueError(
                f"Requested bytes {start} to {end} of a {chunk_size} byte chunk, "
                f"but the store returned {len(chunk_bytes)} bytes."
            )
        block_spec = replace(chunk_spec, shape=block_shape)
        return self._decode_sync(chunk_bytes, block_spec)[block_selection]

    def _encode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        if chunk_array.dtype.itemsize > 1 and self.endian is not None:
            # Compare full dtypes rather than the top-level byteorder: numpy reports
            # byteorder '|' for structured dtypes even when their fields are
            # byte-order-sensitive, so newbyteorder is the only reliable way to
            # detect (and normalize) a byte-order mismatch.
            new_dtype = chunk_array.dtype.newbyteorder(self.endian)
            if new_dtype != chunk_array.dtype:
                chunk_array = chunk_array.astype(new_dtype)

        nd_array = chunk_array.as_ndarray_like()
        # Flatten the nd-array (only copy if needed) and reinterpret as bytes
        nd_array = nd_array.ravel().view(dtype="B")
        return chunk_spec.prototype.buffer.from_array_like(nd_array)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


def _block_window(
    selection: SelectorTuple, shape: tuple[int, ...]
) -> tuple[int, tuple[int, ...], SelectorTuple] | None:
    """The contiguous block of a C-order chunk that a selection touches.

    Along the first axis the block spans the first to the last row the
    selection touches. If that is a single row, the block is narrowed in the
    same way along the next axis, and so on. The axes after that are whole.

    Returns the offset of the block from the start of the chunk in items, the
    shape of the block, and the selection shifted so that it indexes the block.
    The block has as many axes as the chunk; the axes it was narrowed to a
    single index along have length 1. Returns None when the block is the whole
    chunk or cannot be determined, in which case the whole chunk is read.
    """
    if not isinstance(selection, tuple):
        return None
    offset = 0
    block_shape = list(shape)
    shifted = list(selection)
    for axis, (selector, size) in enumerate(zip(selection, shape, strict=False)):
        window = _axis_window(selector, size)
        if window is None:
            break
        first, stop, shifted[axis] = window
        offset += first * math.prod(shape[axis + 1 :])
        block_shape[axis] = stop - first
        if stop - first > 1:
            break
    if tuple(block_shape) == shape:
        return None
    return offset, tuple(block_shape), tuple(shifted)


def _axis_window(selector: Selector, size: int) -> tuple[int, int, Selector] | None:
    """The indices along one axis that a selector touches, and the selector relative to them.

    Returns the first index, one past the last index, and the selector shifted
    so that it indexes an axis holding only those indices. Returns None when
    the indices cannot be determined.
    """
    shifted: Selector
    if isinstance(selector, slice):
        indices = range(*selector.indices(size))
        if len(indices) == 0 or indices.step < 0:
            return None
        first, stop = indices[0], indices[-1] + 1
        shifted = slice(0, stop - first, indices.step)
    elif isinstance(selector, int | np.integer):
        first = int(selector) % size
        stop = first + 1
        shifted = 0
    elif isinstance(selector, np.ndarray):
        if selector.dtype == bool:
            positions = np.nonzero(selector)[0]
        else:
            positions = selector % size
        if positions.size == 0:
            return None
        first, stop = int(positions.min()), int(positions.max()) + 1
        shifted = selector[first:stop] if selector.dtype == bool else positions - first
    else:
        return None
    return first, stop, shifted


def _itemsize(chunk_spec: ArraySpec) -> int:
    """The number of bytes in one item of a chunk."""
    return chunk_spec.dtype.to_native_dtype().itemsize


def _block_byte_range(
    window: tuple[int, tuple[int, ...], SelectorTuple] | None, chunk_spec: ArraySpec
) -> RangeByteRequest | None:
    """The byte range holding a block window, or None to read the whole chunk."""
    if window is None:
        return None
    offset, block_shape, _ = window
    itemsize = _itemsize(chunk_spec)
    start = offset * itemsize
    return RangeByteRequest(start, start + math.prod(block_shape) * itemsize)
