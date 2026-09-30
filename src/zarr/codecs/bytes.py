from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Final, Literal

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
    from zarr.core.indexing import SelectorTuple


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
        selection is applied to them. A selection that touches every row reads
        the whole chunk, as before.
        """
        rows = _selected_rows(selection, chunk_spec.shape)
        if rows is None or rows == (0, chunk_spec.shape[0]):
            chunk_bytes = await byte_getter.get(prototype=chunk_spec.prototype)
            if chunk_bytes is None:
                return None
            return self._decode_sync(chunk_bytes, chunk_spec)[selection]
        first, stop = rows
        row_items = int(np.prod(chunk_spec.shape[1:]))
        row_bytes = chunk_spec.dtype.to_native_dtype().itemsize * row_items
        chunk_bytes = await byte_getter.get(
            prototype=chunk_spec.prototype,
            byte_range=RangeByteRequest(first * row_bytes, stop * row_bytes),
        )
        if chunk_bytes is None:
            return None
        rows_spec = replace(chunk_spec, shape=(stop - first, *chunk_spec.shape[1:]))
        return self._decode_sync(chunk_bytes, rows_spec)[_shift_rows(selection, first)]

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


def _selected_rows(selection: SelectorTuple, shape: tuple[int, ...]) -> tuple[int, int] | None:
    """The first row along axis 0 that a selection touches and one past the last.

    Returns None when the rows cannot be determined, in which case the whole
    chunk is read.
    """
    if len(shape) == 0 or not isinstance(selection, tuple) or len(selection) == 0:
        return None
    first_axis = selection[0]
    if isinstance(first_axis, slice):
        rows = range(*first_axis.indices(shape[0]))
        if len(rows) == 0:
            return None
        return min(rows[0], rows[-1]), max(rows[0], rows[-1]) + 1
    if isinstance(first_axis, int | np.integer):
        row = int(first_axis) % shape[0]
        return row, row + 1
    if isinstance(first_axis, np.ndarray):
        indices = np.nonzero(first_axis)[0] if first_axis.dtype == bool else first_axis
        if indices.size == 0:
            return None
        return int(indices.min()) % shape[0], int(indices.max()) % shape[0] + 1
    return None


def _shift_rows(selection: SelectorTuple, first: int) -> SelectorTuple:
    """The selection relative to the rows fetched, which start at row `first`."""
    first_axis = selection[0]
    if isinstance(first_axis, slice):
        start, stop, step = first_axis.start, first_axis.stop, first_axis.step
        shifted: object = slice(
            None if start is None else start - first,
            None if stop is None else stop - first,
            step,
        )
    elif isinstance(first_axis, np.ndarray) and first_axis.dtype == bool:
        shifted = first_axis[first:]
    else:
        shifted = first_axis - first
    return (shifted, *selection[1:])
