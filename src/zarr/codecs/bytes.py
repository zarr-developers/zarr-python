from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from zarr.abc.codec import ArrayBytesCodec
from zarr.array_spec import ArraySpec
from zarr.buffer import Buffer, NDArrayLike, NDBuffer
from zarr.codecs.registry import register_codec
from zarr.common import JSON, parse_enum, parse_named_configuration

if TYPE_CHECKING:
    from typing_extensions import Self


class Endian(Enum):
    big = "big"
    little = "little"


default_system_endian = Endian(sys.byteorder)


@dataclass(frozen=True)
class BytesCodec(ArrayBytesCodec):
    is_fixed_size = True

    endian: Endian | None

    def __init__(self, *, endian: Endian | str | None = default_system_endian) -> None:
        endian_parsed = None if endian is None else parse_enum(endian, Endian)

        object.__setattr__(self, "endian", endian_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "bytes", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        if self.endian is None:
            return {"name": "bytes"}
        else:
            return {"name": "bytes", "configuration": {"endian": self.endian}}

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if array_spec.dtype.itemsize == 0:
            if self.endian is not None:
                return replace(self, endian=None)
        elif self.endian is None:
            raise ValueError(
                "The `endian` configuration needs to be specified for multi-byte data types."
            )
        return self

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        assert isinstance(chunk_bytes, Buffer)
        if chunk_spec.dtype.itemsize > 0:
            if self.endian == Endian.little:
                prefix = "<"
            else:
                prefix = ">"
            dtype = np.dtype(f"{prefix}{chunk_spec.dtype.str[1:]}")
        else:
            dtype = np.dtype(f"|{chunk_spec.dtype.str[1:]}")

        as_array_like = chunk_bytes.as_array_like()
        if isinstance(as_array_like, NDArrayLike):
            as_nd_array_like = as_array_like
        else:
            as_nd_array_like = np.asanyarray(as_array_like)
        chunk_array = chunk_spec.prototype.nd_buffer.from_ndarray_like(
            as_nd_array_like.view(dtype=dtype)
        )

        # ensure correct chunk shape
        if chunk_array.shape != chunk_spec.shape:
            chunk_array = chunk_array.reshape(
                chunk_spec.shape,
            )
        return chunk_array

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        assert isinstance(chunk_array, NDBuffer)
        if chunk_array.dtype.itemsize > 1:
            if self.endian is not None and self.endian != chunk_array.byteorder:
                # type-ignore is a numpy bug
                # see https://github.com/numpy/numpy/issues/26473
                new_dtype = chunk_array.dtype.newbyteorder(self.endian.name)  # type: ignore[arg-type]
                chunk_array = chunk_array.astype(new_dtype)

        nd_array = chunk_array.as_ndarray_like()
        # Flatten the nd-array (only copy if needed) and reinterpret as bytes
        nd_array = nd_array.ravel().view(dtype="b")
        return chunk_spec.prototype.buffer.from_array_like(nd_array)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


register_codec("bytes", BytesCodec)

# compatibility with earlier versions of ZEP1
register_codec("endian", BytesCodec)
