from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
import sys

from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np

from zarr.codecs.mixins import ArrayBytesCodecBatchMixin
from zarr.codecs.registry import register_codec
from zarr.common import parse_enum, parse_named_configuration

if TYPE_CHECKING:
    from zarr.common import JSON, ArraySpec, BytesLike
    from typing_extensions import Self


class Endian(Enum):
    big = "big"
    little = "little"


default_system_endian = Endian(sys.byteorder)


@dataclass(frozen=True)
class BytesCodec(ArrayBytesCodecBatchMixin):
    is_fixed_size = True

    endian: Optional[Endian]

    def __init__(self, *, endian: Union[Endian, str, None] = default_system_endian) -> None:
        endian_parsed = None if endian is None else parse_enum(endian, Endian)

        object.__setattr__(self, "endian", endian_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "bytes", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, JSON]:
        if self.endian is None:
            return {"name": "bytes"}
        else:
            return {"name": "bytes", "configuration": {"endian": self.endian}}

    def evolve(self, array_spec: ArraySpec) -> Self:
        if array_spec.dtype.itemsize == 0:
            if self.endian is not None:
                return replace(self, endian=None)
        elif self.endian is None:
            raise ValueError(
                "The `endian` configuration needs to be specified for multi-byte data types."
            )
        return self

    def _get_byteorder(self, array: np.ndarray) -> Endian:
        if array.dtype.byteorder == "<":
            return Endian.little
        elif array.dtype.byteorder == ">":
            return Endian.big
        else:
            return default_system_endian

    async def decode_single(
        self,
        chunk_bytes: BytesLike,
        chunk_spec: ArraySpec,
    ) -> np.ndarray:
        if chunk_spec.dtype.itemsize > 0:
            if self.endian == Endian.little:
                prefix = "<"
            else:
                prefix = ">"
            dtype = np.dtype(f"{prefix}{chunk_spec.dtype.str[1:]}")
        else:
            dtype = np.dtype(f"|{chunk_spec.dtype.str[1:]}")
        chunk_array = np.frombuffer(chunk_bytes, dtype)

        # ensure correct chunk shape
        if chunk_array.shape != chunk_spec.shape:
            chunk_array = chunk_array.reshape(
                chunk_spec.shape,
            )
        return chunk_array

    async def encode_single(
        self,
        chunk_array: np.ndarray,
        _chunk_spec: ArraySpec,
    ) -> Optional[BytesLike]:
        if chunk_array.dtype.itemsize > 1:
            byteorder = self._get_byteorder(chunk_array)
            if self.endian is not None and self.endian != byteorder:
                new_dtype = chunk_array.dtype.newbyteorder(self.endian.name)
                chunk_array = chunk_array.astype(new_dtype)
        return chunk_array.tobytes()

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


register_codec("bytes", BytesCodec)

# compatibility with earlier versions of ZEP1
register_codec("endian", BytesCodec)
