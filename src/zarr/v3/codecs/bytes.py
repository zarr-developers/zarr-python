from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum

from typing import TYPE_CHECKING, Dict, Literal, Union

import numpy as np

from zarr.v3.abc.codec import ArrayBytesCodec, Codec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import JSON, ArraySpec, parse_enum

if TYPE_CHECKING:
    from zarr.v3.common import ArraySpec, BytesLike, RuntimeConfiguration
    from zarr.v3.metadata import ArrayMetadata
    from typing_extensions import Self
    from typing import Dict, Optional


class Endian(Enum):
    big = "big"
    little = "little"


def parse_endian(data: JSON) -> Literal["big", "little"]:
    return parse_enum(data, Endian)


def parse_name(data: JSON) -> Literal["bytes"]:
    if data == "bytes":
        return data
    msg = f"Expected 'bytes', got {data} instead."
    raise ValueError(msg)


@dataclass(frozen=True)
class BytesCodec(ArrayBytesCodec):
    is_fixed_size = True

    endian: Optional[Endian]

    def __init__(self, *, endian=Endian.little) -> None:
        endian_parsed = None if endian is None else parse_endian(endian)

        object.__setattr__(self, "endian", endian_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        parse_name(data["name"])
        return BytesCodec(**data.get("configuration", {}))

    def to_dict(self) -> Dict[str, JSON]:
        if self.endian is None:
            return {"name": "bytes"}
        else:
            return {"name": "bytes", "configuration": {"endian": self.endian.name}}

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
            import sys

            return sys.byteorder

    async def decode(
        self,
        chunk_bytes: BytesLike,
        chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        if chunk_spec.dtype.itemsize > 0:
            if self.endian == "little":
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

    async def encode(
        self,
        chunk_array: np.ndarray,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        if chunk_array.dtype.itemsize > 1:
            byteorder = self._get_byteorder(chunk_array)
            if self.endian != byteorder:
                new_dtype = chunk_array.dtype.newbyteorder(self.endian.name)
                chunk_array = chunk_array.astype(new_dtype)
        return chunk_array.tobytes()

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


register_codec("bytes", BytesCodec)

# compatibility with earlier versions of ZEP1
register_codec("endian", BytesCodec)
