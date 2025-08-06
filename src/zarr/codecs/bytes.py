from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Final, Literal, NotRequired, TypedDict, TypeGuard, overload

import numpy as np
from typing_extensions import ReadOnly

from zarr.abc.codec import ArrayBytesCodec, CodecJSON
from zarr.core.buffer import Buffer, NDArrayLike, NDBuffer
from zarr.core.common import JSON, NamedConfig, ZarrFormat
from zarr.core.dtype.common import HasEndianness
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec


class Endian(Enum):
    """
    Enum for endian type used by bytes codec.
    """

    big = "big"
    little = "little"


# TODO: unify with the endianness defined in core.dtype.common
EndiannessStr = Literal["little", "big"]
ENDIANNESS_STR: Final = "little", "big"

default_system_endian = sys.byteorder


class BytesConfig(TypedDict):
    endian: NotRequired[EndiannessStr]


class BytesJSON_V2(BytesConfig):
    """
    JSON representation of the bytes codec for zarr v2.
    """

    id: ReadOnly[Literal["bytes"]]


BytesJSON_V3 = NamedConfig[Literal["bytes"], BytesConfig] | Literal["bytes"]


def parse_endianness(data: object) -> EndiannessStr:
    if data in ENDIANNESS_STR:
        return data  # type: ignore [return-value]
    raise ValueError(f"Invalid endianness: {data!r}. Expected one of {ENDIANNESS_STR}")


def check_json_v2(data: CodecJSON) -> TypeGuard[BytesJSON_V2]:
    return (
        isinstance(data, Mapping)
        and set(data.keys()) in ({"id", "endian"}, {"id"})
        and data["id"] == "bytes"
    )


def check_json_v3(data: CodecJSON) -> TypeGuard[BytesJSON_V3]:
    return data == "bytes" or (
        (
            isinstance(data, Mapping)
            and set(data.keys()) in ({"name"}, {"name", "configuration"})
            and data["name"] == "bytes"
        )
        and isinstance(data.get("configuration", {}), Mapping)
        and set(data.get("configuration", {}).keys()) in ({"endian"}, set())
    )


@dataclass(frozen=True)
class BytesCodec(ArrayBytesCodec):
    is_fixed_size = True

    endian: EndiannessStr | None

    def __init__(self, *, endian: EndiannessStr | str | None = default_system_endian) -> None:
        endian_parsed = None if endian is None else parse_endianness(endian)

        object.__setattr__(self, "endian", endian_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls.from_json(data, zarr_format=3)

    def to_dict(self) -> dict[str, JSON]:
        return self.to_json(zarr_format=3)

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        if check_json_v2(data):
            return cls(endian=data.get("endian", None))
        raise ValueError(f"Invalid JSON: {data}")

    @classmethod
    def _from_json_v3(cls, data: CodecJSON) -> Self:
        if check_json_v3(data):
            # Three different representations of the exact same codec...
            if data in ("bytes", {"name": "bytes"}, {"name": "bytes", "configuration": {}}):
                return cls()
            else:
                return cls(endian=data["configuration"].get("endian", None))
        raise ValueError(f"Invalid JSON: {data}")

    @overload
    def to_json(self, zarr_format: Literal[2]) -> BytesJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> BytesJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> BytesJSON_V2 | BytesJSON_V3:
        if zarr_format == 2:
            if self.endian is not None:
                return {
                    "id": "bytes",
                    "endian": self.endian,
                }
            return {"id": "bytes"}
        elif zarr_format == 3:
            if self.endian is not None:
                return {
                    "name": "bytes",
                    "configuration": {"endian": self.endian},
                }
            return {"name": "bytes"}
        raise ValueError(
            f"Unsupported Zarr format {zarr_format}. Expected 2 or 3."
        )  # pragma: no cover

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if not isinstance(array_spec.dtype, HasEndianness):
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
        # TODO: remove endianness enum in favor of literal union
        endian = self.endian if self.endian is not None else None
        if isinstance(chunk_spec.dtype, HasEndianness) and endian is not None:
            dtype = replace(chunk_spec.dtype, endianness=endian).to_native_dtype()  # type: ignore[call-arg]
        else:
            dtype = chunk_spec.dtype.to_native_dtype()
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
        if (
            chunk_array.dtype.itemsize > 1
            and self.endian is not None
            and self.endian != chunk_array.byteorder
        ):
            # type-ignore is a numpy bug
            # see https://github.com/numpy/numpy/issues/26473
            new_dtype = chunk_array.dtype.newbyteorder(self.endian)  # type: ignore[arg-type]
            chunk_array = chunk_array.astype(new_dtype)

        nd_array = chunk_array.as_ndarray_like()
        # Flatten the nd-array (only copy if needed) and reinterpret as bytes
        nd_array = nd_array.ravel().view(dtype="B")
        return chunk_spec.prototype.buffer.from_array_like(nd_array)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


register_codec("bytes", BytesCodec)

# compatibility with earlier versions of ZEP1
register_codec("endian", BytesCodec)
