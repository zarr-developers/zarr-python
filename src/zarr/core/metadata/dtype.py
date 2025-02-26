from abc import ABC
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Self, get_args

import numpy as np

from zarr.abc.metadata import Metadata
from zarr.core.common import JSON
from zarr.core.strings import _NUMPY_SUPPORTS_VLEN_STRING
from zarr.registry import register_data_type

Endianness = Literal["little", "big", "native"]
DataTypeFlavor = Literal["boolean", "numeric", "string", "bytes"]


def endianness_to_numpy_str(endianness: Endianness | None) -> Literal[">", "<", "=", "|"]:
    match endianness:
        case "little":
            return "<"
        case "big":
            return ">"
        case "native":
            return "="
        case None:
            return "|"
    raise ValueError(
        f"Invalid endianness: {endianness}. Expected one of {get_args(endianness)} or None"
    )


class BaseDataType(ABC, Metadata):
    name: ClassVar[str]
    numpy_character_code: ClassVar[str]
    item_size: ClassVar[int | None]
    type: ClassVar[DataTypeFlavor]
    capacity: int

    def __init_subclass__(cls, **kwargs: object) -> None:
        required_attrs = [
            "name",
            "numpy_character_code",
            "item_size",
            "type",
        ]
        for attr in required_attrs:
            if not hasattr(cls, attr):
                raise ValueError(f"{attr} is a required attribute for a Zarr dtype.")

        return super().__init_subclass__(**kwargs)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name}

    def to_numpy(self: Self, *, endianness: Endianness | None = None) -> np.dtype[Any]:
        endian_str = endianness_to_numpy_str(endianness)
        return np.dtype(endian_str + self.numpy_character_code)


@dataclass(frozen=True, kw_only=True)
class Bool(BaseDataType):
    name = "bool"
    item_size = 1
    type = "boolean"
    numpy_character_code = "?"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.BoolDType:
        return super().to_numpy(endianness=endianness)


register_data_type(Bool)


@dataclass(frozen=True, kw_only=True)
class Int8(BaseDataType):
    name = "int8"
    item_size = 1
    type = "numeric"
    numpy_character_code = "b"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int8DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Int8)


@dataclass(frozen=True, kw_only=True)
class UInt8(BaseDataType):
    name = "uint8"
    item_size = 2
    type = "numeric"
    numpy_character_code = "B"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt8DType:
        return super().to_numpy(endianness=endianness)


register_data_type(UInt8)


@dataclass(frozen=True, kw_only=True)
class Int16(BaseDataType):
    name = "int16"
    item_size = 2
    type = "numeric"
    numpy_character_code = "h"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int16DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Int16)


@dataclass(frozen=True, kw_only=True)
class UInt16(BaseDataType):
    name = "uint16"
    item_size = 2
    type = "numeric"
    numpy_character_code = "H"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt16DType:
        return super().to_numpy(endianness=endianness)


register_data_type(UInt16)


@dataclass(frozen=True, kw_only=True)
class Int32(BaseDataType):
    name = "int32"
    item_size = 4
    type = "numeric"
    numpy_character_code = "i"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int32DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Int32)


@dataclass(frozen=True, kw_only=True)
class UInt32(BaseDataType):
    name = "uint32"
    item_size = 4
    type = "numeric"
    numpy_character_code = "I"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt32DType:
        return super().to_numpy(endianness=endianness)


register_data_type(UInt32)


@dataclass(frozen=True, kw_only=True)
class Int64(BaseDataType):
    name = "int64"
    item_size = 8
    type = "numeric"
    numpy_character_code = "l"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int64DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Int64)


@dataclass(frozen=True, kw_only=True)
class UInt64(BaseDataType):
    name = "uint64"
    item_size = 8
    type = "numeric"
    numpy_character_code = "L"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt64DType:
        return super().to_numpy(endianness=endianness)


register_data_type(UInt64)


@dataclass(frozen=True, kw_only=True)
class Float16(BaseDataType):
    name = "float16"
    item_size = 2
    type = "numeric"
    numpy_character_code = "e"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Float16DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Float16)


@dataclass(frozen=True, kw_only=True)
class Float32(BaseDataType):
    name = "float32"
    item_size = 4
    type = "numeric"
    numpy_character_code = "f"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Float32DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Float32)


@dataclass(frozen=True, kw_only=True)
class Float64(BaseDataType):
    name = "float64"
    item_size = 8
    type = "numeric"
    numpy_character_code = "d"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Float64DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Float64)


@dataclass(frozen=True, kw_only=True)
class Complex64(BaseDataType):
    name = "complex64"
    item_size = 16
    type = "numeric"
    numpy_character_code = "F"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Complex64DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Complex64)


@dataclass(frozen=True, kw_only=True)
class Complex128(BaseDataType):
    name = "complex64"
    item_size = 32
    type = "numeric"
    numpy_character_code = "D"
    capacity: int = field(default=1, init=False)

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Complex128DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Complex128)


@dataclass(frozen=True, kw_only=True)
class StaticByteString(BaseDataType):
    name = "numpy/static_byte_string"
    type = "string"
    numpy_character_code = "S"
    item_size = 1
    capacity: int

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"capacity": self.capacity}}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtype[np.bytes_]:
        endianness_code = endianness_to_numpy_str(endianness)
        return np.dtype(endianness_code + self.numpy_character_code + str(self.capacity))


register_data_type(StaticByteString)

if _NUMPY_SUPPORTS_VLEN_STRING:

    @dataclass(frozen=True, kw_only=True)
    class VlenString(BaseDataType):
        name = "numpy/vlen_string"
        type = "string"
        numpy_character_code = "T"
        item_size = None
        capacity: int

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.name, "configuration": {"capacity": self.capacity}}

        def to_numpy(
            self, endianness: Endianness | None = "native"
        ) -> np.dtype[np.dtypes.StringDType]:
            endianness_code = endianness_to_numpy_str(endianness)
            return np.dtype(endianness_code + self.numpy_character_code)

else:

    @dataclass(frozen=True, kw_only=True)
    class VlenString(BaseDataType):
        name = "numpy/vlen_string"
        type = "string"
        numpy_character_code = "O"
        item_size = None
        capacity: int

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.name, "configuration": {"capacity": self.capacity}}

        def to_numpy(
            self, endianness: Endianness | None = "native"
        ) -> np.dtype[np.dtypes.ObjectDType]:
            endianness_code = endianness_to_numpy_str(endianness)
            return np.dtype(endianness_code + self.numpy_character_code)


register_data_type(VlenString)


@dataclass(frozen=True, kw_only=True)
class StaticUnicodeString(BaseDataType):
    name = "numpy/static_unicode_string"
    type = "string"
    numpy_character_code = "U"
    item_size = 4
    capacity: int

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"capacity": self.capacity}}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtype[np.str_]:
        endianness_code = endianness_to_numpy_str(endianness)
        return np.dtype(endianness_code + self.numpy_character_code + str(self.capacity))


register_data_type(StaticUnicodeString)


@dataclass(frozen=True, kw_only=True)
class StaticRawBytes(BaseDataType):
    name = "r*"
    type = "bytes"
    numpy_character_code = "V"
    item_size = 1
    capacity: int

    def to_dict(self) -> dict[str, JSON]:
        return {"name": f"r{self.capacity * 8}"}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtype[np.void]:
        endianness_code = endianness_to_numpy_str(endianness)
        return np.dtype(endianness_code + self.numpy_character_code + str(self.capacity))


def parse_dtype(dtype: npt.DtypeLike | BaseDataType) -> BaseDataType:
    from zarr.registry import get_data_type_from_numpy

    if isinstance(dtype, BaseDataType):
        return dtype
    return get_data_type_from_numpy(dtype)


register_data_type(StaticRawBytes)
