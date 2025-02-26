from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self, get_args

import numpy as np
import numpy.typing as npt

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


class Flexible:
    capacity: int


class DtypeBase(ABC, Metadata):
    name: ClassVar[str]
    numpy_character_code: ClassVar[str]
    item_size: ClassVar[int | None]
    kind: ClassVar[DataTypeFlavor]

    def __init_subclass__(cls, **kwargs: object) -> None:
        required_attrs = [
            "name",
            "numpy_character_code",
            "item_size",
            "kind",
        ]
        for attr in required_attrs:
            if not hasattr(cls, attr):
                raise ValueError(f"{attr} is a required attribute for a Zarr dtype.")

        return super().__init_subclass__(**kwargs)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name}

    @classmethod
    def from_numpy(cls, dtype: npt.DTypeLike) -> Self:
        """
        Create an instance of this dtype from a numpy dtype.

        Parameters
        ----------
        dtype : npt.DTypeLike
            The numpy dtype to create an instance from.

        Returns
        -------
        Self
            An instance of this dtype.

        Raises
        ------
        ValueError
            If the provided numpy dtype does not match this class.
        """
        if np.dtype(dtype).char != cls.numpy_character_code:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected dtype with character code == {cls.numpy_character_code}."
            )
        return cls()

    def to_numpy(self: Self, *, endianness: Endianness | None = None) -> np.dtype[Any]:
        endian_str = endianness_to_numpy_str(endianness)
        return np.dtype(endian_str + self.numpy_character_code)


@dataclass(frozen=True, kw_only=True)
class Bool(DtypeBase):
    name = "bool"
    item_size = 1
    kind = "boolean"
    numpy_character_code = "?"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.BoolDType:
        return super().to_numpy(endianness=endianness)


register_data_type(Bool)


@dataclass(frozen=True, kw_only=True)
class Int8(DtypeBase):
    name = "int8"
    item_size = 1
    kind = "numeric"
    numpy_character_code = "b"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int8DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Int8)


@dataclass(frozen=True, kw_only=True)
class UInt8(DtypeBase):
    name = "uint8"
    item_size = 2
    kind = "numeric"
    numpy_character_code = "B"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt8DType:
        return super().to_numpy(endianness=endianness)


register_data_type(UInt8)


@dataclass(frozen=True, kw_only=True)
class Int16(DtypeBase):
    name = "int16"
    item_size = 2
    kind = "numeric"
    numpy_character_code = "h"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int16DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Int16)


@dataclass(frozen=True, kw_only=True)
class UInt16(DtypeBase):
    name = "uint16"
    item_size = 2
    kind = "numeric"
    numpy_character_code = "H"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt16DType:
        return super().to_numpy(endianness=endianness)


register_data_type(UInt16)


@dataclass(frozen=True, kw_only=True)
class Int32(DtypeBase):
    name = "int32"
    item_size = 4
    kind = "numeric"
    numpy_character_code = "i"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int32DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Int32)


@dataclass(frozen=True, kw_only=True)
class UInt32(DtypeBase):
    name = "uint32"
    item_size = 4
    kind = "numeric"
    numpy_character_code = "I"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt32DType:
        return super().to_numpy(endianness=endianness)


register_data_type(UInt32)


@dataclass(frozen=True, kw_only=True)
class Int64(DtypeBase):
    name = "int64"
    item_size = 8
    kind = "numeric"
    numpy_character_code = "l"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int64DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Int64)


@dataclass(frozen=True, kw_only=True)
class UInt64(DtypeBase):
    name = "uint64"
    item_size = 8
    kind = "numeric"
    numpy_character_code = "L"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt64DType:
        return super().to_numpy(endianness=endianness)


register_data_type(UInt64)


@dataclass(frozen=True, kw_only=True)
class Float16(DtypeBase):
    name = "float16"
    item_size = 2
    kind = "numeric"
    numpy_character_code = "e"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Float16DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Float16)


@dataclass(frozen=True, kw_only=True)
class Float32(DtypeBase):
    name = "float32"
    item_size = 4
    kind = "numeric"
    numpy_character_code = "f"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Float32DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Float32)


@dataclass(frozen=True, kw_only=True)
class Float64(DtypeBase):
    name = "float64"
    item_size = 8
    kind = "numeric"
    numpy_character_code = "d"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Float64DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Float64)


@dataclass(frozen=True, kw_only=True)
class Complex64(DtypeBase):
    name = "complex64"
    item_size = 16
    kind = "numeric"
    numpy_character_code = "F"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Complex64DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Complex64)


@dataclass(frozen=True, kw_only=True)
class Complex128(DtypeBase):
    name = "complex64"
    item_size = 32
    kind = "numeric"
    numpy_character_code = "D"

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Complex128DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Complex128)


@dataclass(frozen=True, kw_only=True)
class StaticByteString(DtypeBase, Flexible):
    name = "numpy/static_byte_string"
    kind = "string"
    numpy_character_code = "S"
    item_size = 1

    def from_numpy(cls: type[Self], dtype: npt.DTypeLike) -> Self:
        dtype = np.dtype(dtype)
        if dtype.kind != cls.numpy_character_code:
            raise ValueError(f"Invalid dtype {dtype}. Expected a string dtype.")
        return cls(capacity=dtype.itemsize)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"capacity": self.capacity}}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtype[np.bytes_]:
        endianness_code = endianness_to_numpy_str(endianness)
        return np.dtype(endianness_code + self.numpy_character_code + str(self.capacity))


@dataclass(frozen=True, kw_only=True)
class StaticRawBytes(DtypeBase, Flexible):
    name = "r*"
    kind = "bytes"
    numpy_character_code = "V"
    item_size = 1

    def from_numpy(cls: type[Self], dtype: npt.DTypeLike) -> Self:
        dtype = np.dtype(dtype)
        if dtype.kind != "V":
            raise ValueError(f"Invalid dtype {dtype}. Expected a bytes dtype.")
        return cls(capacity=dtype.itemsize)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": f"r{self.capacity * 8}"}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtype[np.void]:
        endianness_code = endianness_to_numpy_str(endianness)
        return np.dtype(endianness_code + self.numpy_character_code + str(self.capacity))


register_data_type(StaticByteString)

if _NUMPY_SUPPORTS_VLEN_STRING:

    @dataclass(frozen=True, kw_only=True)
    class VlenString(DtypeBase):
        name = "numpy/vlen_string"
        kind = "string"
        numpy_character_code = "T"
        # this uses UTF-8, so the encoding of a code point varies between
        # 1 and 4 bytes
        item_size = None

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.name}

        def to_numpy(
            self, endianness: Endianness | None = "native"
        ) -> np.dtype[np.dtypes.StringDType]:
            endianness_code = endianness_to_numpy_str(endianness)
            return np.dtype(endianness_code + self.numpy_character_code)

else:

    @dataclass(frozen=True, kw_only=True)
    class VlenString(DtypeBase):
        name = "numpy/vlen_string"
        kind = "string"
        numpy_character_code = "O"
        item_size = None

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.name}

        def to_numpy(
            self, endianness: Endianness | None = "native"
        ) -> np.dtype[np.dtypes.ObjectDType]:
            endianness_code = endianness_to_numpy_str(endianness)
            return np.dtype(endianness_code + self.numpy_character_code)


register_data_type(VlenString)


@dataclass(frozen=True, kw_only=True)
class StaticUnicodeString(DtypeBase, Flexible):
    name = "numpy/static_unicode_string"
    kind = "string"
    numpy_character_code = "U"
    item_size = 4

    def from_numpy(cls: type[Self], dtype: npt.DTypeLike) -> Self:
        dtype = np.dtype(dtype)
        if dtype.kind != "U":
            raise ValueError(f"Invalid dtype {dtype}. Expected a string dtype.")
        return cls(capacity=dtype.itemsize)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"capacity": self.capacity}}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtype[np.str_]:
        endianness_code = endianness_to_numpy_str(endianness)
        return np.dtype(endianness_code + self.numpy_character_code + str(self.capacity))


register_data_type(StaticUnicodeString)


def resolve_dtype(dtype: npt.DTypeLike | DtypeBase) -> DtypeBase:
    from zarr.registry import get_data_type_from_numpy

    if isinstance(dtype, DtypeBase):
        return dtype
    cls = get_data_type_from_numpy(dtype)
    return cls.from_numpy(dtype)


register_data_type(StaticRawBytes)
