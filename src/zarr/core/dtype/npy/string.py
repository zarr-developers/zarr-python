from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self, TypeGuard, cast

import numpy as np

from zarr.core.dtype.common import HasEndianness, HasItemSize, HasLength
from zarr.core.dtype.npy.common import (
    EndiannessNumpy,
    check_json_str,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
)
from zarr.core.dtype.wrapper import ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat
    from zarr.core.dtype.wrapper import TBaseDType

_NUMPY_SUPPORTS_VLEN_STRING = hasattr(np.dtypes, "StringDType")


@dataclass(frozen=True, kw_only=True)
class FixedLengthASCII(ZDType[np.dtypes.BytesDType[int], np.bytes_], HasLength, HasItemSize):
    dtype_cls = np.dtypes.BytesDType
    _zarr_v3_name = "numpy.fixed_length_ascii"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
        return cls(length=dtype.itemsize)

    def to_dtype(self) -> np.dtypes.BytesDType[int]:
        return self.dtype_cls(self.length)

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        """
        Check that the input is a valid JSON representation of a numpy S dtype.
        """
        if zarr_format == 2:
            # match |S1, |S2, etc
            return isinstance(data, str) and re.match(r"^\|S\d+$", data) is not None
        elif zarr_format == 3:
            return (
                isinstance(data, dict)
                and set(data.keys()) == {"name", "configuration"}
                and data["name"] == cls._zarr_v3_name
                and isinstance(data["configuration"], dict)
                and "length_bytes" in data["configuration"]
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return {
                "name": self._zarr_v3_name,
                "configuration": {"length_bytes": self.length},
            }
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(length=data["configuration"]["length_bytes"])  # type: ignore[arg-type, index, call-overload]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.bytes_:
        return np.bytes_(b"")

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(data).decode("ascii")  # type: ignore[arg-type]

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.bytes_:
        if check_json_str(data):
            return self.to_dtype().type(base64.standard_b64decode(data.encode("ascii")))
        raise TypeError(f"Invalid type: {data}. Expected a string.")  # pragma: no cover

    def check_value(self, data: object) -> bool:
        # this is generous for backwards compatibility
        return isinstance(data, np.bytes_ | str | bytes | int)

    def _cast_value_unsafe(self, value: object) -> np.bytes_:
        return self.to_dtype().type(value)

    @property
    def item_size(self) -> int:
        return self.length


@dataclass(frozen=True, kw_only=True)
class FixedLengthUTF32(
    ZDType[np.dtypes.StrDType[int], np.str_], HasEndianness, HasLength, HasItemSize
):
    dtype_cls = np.dtypes.StrDType
    _zarr_v3_name = "numpy.fixed_length_utf32"
    code_point_bytes: ClassVar[int] = 4  # utf32 is 4 bytes per code point

    @classmethod
    def _from_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(
            length=dtype.itemsize // (cls.code_point_bytes),
            endianness=endianness_from_numpy_str(byte_order),
        )

    def to_dtype(self) -> np.dtypes.StrDType[int]:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls(self.length).newbyteorder(byte_order)

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        """
        Check that the input is a valid JSON representation of a numpy S dtype.
        """
        if zarr_format == 2:
            # match >U1, <U2, etc
            return isinstance(data, str) and re.match(r"^[><]U\d+$", data) is not None
        elif zarr_format == 3:
            return (
                isinstance(data, dict)
                and set(data.keys()) == {"name", "configuration"}
                and data["name"] == cls._zarr_v3_name
                and "configuration" in data
                and isinstance(data["configuration"], dict)
                and set(data["configuration"].keys()) == {"length_bytes"}
                and isinstance(data["configuration"]["length_bytes"], int)
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return {
                "name": self._zarr_v3_name,
                "configuration": {"length_bytes": self.length * self.code_point_bytes},
            }
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(length=data["configuration"]["length_bytes"] // cls.code_point_bytes)  # type: ignore[arg-type, index, call-overload, operator]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.str_:
        return np.str_("")

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return str(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.str_:
        if check_json_str(data):
            return self.to_dtype().type(data)
        raise TypeError(f"Invalid type: {data}. Expected a string.")  # pragma: no cover

    def check_value(self, data: object) -> bool:
        # this is generous for backwards compatibility
        return isinstance(data, str | np.str_ | bytes | int)

    def _cast_value_unsafe(self, data: object) -> np.str_:
        return self.to_dtype().type(data)

    @property
    def item_size(self) -> int:
        return self.length * self.code_point_bytes


if _NUMPY_SUPPORTS_VLEN_STRING:

    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(ZDType[np.dtypes.StringDType, str]):  # type: ignore[type-var]
        dtype_cls = np.dtypes.StringDType
        _zarr_v3_name = "numpy.variable_length_utf8"

        @classmethod
        def _from_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
            return cls()

        def to_dtype(self) -> np.dtypes.StringDType:
            return self.dtype_cls()

        @classmethod
        def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
            """
            Check that the input is a valid JSON representation of a numpy string dtype.
            """
            if zarr_format == 2:
                # TODO: take the entire metadata document in here, and
                # check the compressors / filters for vlen-utf8
                # Note that we are checking for the object dtype name.
                return data == "|O"
            elif zarr_format == 3:
                return data == cls._zarr_v3_name
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

        def to_json(self, zarr_format: ZarrFormat) -> JSON:
            if zarr_format == 2:
                # Note: unlike many other numpy data types, we don't serialize the .str attribute
                # of the data type to JSON. This is because Zarr was using `|O` for strings before the
                # numpy variable length string data type existed, and we want to be consistent with
                # that practice
                return "|O"
            elif zarr_format == 3:
                return self._zarr_v3_name
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

        @classmethod
        def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
            return cls()

        def default_value(self) -> str:
            return ""

        def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
            return str(data)

        def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
            if not check_json_str(data):
                raise TypeError(f"Invalid type: {data}. Expected a string.")
            return data

        def check_value(self, data: object) -> bool:
            return isinstance(data, str)

        def _cast_value_unsafe(self, data: object) -> str:
            return str(data)

else:
    # Numpy pre-2 does not have a variable length string dtype, so we use the Object dtype instead.
    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(ZDType[np.dtypes.ObjectDType, str]):  # type: ignore[no-redef]
        dtype_cls = np.dtypes.ObjectDType
        _zarr_v3_name = "numpy.variable_length_utf8"

        @classmethod
        def _from_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
            return cls()

        def to_dtype(self) -> np.dtypes.ObjectDType:
            return self.dtype_cls()

        @classmethod
        def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
            """
            Check that the input is a valid JSON representation of a numpy O dtype.
            """
            if zarr_format == 2:
                # TODO: take the entire metadata document in here, and
                # check the compressors / filters for vlen-utf8
                return data == "|O"
            elif zarr_format == 3:
                return data == cls._zarr_v3_name
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

        def to_json(self, zarr_format: ZarrFormat) -> JSON:
            if zarr_format == 2:
                return self.to_dtype().str
            elif zarr_format == 3:
                return self._zarr_v3_name
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

        @classmethod
        def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
            return cls()

        def default_value(self) -> str:
            return ""

        def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
            return data  # type: ignore[return-value]

        def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
            """
            Strings pass through
            """
            if not check_json_str(data):
                raise TypeError(f"Invalid type: {data}. Expected a string.")
            return data

        def check_value(self, data: object) -> bool:
            return isinstance(data, str)

        def _cast_value_unsafe(self, data: object) -> str:
            return str(data)
