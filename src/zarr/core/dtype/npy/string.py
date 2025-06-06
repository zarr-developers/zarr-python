from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Self, TypedDict, TypeGuard, cast, overload

import numpy as np

from zarr.core.common import NamedConfig
from zarr.core.dtype.common import (
    HasEndianness,
    HasItemSize,
    HasLength,
    HasObjectCodec,
    v3_unstable_dtype_warning,
)
from zarr.core.dtype.npy.common import (
    EndiannessNumpy,
    check_json_str,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
)
from zarr.core.dtype.wrapper import DTypeJSON_V2, DTypeJSON_V3, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat
    from zarr.core.dtype.wrapper import TBaseDType

_NUMPY_SUPPORTS_VLEN_STRING = hasattr(np.dtypes, "StringDType")


class LengthBytesConfig(TypedDict):
    length_bytes: int


# TDO: Fix this terrible name
FixedLengthBytesJSONV3 = NamedConfig[Literal["fixed_length_bytes"], LengthBytesConfig]


# TODO: Fix this terrible name
FixedLengthUTF32JSONV3 = NamedConfig[Literal["fixed_length_utf32"], LengthBytesConfig]


@dataclass(frozen=True, kw_only=True)
class FixedLengthUTF32(
    ZDType[np.dtypes.StrDType[int], np.str_], HasEndianness, HasLength, HasItemSize
):
    dtype_cls = np.dtypes.StrDType
    _zarr_v3_name: ClassVar[Literal["fixed_length_utf32"]] = "fixed_length_utf32"
    code_point_bytes: ClassVar[int] = 4  # utf32 is 4 bytes per code point

    @classmethod
    def _from_native_dtype_unchecked(cls, dtype: TBaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(
            length=dtype.itemsize // (cls.code_point_bytes),
            endianness=endianness_from_numpy_str(byte_order),
        )

    def to_native_dtype(self) -> np.dtypes.StrDType[int]:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls(self.length).newbyteorder(byte_order)

    @classmethod
    def _check_json_v2(cls, data: JSON, object_codec_id: str | None = None) -> TypeGuard[str]:
        """
        Check that the input is a valid JSON representation of a numpy S dtype.
        """
        return isinstance(data, str) and re.match(r"^[><]U\d+$", data) is not None

    @classmethod
    def _check_json_v3(cls, data: JSON) -> TypeGuard[FixedLengthUTF32JSONV3]:
        return (
            isinstance(data, dict)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == cls._zarr_v3_name
            and "configuration" in data
            and isinstance(data["configuration"], dict)
            and set(data["configuration"].keys()) == {"length_bytes"}
            and isinstance(data["configuration"]["length_bytes"], int)
        )

    @overload
    def to_json(self, zarr_format: Literal[2]) -> str: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> FixedLengthUTF32JSONV3: ...

    def to_json(self, zarr_format: ZarrFormat) -> str | FixedLengthUTF32JSONV3:
        if zarr_format == 2:
            return self.to_native_dtype().str
        elif zarr_format == 3:
            v3_unstable_dtype_warning(self)
            return {
                "name": self._zarr_v3_name,
                "configuration": {"length_bytes": self.length * self.code_point_bytes},
            }
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unchecked(
        cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
    ) -> Self:
        if zarr_format == 2:
            return cls.from_native_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(length=data["configuration"]["length_bytes"] // cls.code_point_bytes)  # type: ignore[index, call-overload]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_scalar(self) -> np.str_:
        return np.str_("")

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return str(data)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.str_:
        if check_json_str(data):
            return self.to_native_dtype().type(data)
        raise TypeError(f"Invalid type: {data}. Expected a string.")  # pragma: no cover

    def _check_scalar(self, data: object) -> TypeGuard[str | np.str_ | bytes | int]:
        # this is generous for backwards compatibility
        return isinstance(data, str | np.str_ | bytes | int)

    def _cast_scalar_unchecked(self, data: object) -> np.str_:
        # We explicitly truncate the result because of the following numpy behavior:
        # >>> x = np.dtype('U3').type('hello world')
        # >>> x
        # np.str_('hello world')
        # >>> x.dtype
        # dtype('U11')

        if isinstance(data, int):
            return self.to_native_dtype().type(str(data)[: self.length])
        else:
            return self.to_native_dtype().type(data[: self.length])  # type: ignore[index]

    @property
    def item_size(self) -> int:
        return self.length * self.code_point_bytes


def check_vlen_string_json_scalar(data: object) -> TypeGuard[int | str | float]:
    """
    This function checks the type of JSON-encoded variable length strings. It is generous for
    backwards compatibility, as zarr-python v2 would use ints for variable length strings
    fill values
    """
    return isinstance(data, int | str | float)


if _NUMPY_SUPPORTS_VLEN_STRING:

    @dataclass(frozen=True, kw_only=True)
    class VariableLengthUTF8(ZDType[np.dtypes.StringDType, str], HasObjectCodec):  # type: ignore[type-var]
        dtype_cls = np.dtypes.StringDType
        _zarr_v3_name: ClassVar[Literal["variable_length_utf8"]] = "variable_length_utf8"
        object_codec_id = "vlen-utf8"

        @classmethod
        def _from_native_dtype_unchecked(cls, dtype: TBaseDType) -> Self:
            return cls()

        def to_native_dtype(self) -> np.dtypes.StringDType:
            return self.dtype_cls()

        @classmethod
        def _check_json_v2(
            cls, data: JSON, *, object_codec_id: str | None = None
        ) -> TypeGuard[Literal["|O"]]:
            """
            Check that the input is a valid JSON representation of a numpy O dtype, and that the
            object codec id is appropriate for variable-length UTF-8 strings.
            """
            return data == "|O" and object_codec_id == cls.object_codec_id

        @classmethod
        def _check_json_v3(cls, data: JSON) -> TypeGuard[Literal["variable_length_utf8"]]:
            return data == cls._zarr_v3_name

        @overload
        def to_json(self, zarr_format: Literal[2]) -> Literal["|O"]: ...
        @overload
        def to_json(self, zarr_format: Literal[3]) -> Literal["variable_length_utf8"]: ...

        def to_json(self, zarr_format: ZarrFormat) -> Literal["|O", "variable_length_utf8"]:
            if zarr_format == 2:
                # Note: unlike many other numpy data types, we don't serialize the .str attribute
                # of the data type to JSON. This is because Zarr was using `|O` for strings before the
                # numpy variable length string data type existed, and we want to be consistent with
                # that practice
                return "|O"
            elif zarr_format == 3:
                v3_unstable_dtype_warning(self)
                return self._zarr_v3_name
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

        @classmethod
        def _from_json_unchecked(
            cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
        ) -> Self:
            return cls()

        def default_scalar(self) -> str:
            return ""

        def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> str:
            if self._check_scalar(data):
                return data
            raise TypeError(f"Invalid type: {data}. Expected a string.")

        def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
            if not check_vlen_string_json_scalar(data):
                raise TypeError(f"Invalid type: {data}. Expected a string or number.")
            return str(data)

        def _check_scalar(self, data: object) -> TypeGuard[str]:
            return isinstance(data, str)

        def _cast_scalar_unchecked(self, data: object) -> str:
            return str(data)

else:
    # Numpy pre-2 does not have a variable length string dtype, so we use the Object dtype instead.
    @dataclass(frozen=True, kw_only=True)
    class VariableLengthUTF8(ZDType[np.dtypes.ObjectDType, str], HasObjectCodec):  # type: ignore[no-redef]
        dtype_cls = np.dtypes.ObjectDType
        _zarr_v3_name: ClassVar[Literal["variable_length_utf8"]] = "variable_length_utf8"
        object_codec_id = "vlen-utf8"

        @classmethod
        def _from_native_dtype_unchecked(cls, dtype: TBaseDType) -> Self:
            return cls()

        def to_native_dtype(self) -> np.dtypes.ObjectDType:
            return self.dtype_cls()

        @classmethod
        def _check_json_v2(
            cls, data: JSON, *, object_codec_id: str | None = None
        ) -> TypeGuard[Literal["|O"]]:
            """
            Check that the input is a valid JSON representation of a numpy O dtype, and that the
            object codec id is appropriate for variable-length UTF-8 strings.
            """
            return data == "|O" and object_codec_id == cls.object_codec_id

        @classmethod
        def _check_json_v3(cls, data: JSON) -> TypeGuard[Literal["variable_length_utf8"]]:
            return data == cls._zarr_v3_name

        @overload
        def to_json(self, zarr_format: Literal[2]) -> Literal["|O"]: ...

        @overload
        def to_json(self, zarr_format: Literal[3]) -> Literal["variable_length_utf8"]: ...

        def to_json(self, zarr_format: ZarrFormat) -> Literal["|O", "variable_length_utf8"]:
            if zarr_format == 2:
                return "|O"
            elif zarr_format == 3:
                v3_unstable_dtype_warning(self)
                return self._zarr_v3_name
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

        @classmethod
        def _from_json_unchecked(
            cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
        ) -> Self:
            return cls()

        def default_scalar(self) -> str:
            return ""

        def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> str:
            if self._check_scalar(data):
                return data
            raise TypeError(f"Invalid type: {data}. Expected a string.")

        def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
            """
            Strings pass through
            """
            if not check_json_str(data):
                raise TypeError(f"Invalid type: {data}. Expected a string.")
            return data

        def _check_scalar(self, data: object) -> TypeGuard[str]:
            return isinstance(data, str)

        def _cast_scalar_unchecked(self, data: object) -> str:
            return str(data)
