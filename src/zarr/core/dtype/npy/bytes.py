import base64
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self, TypedDict, TypeGuard, cast, overload

import numpy as np

from zarr.core.common import JSON, NamedConfig, ZarrFormat
from zarr.core.dtype.common import HasItemSize, HasLength, HasObjectCodec, v3_unstable_dtype_warning
from zarr.core.dtype.npy.common import check_json_str
from zarr.core.dtype.wrapper import DTypeJSON_V2, DTypeJSON_V3, TBaseDType, ZDType


class FixedLengthBytesConfig(TypedDict):
    length_bytes: int


NTBytesJSONV3 = NamedConfig[Literal["null_terminated_bytes"], FixedLengthBytesConfig]
RawBytesJSONV3 = NamedConfig[Literal["raw_bytes"], FixedLengthBytesConfig]


@dataclass(frozen=True, kw_only=True)
class NullTerminatedBytes(ZDType[np.dtypes.BytesDType[int], np.bytes_], HasLength, HasItemSize):
    dtype_cls = np.dtypes.BytesDType
    _zarr_v3_name: ClassVar[Literal["null_terminated_bytes"]] = "null_terminated_bytes"

    @classmethod
    def _from_native_dtype_unchecked(cls, dtype: TBaseDType) -> Self:
        return cls(length=dtype.itemsize)

    def to_native_dtype(self) -> np.dtypes.BytesDType[int]:
        return self.dtype_cls(self.length)

    @classmethod
    def _check_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> TypeGuard[str]:
        """
        Check that the input is a valid JSON representation of a numpy S dtype.
        """
        # match |S1, |S2, etc
        return isinstance(data, str) and re.match(r"^\|S\d+$", data) is not None

    @classmethod
    def _check_json_v3(cls, data: JSON) -> TypeGuard[NTBytesJSONV3]:
        return (
            isinstance(data, dict)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == cls._zarr_v3_name
            and isinstance(data["configuration"], dict)
            and "length_bytes" in data["configuration"]
        )

    @overload
    def to_json(self, zarr_format: Literal[2]) -> str: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> NTBytesJSONV3: ...

    def to_json(self, zarr_format: ZarrFormat) -> str | NTBytesJSONV3:
        if zarr_format == 2:
            return self.to_native_dtype().str
        elif zarr_format == 3:
            v3_unstable_dtype_warning(self)
            return {
                "name": self._zarr_v3_name,
                "configuration": {"length_bytes": self.length},
            }
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unchecked(
        cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
    ) -> Self:
        if zarr_format == 2:
            return cls.from_native_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(length=data["configuration"]["length_bytes"])  # type: ignore[index, call-overload]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_scalar(self) -> np.bytes_:
        return np.bytes_(b"")

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> str:
        as_bytes = self.cast_scalar(data)
        return base64.standard_b64encode(as_bytes).decode("ascii")

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.bytes_:
        if check_json_str(data):
            return self.to_native_dtype().type(base64.standard_b64decode(data.encode("ascii")))
        raise TypeError(
            f"Invalid type: {data}. Expected a base64-encoded string."
        )  # pragma: no cover

    def _check_scalar(self, data: object) -> bool:
        # this is generous for backwards compatibility
        return isinstance(data, np.bytes_ | str | bytes | int)

    def _cast_scalar_unchecked(self, data: object) -> np.bytes_:
        # We explicitly truncate the result because of the following numpy behavior:
        # >>> x = np.dtype('S3').type('hello world')
        # >>> x
        # np.bytes_(b'hello world')
        # >>> x.dtype
        # dtype('S11')

        if isinstance(data, int):
            return self.to_native_dtype().type(str(data)[: self.length])
        else:
            return self.to_native_dtype().type(data[: self.length])  # type: ignore[index]

    @property
    def item_size(self) -> int:
        return self.length


@dataclass(frozen=True, kw_only=True)
class RawBytes(ZDType[np.dtypes.VoidDType[int], np.void], HasLength, HasItemSize):
    # np.dtypes.VoidDType is specified in an odd way in numpy
    # it cannot be used to create instances of the dtype
    # so we have to tell mypy to ignore this here
    dtype_cls = np.dtypes.VoidDType  # type: ignore[assignment]
    _zarr_v3_name: ClassVar[Literal["raw_bytes"]] = "raw_bytes"

    @classmethod
    def _from_native_dtype_unchecked(cls, dtype: TBaseDType) -> Self:
        return cls(length=dtype.itemsize)

    def to_native_dtype(self) -> np.dtypes.VoidDType[int]:
        # Numpy does not allow creating a void type
        # by invoking np.dtypes.VoidDType directly
        return cast("np.dtypes.VoidDType[int]", np.dtype(f"V{self.length}"))

    @classmethod
    def _check_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> TypeGuard[str]:
        # Check that the dtype is |V1, |V2, ...
        return isinstance(data, str) and re.match(r"^\|V\d+$", data) is not None

    @classmethod
    def _check_json_v3(cls, data: JSON) -> TypeGuard[RawBytesJSONV3]:
        return (
            isinstance(data, dict)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == cls._zarr_v3_name
            and isinstance(data["configuration"], dict)
            and set(data["configuration"].keys()) == {"length_bytes"}
        )

    @overload
    def to_json(self, zarr_format: Literal[2]) -> str: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> RawBytesJSONV3: ...

    def to_json(self, zarr_format: ZarrFormat) -> str | RawBytesJSONV3:
        if zarr_format == 2:
            return self.to_native_dtype().str
        elif zarr_format == 3:
            v3_unstable_dtype_warning(self)
            return {"name": self._zarr_v3_name, "configuration": {"length_bytes": self.length}}
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unchecked(
        cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
    ) -> Self:
        if zarr_format == 2:
            return cls.from_native_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(length=data["configuration"]["length_bytes"])  # type: ignore[index, call-overload]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _check_native_dtype(
        cls: type[Self], dtype: TBaseDType
    ) -> TypeGuard[np.dtypes.VoidDType[Any]]:
        """
        Numpy void dtype comes in two forms:
        * If the ``fields`` attribute is ``None``, then the dtype represents N raw bytes.
        * If the ``fields`` attribute is not ``None``, then the dtype represents a structured dtype,

        In this check we ensure that ``fields`` is ``None``.

        Parameters
        ----------
        dtype : TDType
            The dtype to check.

        Returns
        -------
        Bool
            True if the dtype matches, False otherwise.
        """
        return cls.dtype_cls is type(dtype) and dtype.fields is None  # type: ignore[has-type]

    def default_scalar(self) -> np.void:
        return self.to_native_dtype().type(("\x00" * self.length).encode("ascii"))

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(self.cast_scalar(data).tobytes()).decode("ascii")

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        if check_json_str(data):
            return self.to_native_dtype().type(base64.standard_b64decode(data))
        raise TypeError(f"Invalid type: {data}. Expected a string.")  # pragma: no cover

    def _check_scalar(self, data: object) -> bool:
        return isinstance(data, np.bytes_ | str | bytes | np.void)

    def _cast_scalar_unchecked(self, data: object) -> np.void:
        native_dtype = self.to_native_dtype()
        # Without the second argument, numpy will return a void scalar for dtype V1.
        # The second argument ensures that, if native_dtype is something like V10,
        # the result will actually be a V10 scalar.
        return native_dtype.type(data, native_dtype)

    @property
    def item_size(self) -> int:
        return self.length


@dataclass(frozen=True, kw_only=True)
class VariableLengthBytes(ZDType[np.dtypes.ObjectDType, bytes], HasObjectCodec):
    dtype_cls = np.dtypes.ObjectDType
    _zarr_v3_name: ClassVar[Literal["variable_length_bytes"]] = "variable_length_bytes"
    object_codec_id = "vlen-bytes"

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
    def _check_json_v3(cls, data: JSON) -> TypeGuard[Literal["variable_length_bytes"]]:
        return data == cls._zarr_v3_name

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Literal["|O"]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["variable_length_bytes"]: ...

    def to_json(self, zarr_format: ZarrFormat) -> Literal["|O", "variable_length_bytes"]:
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

    def default_scalar(self) -> bytes:
        return b""

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(data).decode("ascii")  # type: ignore[arg-type]

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> bytes:
        if check_json_str(data):
            return base64.standard_b64decode(data.encode("ascii"))
        raise TypeError(f"Invalid type: {data}. Expected a string.")  # pragma: no cover

    def _check_scalar(self, data: object) -> bool:
        return isinstance(data, bytes | str)

    def _cast_scalar_unchecked(self, data: object) -> bytes:
        if isinstance(data, str):
            return bytes(data, encoding="utf-8")
        return bytes(data)  # type: ignore[no-any-return, call-overload]
