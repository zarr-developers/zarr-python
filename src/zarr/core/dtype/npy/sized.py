import base64
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Self, TypeGuard, cast

import numpy as np

from zarr.core.common import JSON, ZarrFormat
from zarr.core.dtype.common import DataTypeValidationError, HasEndianness, HasLength
from zarr.core.dtype.npy.common import (
    EndiannessNumpy,
    bytes_from_json,
    bytes_to_json,
    check_json_str,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
)
from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


@dataclass(frozen=True, kw_only=True)
class FixedLengthAscii(ZDType[np.dtypes.BytesDType[int], np.bytes_], HasLength):
    dtype_cls = np.dtypes.BytesDType
    _zarr_v3_name = "numpy.fixed_length_ascii"
    item_size_bits: ClassVar[int] = 8

    @classmethod
    def _from_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
        return cls(length=dtype.itemsize // (cls.item_size_bits // 8))

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
                and "name" in data
                and data["name"] == cls._zarr_v3_name
                and "configuration" in data
                and isinstance(data["configuration"], dict)
                and "length_bits" in data["configuration"]
                and isinstance(data["configuration"]["length_bits"], int)
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return {
                "name": self._zarr_v3_name,
                "configuration": {"length_bits": self.length * self.item_size_bits},
            }
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(length=data["configuration"]["length_bits"] // cls.item_size_bits)  # type: ignore[arg-type, index, call-overload, operator]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.bytes_:
        return np.bytes_(b"")

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(data).decode("ascii")  # type: ignore[arg-type]

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.bytes_:
        if check_json_str(data):
            return self.to_dtype().type(base64.standard_b64decode(data.encode("ascii")))
        raise TypeError(f"Invalid type: {data}. Expected a string.")

    def check_value(self, data: object) -> bool:
        return isinstance(data, np.bytes_ | str | bytes)

    def _cast_value_unsafe(self, value: object) -> np.bytes_:
        return self.to_dtype().type(value)


@dataclass(frozen=True, kw_only=True)
class FixedLengthBytes(ZDType[np.dtypes.VoidDType[int], np.void], HasLength):
    # np.dtypes.VoidDType is specified in an odd way in numpy
    # it cannot be used to create instances of the dtype
    # so we have to tell mypy to ignore this here
    dtype_cls = np.dtypes.VoidDType  # type: ignore[assignment]
    _zarr_v3_name = "numpy.void"
    item_size_bits: ClassVar[int] = 8

    @classmethod
    def _from_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
        return cls(length=dtype.itemsize // (cls.item_size_bits // 8))

    def to_dtype(self) -> np.dtypes.VoidDType[int]:
        # Numpy does not allow creating a void type
        # by invoking np.dtypes.VoidDType directly
        return cast("np.dtypes.VoidDType[int]", np.dtype(f"V{self.length}"))

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        if zarr_format == 2:
            # Check that the dtype is |V1, |V2, ...
            return isinstance(data, str) and re.match(r"^\|V\d+$", data) is not None
        elif zarr_format == 3:
            return (
                isinstance(data, dict)
                and "name" in data
                and isinstance(data["name"], str)
                and (re.match(r"^r\d+$", data["name"]) is not None)
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return {"name": f"r{self.length * self.item_size_bits}"}
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(length=int(data["name"][1:]) // cls.item_size_bits)  # type: ignore[arg-type, index, call-overload]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def check_dtype(cls: type[Self], dtype: TBaseDType) -> TypeGuard[np.dtypes.VoidDType[Any]]:
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

    def default_value(self) -> np.void:
        return self.to_dtype().type(("\x00" * self.length).encode("ascii"))

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(self.cast_value(data).tobytes()).decode("ascii")

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        if check_json_str(data):
            return self.to_dtype().type(base64.standard_b64decode(data))
        raise DataTypeValidationError(f"Invalid type: {data}. Expected a string.")

    def check_value(self, data: object) -> bool:
        return isinstance(data, np.bytes_ | str | bytes | np.void)

    def _cast_value_unsafe(self, value: object) -> np.void:
        return self.to_dtype().type(value)  # type: ignore[call-overload, no-any-return]


@dataclass(frozen=True, kw_only=True)
class FixedLengthUnicode(ZDType[np.dtypes.StrDType[int], np.str_], HasEndianness, HasLength):
    dtype_cls = np.dtypes.StrDType
    _zarr_v3_name = "numpy.fixed_length_ucs4"
    item_size_bits: ClassVar[int] = 32  # UCS4 is 32 bits per code point

    @classmethod
    def _from_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(
            length=dtype.itemsize // (cls.item_size_bits // 8),
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
                and "name" in data
                and data["name"] == cls._zarr_v3_name
                and "configuration" in data
                and isinstance(data["configuration"], dict)
                and "length_bits" in data["configuration"]
                and isinstance(data["configuration"]["length_bits"], int)
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return {
                "name": self._zarr_v3_name,
                "configuration": {"length_bits": self.length * self.item_size_bits},
            }
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(length=data["configuration"]["length_bits"] // cls.item_size_bits)  # type: ignore[arg-type, index, call-overload, operator]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.str_:
        return np.str_("")

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return str(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.str_:
        if not check_json_str(data):
            raise TypeError(f"Invalid type: {data}. Expected a string.")
        return self.to_dtype().type(data)

    def check_value(self, data: object) -> bool:
        return isinstance(data, str | np.str_ | bytes)

    def _cast_value_unsafe(self, value: object) -> np.str_:
        return self.to_dtype().type(value)


@dataclass(frozen=True, kw_only=True)
class Structured(ZDType[np.dtypes.VoidDType[int], np.void]):
    dtype_cls = np.dtypes.VoidDType  # type: ignore[assignment]
    _zarr_v3_name = "structured"
    fields: tuple[tuple[str, ZDType[TBaseDType, TBaseScalar]], ...]

    def default_value(self) -> np.void:
        return self._cast_value_unsafe(0)

    def _cast_value_unsafe(self, value: object) -> np.void:
        return cast("np.void", np.array([value], dtype=self.to_dtype())[0])

    @classmethod
    def check_dtype(cls, dtype: TBaseDType) -> TypeGuard[np.dtypes.VoidDType[int]]:
        """
        Check that this dtype is a numpy structured dtype

        Parameters
        ----------
        dtype : np.dtypes.DTypeLike
            The dtype to check.

        Returns
        -------
        TypeGuard[np.dtypes.VoidDType]
            True if the dtype matches, False otherwise.
        """
        return super().check_dtype(dtype) and dtype.fields is not None

    @classmethod
    def _from_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
        from zarr.core.dtype import get_data_type_from_native_dtype

        fields: list[tuple[str, ZDType[TBaseDType, TBaseScalar]]] = []

        if dtype.fields is None:
            raise ValueError("numpy dtype has no fields")

        # fields of a structured numpy dtype are either 2-tuples or 3-tuples. we only
        # care about the first element in either case.
        for key, (dtype_instance, *_) in dtype.fields.items():
            dtype_wrapped = get_data_type_from_native_dtype(dtype_instance)
            fields.append((key, dtype_wrapped))

        return cls(fields=tuple(fields))

    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        fields = [
            (f_name, f_dtype.to_json(zarr_format=zarr_format)) for f_name, f_dtype in self.fields
        ]
        if zarr_format == 2:
            return fields
        elif zarr_format == 3:
            base_dict = {"name": self._zarr_v3_name}
            base_dict["configuration"] = {"fields": fields}  # type: ignore[assignment]
            return cast("JSON", base_dict)
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[dict[str, JSON] | list[Any]]:
        # the actual JSON form is recursive and hard to annotate, so we give up and do
        # list[Any] for now
        if zarr_format == 2:
            return (
                not isinstance(data, str)
                and isinstance(data, Sequence)
                and all(
                    not isinstance(field, str) and isinstance(field, Sequence) and len(field) == 2
                    for field in data
                )
            )
        elif zarr_format == 3:
            return (
                isinstance(data, dict)
                and "name" in data
                and "configuration" in data
                and isinstance(data["configuration"], dict)
                and "fields" in data["configuration"]
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        from zarr.core.dtype import get_data_type_from_json

        if cls.check_json(data, zarr_format=zarr_format):
            if zarr_format == 2:
                # structured dtypes are constructed directly from a list of lists
                return cls(
                    fields=tuple(  # type: ignore[misc]
                        (f_name, get_data_type_from_json(f_dtype, zarr_format=zarr_format))
                        for f_name, f_dtype in data
                    )
                )
            elif zarr_format == 3:  # noqa: SIM102
                if isinstance(data, dict) and "configuration" in data:
                    config = data["configuration"]
                    if isinstance(config, dict) and "fields" in config:
                        meta_fields = config["fields"]
                        fields = tuple(
                            (f_name, get_data_type_from_json(f_dtype, zarr_format=zarr_format))
                            for f_name, f_dtype in meta_fields
                        )
                        return cls(fields=fields)
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover
        raise DataTypeValidationError(f"Invalid JSON representation of data type {cls}.")

    def to_dtype(self) -> np.dtypes.VoidDType[int]:
        return cast(
            "np.dtypes.VoidDType[int]",
            np.dtype([(key, dtype.to_dtype()) for (key, dtype) in self.fields]),
        )

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return bytes_to_json(self.cast_value(data).tobytes(), zarr_format)

    def check_value(self, data: object) -> bool:
        # not sure which values we should accept for structured dtypes.
        try:
            np.array([data], dtype=self.to_dtype())
            return True  # noqa: TRY300
        except ValueError:
            return False

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        if not check_json_str(data):
            raise TypeError(f"Invalid type: {data}. Expected a string.")
        as_bytes = bytes_from_json(data, zarr_format=zarr_format)
        dtype = self.to_dtype()
        return cast("np.void", np.array([as_bytes], dtype=dtype.str).view(dtype)[0])
