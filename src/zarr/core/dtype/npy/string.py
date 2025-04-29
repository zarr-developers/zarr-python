from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, TypeGuard

import numpy as np

from zarr.core.dtype.npy.common import check_json_str
from zarr.core.dtype.wrapper import ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat
    from zarr.core.dtype.wrapper import _BaseDType

_NUMPY_SUPPORTS_VLEN_STRING = hasattr(np.dtypes, "StringDType")


if _NUMPY_SUPPORTS_VLEN_STRING:

    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(ZDType[np.dtypes.StringDType, str]):  # type: ignore[type-var]
        dtype_cls = np.dtypes.StringDType
        _zarr_v3_name = "numpy.variable_length_utf8"

        @classmethod
        def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
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

        def _cast_value_unsafe(self, value: object) -> str:
            return str(value)

else:
    # Numpy pre-2 does not have a variable length string dtype, so we use the Object dtype instead.
    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(ZDType[np.dtypes.ObjectDType, str]):  # type: ignore[no-redef]
        dtype_cls = np.dtypes.ObjectDType
        _zarr_v3_name = "numpy.variable_length_utf8"

        @classmethod
        def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
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

        def _cast_value_unsafe(self, value: object) -> str:
            return str(value)
