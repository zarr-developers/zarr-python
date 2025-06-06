from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self, TypeGuard, cast, overload

import numpy as np

from zarr.core.common import JSON, NamedConfig, ZarrFormat
from zarr.core.dtype.common import (
    DataTypeValidationError,
    HasItemSize,
    v3_unstable_dtype_warning,
)
from zarr.core.dtype.npy.common import (
    bytes_from_json,
    bytes_to_json,
    check_json_str,
)
from zarr.core.dtype.wrapper import DTypeJSON_V2, DTypeJSON_V3, TBaseDType, TBaseScalar, ZDType


# TODO: tighten this up, get a v3 spec in place, handle endianness, etc.
@dataclass(frozen=True, kw_only=True)
class Structured(ZDType[np.dtypes.VoidDType[int], np.void], HasItemSize):
    dtype_cls = np.dtypes.VoidDType  # type: ignore[assignment]
    _zarr_v3_name = "structured"
    fields: tuple[tuple[str, ZDType[TBaseDType, TBaseScalar]], ...]

    def default_scalar(self) -> np.void:
        return self._cast_scalar_unchecked(0)

    def _cast_scalar_unchecked(self, data: object) -> np.void:
        na_dtype = self.to_native_dtype()
        if isinstance(data, bytes):
            res = np.frombuffer(data, dtype=na_dtype)[0]
        elif isinstance(data, list | tuple):
            res = np.array([tuple(data)], dtype=na_dtype)[0]
        else:
            res = np.array([data], dtype=na_dtype)[0]
        return cast("np.void", res)

    @classmethod
    def _check_native_dtype(cls, dtype: TBaseDType) -> TypeGuard[np.dtypes.VoidDType[int]]:
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
        return super()._check_native_dtype(dtype) and dtype.fields is not None

    @classmethod
    def _from_native_dtype_unchecked(cls, dtype: TBaseDType) -> Self:
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

    @overload
    def to_json(self, zarr_format: Literal[2]) -> DTypeJSON_V2: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> DTypeJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> DTypeJSON_V3 | DTypeJSON_V2:
        fields = [
            (f_name, f_dtype.to_json(zarr_format=zarr_format)) for f_name, f_dtype in self.fields
        ]
        if zarr_format == 2:
            return fields
        elif zarr_format == 3:
            v3_unstable_dtype_warning(self)
            base_dict = {"name": self._zarr_v3_name}
            base_dict["configuration"] = {"fields": fields}  # type: ignore[assignment]
            return cast("DTypeJSON_V3", base_dict)
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _check_json_v2(
        cls, data: JSON, *, object_codec_id: str | None = None
    ) -> TypeGuard[list[object]]:
        # the actual JSON form is recursive and hard to annotate, so we give up and do
        # list[object] for now

        return (
            not isinstance(data, str)
            and isinstance(data, Sequence)
            and all(
                not isinstance(field, str) and isinstance(field, Sequence) and len(field) == 2
                for field in data
            )
        )

    @classmethod
    def _check_json_v3(
        cls, data: JSON
    ) -> TypeGuard[NamedConfig[Literal["structured"], dict[str, Sequence[tuple[str, JSON]]]]]:
        return (
            isinstance(data, dict)
            and "name" in data
            and data["name"] == cls._zarr_v3_name
            and "configuration" in data
            and isinstance(data["configuration"], dict)
            and "fields" in data["configuration"]
        )

    @classmethod
    def _from_json_unchecked(
        cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
    ) -> Self:
        # avoid circular import issues by importing these functions here
        from zarr.core.dtype import get_data_type_from_json_v2, get_data_type_from_json_v3

        # This is a horrible mess, because this data type is recursive
        if zarr_format == 2:
            if cls._check_json_v2(data):  # type: ignore[arg-type]
                # structured dtypes are constructed directly from a list of lists
                # note that we do not handle the object codec here! this will prevent structured
                # dtypes from containing object dtypes.
                return cls(
                    fields=tuple(  # type: ignore[misc]
                        (f_name, get_data_type_from_json_v2(f_dtype, object_codec_id=None))  # type: ignore[has-type]
                        for f_name, f_dtype in data
                    )
                )
            else:
                raise DataTypeValidationError(f"Invalid JSON representation of data type {cls}.")
        elif zarr_format == 3:
            if cls._check_json_v3(data):  # type: ignore[arg-type]
                config = data["configuration"]
                meta_fields = config["fields"]
                fields = tuple(
                    (f_name, get_data_type_from_json_v3(f_dtype)) for f_name, f_dtype in meta_fields
                )
            else:
                raise DataTypeValidationError(f"Invalid JSON representation of data type {cls}.")
        else:
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

        return cls(fields=fields)

    def to_native_dtype(self) -> np.dtypes.VoidDType[int]:
        return cast(
            "np.dtypes.VoidDType[int]",
            np.dtype([(key, dtype.to_native_dtype()) for (key, dtype) in self.fields]),
        )

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return bytes_to_json(self.cast_scalar(data).tobytes(), zarr_format)

    def _check_scalar(self, data: object) -> bool:
        # TODO: implement something here!
        return True

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        if check_json_str(data):
            as_bytes = bytes_from_json(data, zarr_format=zarr_format)
            dtype = self.to_native_dtype()
            return cast("np.void", np.array([as_bytes]).view(dtype)[0])
        raise TypeError(f"Invalid type: {data}. Expected a string.")  # pragma: no cover

    @property
    def item_size(self) -> int:
        # Lets have numpy do the arithmetic here
        return self.to_native_dtype().itemsize
