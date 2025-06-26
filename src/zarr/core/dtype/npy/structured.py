from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, TypeGuard, cast, overload

import numpy as np

from zarr.core.dtype.common import (
    DataTypeValidationError,
    DTypeConfig_V2,
    DTypeJSON,
    DTypeSpec_V3,
    HasItemSize,
    StructuredName_V2,
    check_dtype_spec_v2,
    check_structured_dtype_name_v2,
    v3_unstable_dtype_warning,
)
from zarr.core.dtype.npy.common import (
    bytes_from_json,
    bytes_to_json,
    check_json_str,
)
from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr.core.common import JSON, NamedConfig, ZarrFormat

StructuredScalarLike = list[object] | tuple[object, ...] | bytes | int


@dataclass(frozen=True, kw_only=True)
class Structured(ZDType[np.dtypes.VoidDType[int], np.void], HasItemSize):
    dtype_cls = np.dtypes.VoidDType  # type: ignore[assignment]
    _zarr_v3_name = "structured"
    fields: tuple[tuple[str, ZDType[TBaseDType, TBaseScalar]], ...]

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
        return isinstance(dtype, cls.dtype_cls) and dtype.fields is not None

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        from zarr.core.dtype import get_data_type_from_native_dtype

        fields: list[tuple[str, ZDType[TBaseDType, TBaseScalar]]] = []
        if cls._check_native_dtype(dtype):
            # fields of a structured numpy dtype are either 2-tuples or 3-tuples. we only
            # care about the first element in either case.
            for key, (dtype_instance, *_) in dtype.fields.items():  # type: ignore[union-attr]
                dtype_wrapped = get_data_type_from_native_dtype(dtype_instance)
                fields.append((key, dtype_wrapped))

            return cls(fields=tuple(fields))
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self) -> np.dtypes.VoidDType[int]:
        return cast(
            "np.dtypes.VoidDType[int]",
            np.dtype([(key, dtype.to_native_dtype()) for (key, dtype) in self.fields]),
        )

    @classmethod
    def _check_json_v2(
        cls,
        data: DTypeJSON,
    ) -> TypeGuard[DTypeConfig_V2[StructuredName_V2, None]]:
        return (
            check_dtype_spec_v2(data)
            and not isinstance(data["name"], str)
            and check_structured_dtype_name_v2(data["name"])
            and data["object_codec_id"] is None
        )

    @classmethod
    def _check_json_v3(
        cls, data: DTypeJSON
    ) -> TypeGuard[NamedConfig[Literal["structured"], dict[str, Sequence[tuple[str, DTypeJSON]]]]]:
        return (
            isinstance(data, dict)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == cls._zarr_v3_name
            and isinstance(data["configuration"], dict)
            and set(data["configuration"].keys()) == {"fields"}
        )

    @classmethod
    def _from_json_v2(cls, data: DTypeJSON) -> Self:
        # avoid circular import
        from zarr.core.dtype import get_data_type_from_json

        if cls._check_json_v2(data):
            # structured dtypes are constructed directly from a list of lists
            # note that we do not handle the object codec here! this will prevent structured
            # dtypes from containing object dtypes.
            return cls(
                fields=tuple(  # type: ignore[misc]
                    (  # type: ignore[misc]
                        f_name,
                        get_data_type_from_json(
                            {"name": f_dtype, "object_codec_id": None}, zarr_format=2
                        ),
                    )
                    for f_name, f_dtype in data["name"]
                )
            )
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected a JSON array of arrays"
        raise DataTypeValidationError(msg)

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> Self:
        # avoid circular import
        from zarr.core.dtype import get_data_type_from_json

        if cls._check_json_v3(data):
            config = data["configuration"]
            meta_fields = config["fields"]
            return cls(
                fields=tuple(
                    (f_name, get_data_type_from_json(f_dtype, zarr_format=3))
                    for f_name, f_dtype in meta_fields
                )
            )
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected a JSON object with the key {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload  # type: ignore[override]
    def to_json(self, zarr_format: Literal[2]) -> DTypeConfig_V2[StructuredName_V2, None]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> DTypeSpec_V3: ...

    def to_json(
        self, zarr_format: ZarrFormat
    ) -> DTypeConfig_V2[StructuredName_V2, None] | DTypeSpec_V3:
        if zarr_format == 2:
            fields = [
                [f_name, f_dtype.to_json(zarr_format=zarr_format)["name"]]
                for f_name, f_dtype in self.fields
            ]
            return {"name": fields, "object_codec_id": None}
        elif zarr_format == 3:
            v3_unstable_dtype_warning(self)
            fields = [
                [f_name, f_dtype.to_json(zarr_format=zarr_format)]  # type: ignore[list-item]
                for f_name, f_dtype in self.fields
            ]
            base_dict = {
                "name": self._zarr_v3_name,
                "configuration": {"fields": fields},
            }
            return cast("DTypeSpec_V3", base_dict)
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def _check_scalar(self, data: object) -> TypeGuard[StructuredScalarLike]:
        # TODO: implement something more precise here!
        return isinstance(data, (bytes, list, tuple, int, np.void))

    def _cast_scalar_unchecked(self, data: StructuredScalarLike) -> np.void:
        na_dtype = self.to_native_dtype()
        if isinstance(data, bytes):
            res = np.frombuffer(data, dtype=na_dtype)[0]
        elif isinstance(data, list | tuple):
            res = np.array([tuple(data)], dtype=na_dtype)[0]
        else:
            res = np.array([data], dtype=na_dtype)[0]
        return cast("np.void", res)

    def cast_scalar(self, data: object) -> np.void:
        if self._check_scalar(data):
            return self._cast_scalar_unchecked(data)
        msg = f"Cannot convert object with type {type(data)} to a numpy structured scalar."
        raise TypeError(msg)

    def default_scalar(self) -> np.void:
        return self._cast_scalar_unchecked(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        if check_json_str(data):
            as_bytes = bytes_from_json(data, zarr_format=zarr_format)
            dtype = self.to_native_dtype()
            return cast("np.void", np.array([as_bytes]).view(dtype)[0])
        raise TypeError(f"Invalid type: {data}. Expected a string.")

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return bytes_to_json(self.cast_scalar(data).tobytes(), zarr_format)

    @property
    def item_size(self) -> int:
        # Lets have numpy do the arithmetic here
        return self.to_native_dtype().itemsize
