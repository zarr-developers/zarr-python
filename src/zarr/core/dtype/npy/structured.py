from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Self, TypeGuard, cast, overload

import numpy as np

from zarr.core.common import NamedConfig
from zarr.core.dtype.common import (
    DataTypeValidationError,
    DTypeConfig_V2,
    DTypeJSON,
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
    from zarr.core.common import JSON, ZarrFormat

StructuredScalarLike = list[object] | tuple[object, ...] | bytes | int


class StructuredJSON_V2(DTypeConfig_V2[StructuredName_V2, None]):
    """
    A wrapper around the JSON representation of the ``Structured`` data type in Zarr V2.

    The ``name`` field is a sequence of sequences, where each inner sequence has two values:
    the field name and the data type name for that field (which could be another sequence).
    The data type names are strings, and the object codec ID is always None.

    References
    ----------
    The structure of the ``name`` field is defined in the Zarr V2
    [specification document](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v2/v2.0.rst#data-type-encoding).

    Examples
    --------
    ```python
    {
        "name": [
            ["f0", "<m8[10s]"],
            ["f1", "<m8[10s]"],
        ],
        "object_codec_id": None
    }
    ```
    """


class StructuredJSON_V3(
    NamedConfig[Literal["struct", "structured"], dict[str, Sequence[dict[str, str | DTypeJSON]]]]
):
    """
    A JSON representation of a structured data type in Zarr V3.

    References
    ----------
    The Zarr V3 specification for this data type is defined in the zarr-extensions repository:
    https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/struct

    Examples
    --------
    ```python
    {
        "name": "struct",
        "configuration": {
            "fields": [
                {"name": "f0", "data_type": "int32"},
                {"name": "f1", "data_type": "float64"},
            ]
        }
    }
    ```

    The legacy tuple format ``[["f0", "int32"], ["f1", "float64"]]`` is also
    accepted when reading for backward compatibility.
    """


@dataclass(frozen=True, kw_only=True)
class Structured(ZDType[np.dtypes.VoidDType[int], np.void], HasItemSize):
    """
    A Zarr data type for arrays containing structured scalars, AKA "record arrays".

    Wraps the NumPy `np.dtypes.VoidDType` if the data type has fields. Scalars for this data
    type are instances of `np.void`, with a ``fields`` attribute.

    Attributes
    ----------
    fields : Sequence[tuple[str, ZDType]]
        The fields of the structured dtype.

    References
    ----------
    The Zarr V3 specification for this data type is defined in the zarr-extensions repository:
    https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/struct

    The Zarr V2 data type specification can be found [here](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v2/v2.0.rst#data-type-encoding).
    """

    _zarr_v3_name: ClassVar[Literal["struct"]] = "struct"
    _zarr_v3_names: ClassVar[tuple[str, ...]] = ("struct", "structured")
    dtype_cls = np.dtypes.VoidDType  # type: ignore[assignment]
    fields: tuple[tuple[str, ZDType[TBaseDType, TBaseScalar]], ...]

    def __post_init__(self) -> None:
        if len(self.fields) < 1:
            raise ValueError(f"must have at least one field. Got {self.fields!r}")

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
        """
        Create a Structured ZDType from a native NumPy data type.

        Parameters
        ----------
        dtype : TBaseDType
            The native data type.

        Returns
        -------
        Self
            An instance of this data type.

        Raises
        ------
        DataTypeValidationError
            If the input data type is not an instance of np.dtypes.VoidDType with a non-null
            ``fields`` attribute.

        Notes
        -----
        This method attempts to resolve the fields of the structured dtype using the data type
        registry.
        """
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
        """
        Convert the structured Zarr data type to a native NumPy void dtype.

        This method constructs a NumPy dtype with fields corresponding to the
        fields of the structured Zarr data type, by converting each field's
        data type to its native dtype representation.

        Returns
        -------
        np.dtypes.VoidDType[int]
            The native NumPy void dtype representing the structured data type.
        """

        return cast(
            "np.dtypes.VoidDType[int]",
            np.dtype([(key, dtype.to_native_dtype()) for (key, dtype) in self.fields]),
        )

    @classmethod
    def _check_json_v2(
        cls,
        data: DTypeJSON,
    ) -> TypeGuard[StructuredJSON_V2]:
        """
        Check if the input is a valid JSON representation of a Structured data type
        for Zarr V2.

        The input data must be a mapping that contains a "name" key that is not a str,
        and an "object_codec_id" key that is None.

        Parameters
        ----------
        data : DTypeJSON
            The JSON data to check.

        Returns
        -------
        TypeGuard[StructuredJSON_V2]
            True if the input is a valid JSON representation of a Structured data type
            for Zarr V2, False otherwise.
        """
        return (
            check_dtype_spec_v2(data)
            and not isinstance(data["name"], str)
            and check_structured_dtype_name_v2(data["name"])
            and data["object_codec_id"] is None
        )

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[StructuredJSON_V3]:
        """
        Check that the input is a valid JSON representation of this class in Zarr V3.

        Parameters
        ----------
        data : DTypeJSON
            The JSON data to check.

        Returns
        -------
        TypeGuard[StructuredJSON_V3]
            True if the input is a valid JSON representation of a structured data type for Zarr V3,
            False otherwise.
        """
        return (
            isinstance(data, dict)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] in cls._zarr_v3_names
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
            dtype_name = data["name"]
            parsed_fields: list[tuple[str, ZDType[TBaseDType, TBaseScalar]]] = []
            for field in meta_fields:
                if dtype_name == "struct":
                    if not isinstance(field, dict):
                        msg = f"Invalid field format for 'struct' dtype. Expected object with 'name' and 'data_type' keys, got {field!r}"
                        raise DataTypeValidationError(msg)
                    f_name = field["name"]
                    f_dtype = field["data_type"]
                else:
                    if isinstance(field, dict):
                        msg = f"Invalid field format for 'structured' dtype. Expected [name, dtype] tuple, got {field!r}"
                        raise DataTypeValidationError(msg)
                    f_name, f_dtype = field
                parsed_fields.append(
                    (f_name, get_data_type_from_json(f_dtype, zarr_format=3))  # type: ignore[misc]
                )
            return cls(fields=tuple(parsed_fields))
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected a JSON object with the key {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> StructuredJSON_V2: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> StructuredJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> StructuredJSON_V2 | StructuredJSON_V3:
        """
        Convert the structured data type to a JSON-serializable form.

        Parameters
        ----------
        zarr_format : ZarrFormat
            The Zarr format version. Accepted values are 2 and 3.

        Returns
        -------
        StructuredJSON_V2 | StructuredJSON_V3
            The JSON representation of the structured data type.

        Raises
        ------
        ValueError
            If the zarr_format is not 2 or 3.
        """
        if zarr_format == 2:
            fields = [
                [f_name, f_dtype.to_json(zarr_format=zarr_format)["name"]]
                for f_name, f_dtype in self.fields
            ]
            return {"name": fields, "object_codec_id": None}
        elif zarr_format == 3:
            v3_unstable_dtype_warning(self)
            fields = [
                {"name": f_name, "data_type": f_dtype.to_json(zarr_format=zarr_format)}
                for f_name, f_dtype in self.fields
            ]
            base_dict = {
                "name": self._zarr_v3_name,
                "configuration": {"fields": fields},
            }
            return cast("StructuredJSON_V3", base_dict)
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def _check_scalar(self, data: object) -> TypeGuard[StructuredScalarLike]:
        # TODO: implement something more precise here!
        """
        Check that the input is a valid scalar value for this structured data type.

        Parameters
        ----------
        data : object
            The scalar value to check.

        Returns
        -------
        TypeGuard[StructuredScalarLike]
            Whether the input is a valid scalar value for this structured data type.
        """
        return isinstance(data, (bytes, list, tuple, int, np.void))

    def _cast_scalar_unchecked(self, data: StructuredScalarLike) -> np.void:
        """
        Cast a python object to a numpy structured scalar without type checking.

        Parameters
        ----------
        data : StructuredScalarLike
            The data to cast.

        Returns
        -------
        np.void
            The casted data as a numpy structured scalar.

        Notes
        -----
        This method does not perform any type checking.
        The input data must be castable to a numpy structured scalar.

        """
        na_dtype = self.to_native_dtype()
        if isinstance(data, bytes):
            res = np.frombuffer(data, dtype=na_dtype)[0]
        elif isinstance(data, list | tuple):
            res = np.array([tuple(data)], dtype=na_dtype)[0]
        else:
            res = np.array([data], dtype=na_dtype)[0]
        return cast("np.void", res)

    def cast_scalar(self, data: object) -> np.void:
        """
        Cast a Python object to a NumPy structured scalar.

        This function attempts to cast the provided data to a NumPy structured scalar.
        If the data is compatible with the structured scalar type, it is cast without
        type checking. Otherwise, a TypeError is raised.

        Parameters
        ----------
        data : object
            The data to be cast to a NumPy structured scalar.

        Returns
        -------
        np.void
            The data cast as a NumPy structured scalar.

        Raises
        ------
        TypeError
            If the data cannot be converted to a NumPy structured scalar.
        """

        if self._check_scalar(data):
            return self._cast_scalar_unchecked(data)
        msg = (
            f"Cannot convert object {data!r} with type {type(data)} to a scalar compatible with the "
            f"data type {self}."
        )
        raise TypeError(msg)

    def default_scalar(self) -> np.void:
        """
        Get the default scalar value for this structured data type.

        Returns
        -------
        np.void
            The default scalar value, which is the scalar representation of 0
            cast to this structured data type.
        """

        return self._cast_scalar_unchecked(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        """
        Read a JSON-serializable value as a NumPy structured scalar.

        Parameters
        ----------
        data : JSON
            The JSON-serializable value. Can be either:
            - A dict mapping field names to values (primary format for V3)
            - A base64-encoded string (legacy format, for backward compatibility)
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        np.void
            The NumPy structured scalar.

        Raises
        ------
        TypeError
            If the input is not a dict or base64-encoded string.
        """
        if isinstance(data, dict):
            field_values = []
            for field_name, field_dtype in self.fields:
                if field_name in data:
                    field_values.append(
                        field_dtype.from_json_scalar(data[field_name], zarr_format=zarr_format)
                    )
                else:
                    field_values.append(field_dtype.default_scalar())
            return self._cast_scalar_unchecked(tuple(field_values))
        elif check_json_str(data):
            as_bytes = bytes_from_json(data, zarr_format=zarr_format)
            dtype = self.to_native_dtype()
            return cast("np.void", np.array([as_bytes]).view(dtype)[0])
        raise TypeError(f"Invalid type: {data}. Expected a dict or base64-encoded string.")

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> str | dict[str, JSON]:
        """
        Convert a scalar to a JSON-serializable representation.

        Parameters
        ----------
        data : object
            The scalar to convert.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        str | dict[str, JSON]
            For V2: A base64-encoded string of the bytes that make up the scalar.
            For V3: A dict mapping field names to their JSON-serialized values.
        """
        scalar = self.cast_scalar(data)
        if zarr_format == 2:
            return bytes_to_json(scalar.tobytes(), zarr_format)
        result: dict[str, JSON] = {}
        for field_name, field_dtype in self.fields:
            result[field_name] = field_dtype.to_json_scalar(
                scalar[field_name], zarr_format=zarr_format
            )
        return result

    @property
    def item_size(self) -> int:
        """
        The size of a single scalar in bytes.

        Returns
        -------
        int
            The size of a single scalar in bytes.
        """
        return self.to_native_dtype().itemsize

    def has_multi_byte_fields(self) -> bool:
        """
        Check if this structured dtype has any fields with item_size > 1.

        Returns
        -------
        bool
            True if any field has item_size > 1, False otherwise.
        """
        return any(
            isinstance(field_dtype, HasItemSize) and field_dtype.item_size > 1
            for _, field_dtype in self.fields
        )
