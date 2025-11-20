from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypedDict, TypeGuard, cast, overload

import numpy as np
from numpy.typing import NDArray

from zarr.core.common import NamedRequiredConfig
from zarr.core.dtype.common import (
    DataTypeValidationError,
    DTypeJSON,
    DTypeSpec_V2,
    HasItemSize,
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

    from zarr.core.common import JSON, ZarrFormat

SubarrayScalarLike = int | bytes | list[object] | tuple[object, ...] | NDArray[Any]


class SubarrayConfigDict_V3(TypedDict):
    subdtype: DTypeJSON
    shape: Sequence[int]


class SubarrayJSON_V3(NamedRequiredConfig[Literal["subarray"], SubarrayConfigDict_V3]):
    """
    A JSON representation of a structured data type in Zarr V3.

    References
    ----------
    This representation is not currently defined in an external specification.

    Examples
    --------
    ```python
    {
        "name": "subarray",
        "configuration": {
            "subdtype": "int32",
            "shape": [2, 3],
        }
    }
    ```
    """


@dataclass(frozen=True, kw_only=True)
class Subarray(ZDType[np.dtypes.VoidDType[int], np.generic], HasItemSize):
    """
    A Zarr data type for arrays containing subarrays (i.e., arrays of arrays).

    Wraps the NumPy `np.dtypes.VoidDType` data type if the subdtype field set.
    Scalars for this data type are instances of `np.ndarray`.

    Attributes
    ----------
    subdtype : ZDType[TBaseDType, TBaseScalar]
        The data type of the subarray.
    shape : tuple[int, ...]
        The shape of the subarray.
    """

    # Note: Subarray's are in a weird position because they are represented
    # in numpy as a VoidDType, but they do not have proper scalar values of their own.
    # I.e. it is impossible to create a np.void scalar that is a subarray.
    #
    # While a np.ndarray is not a np.generic (i.e. a scalar), we use it here because it is the closest
    # match we have and just cast when needed to satisfy mypy. np.ndarray also behaves like a np.generic in
    # many ways.
    #
    # In practice, subarray types almost exclusively appear as fields in structured dtypes, so this is not a big issue.
    # Structured dtypes handle the scalars correctly. It is, however, still very practical
    # to have a distinct Subarray ZDType to handle serialization and deserialization of subarray dtypes.

    _zarr_v3_name: ClassVar[Literal["subarray"]] = "subarray"
    dtype_cls = np.dtypes.VoidDType  # type: ignore[assignment]
    subdtype: ZDType[TBaseDType, TBaseScalar]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.shape) < 1:
            raise ValueError(f"shape must have at least one dimension. Got {self.shape!r}")

    @classmethod
    def _check_native_dtype(cls, dtype: TBaseDType) -> TypeGuard[np.dtypes.VoidDType[int]]:
        """
        Check that this dtype is a numpy subarray dtype

        Parameters
        ----------
        dtype : np.dtypes.DTypeLike
            The dtype to check.

        Returns
        -------
        TypeGuard[np.dtypes.VoidDType]
            True if the dtype matches, False otherwise.
        """
        return (
            isinstance(dtype, cls.dtype_cls) and dtype.fields is None and dtype.subdtype is not None
        )

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        """
        Create a Subarray ZDType from a native NumPy data type.

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
            ``subdtype`` attribute.
        """
        from zarr.core.dtype import get_data_type_from_native_dtype

        if cls._check_native_dtype(dtype):
            base_dtype: TBaseDType = dtype.subdtype[0]  # type: ignore[index]
            shape: tuple[int, ...] = dtype.subdtype[1]  # type: ignore[index]
            dtype_wrapped = get_data_type_from_native_dtype(base_dtype)
            return cls(subdtype=dtype_wrapped, shape=shape)

        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self) -> np.dtypes.VoidDType[int]:
        """
        Convert the subarray Zarr data type to a native NumPy void dtype.

        Returns
        -------
        np.dtypes.VoidDType[int]
            The native NumPy void dtype representing the subarray data type.
        """

        return cast(
            "np.dtypes.VoidDType[int]",
            np.dtype((self.subdtype.to_native_dtype(), self.shape)),
        )

    @classmethod
    def _check_json_v2(
        cls,
        data: DTypeJSON,
    ) -> TypeGuard[DTypeJSON]:
        return False

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[SubarrayJSON_V3]:
        """
        Check that the input is a valid JSON representation of this class in Zarr V3.

        Parameters
        ----------
        data : DTypeJSON
            The JSON data to check.

        Returns
        -------
        TypeGuard[SubarrayJSON_V3]
            True if the input is a valid JSON representation of a subarray data type for Zarr V3,
            False otherwise.
        """

        return (
            isinstance(data, dict)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == cls._zarr_v3_name
            and isinstance(data["configuration"], dict)
            and set(data["configuration"].keys()) == {"subdtype", "shape"}
        )

    @classmethod
    def _from_json_v2(cls, data: DTypeJSON) -> Self:
        raise DataTypeValidationError(
            "Standalone Subarray dtype is not supported in Zarr V2. Use the Structured dtype"
        )  # pragma: no cover

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> Self:
        # avoid circular import
        from zarr.core.dtype import get_data_type_from_json

        if cls._check_json_v3(data):
            config = data["configuration"]
            subdtype = config["subdtype"]
            shape = tuple(config["shape"])
            return cls(
                subdtype=get_data_type_from_json(subdtype, zarr_format=3),
                shape=shape,
            )

        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected a JSON object with the key {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> DTypeSpec_V2: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> SubarrayJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> DTypeSpec_V2 | SubarrayJSON_V3:
        """
        Convert the subarray data type to a JSON-serializable form.

        Parameters
        ----------
        zarr_format : ZarrFormat
            The Zarr format version. Accepted values are 2 and 3.

        Returns
        -------
        SubarrayJSON_V3
            The JSON representation of the subarray data type.

        Raises
        ------
        ValueError
            If the zarr_format is not 2 or 3.
        """
        # For consistency with Structured, we always encode the shape as list, not tuple
        if zarr_format == 2:
            raise NotImplementedError(
                "Standalone Subarray dtype is not supported in Zarr V2. Use the Structured dtype"
            )  # pragma: no cover
        elif zarr_format == 3:
            v3_unstable_dtype_warning(self)
            subdtype_dict = self.subdtype.to_json(zarr_format=zarr_format)
            base_dict = {
                "name": self._zarr_v3_name,
                "configuration": {
                    "subdtype": subdtype_dict,
                    "shape": list(self.shape),
                },
            }
            return cast("SubarrayJSON_V3", base_dict)
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def _check_scalar(self, data: object) -> TypeGuard[SubarrayScalarLike]:
        """
        Check that the input is a valid scalar value for this subarray data type.

        Parameters
        ----------
        data : object
            The scalar value to check.

        Returns
        -------
        TypeGuard[SubarrayScalarLike]
            Whether the input is a valid scalar value for this subarray data type.
        """
        if isinstance(data, np.ndarray):
            na_subdtype = self.to_native_dtype().base
            return bool(data.dtype == na_subdtype and data.shape == self.shape)
        elif isinstance(data, bytes):
            return len(data) == self.item_size
        elif isinstance(data, (list, tuple)):
            for dim_size in self.shape:
                if not isinstance(data, (list, tuple)) or len(data) != dim_size:
                    return False
                data = data[0] if len(data) > 0 else []
            return True
        else:
            return self.subdtype._check_scalar(data)

    def _cast_scalar_unchecked(self, data: SubarrayScalarLike) -> np.generic:
        """
        Cast a python object to a numpy subarray scalar without type checking.

        Parameters
        ----------
        data : SubarrayScalarLike
            The data to cast.

        Returns
        -------
        np.ndarray
            The casted data as a numpy structured scalar.

        Notes
        -----
        This method does not perform any type checking.
        The input data must be castable to a numpy array.

        """
        na_dtype = self.to_native_dtype()
        if isinstance(data, bytes):
            res = np.frombuffer(data, dtype=na_dtype)[0]
        elif isinstance(data, list | tuple):
            res = np.array([tuple(data)], dtype=na_dtype)[0]
        elif isinstance(data, np.ndarray):
            res = data
        else:
            res = np.array([data], dtype=na_dtype)[0]
        return cast("np.generic", res)

    def cast_scalar(self, data: object) -> np.generic:
        """
        Cast a Python object to a NumPy array scalar.

        Parameters
        ----------
        data : object
            The data to be cast to a NumPy structured scalar.

        Returns
        -------
        np.ndarray
            The data cast as a NumPy array.

        Raises
        ------
        TypeError
            If the data cannot be converted to a NumPy array.
        """

        if self._check_scalar(data):
            return self._cast_scalar_unchecked(data)
        msg = (
            f"Cannot convert object {data!r} with type {type(data)} to a scalar compatible with the "
            f"data type {self}."
        )
        raise TypeError(msg)

    def default_scalar(self) -> np.generic:
        """
        Get the default scalar value for this subarray data type.

        Returns
        -------
        np.ndarray
            The default scalar value, which is the scalar representation of 0
            cast to this subarray data type.
        """

        return self._cast_scalar_unchecked(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.generic:
        """
        Read a JSON-serializable value as a NumPy subarray scalar.

        Parameters
        ----------
        data : JSON
            The JSON-serializable value.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        np.ndarray
            The NumPy subarray scalar.

        Raises
        ------
        TypeError
            If the input is not a base64-encoded string or an encoded scalar value from the sub data type.
        """
        try:
            single_element = self.subdtype.from_json_scalar(data, zarr_format=zarr_format)
            return self.cast_scalar(single_element)
        except TypeError:
            pass
        if check_json_str(data):
            as_bytes = bytes_from_json(data, zarr_format=zarr_format)
            return self.cast_scalar(as_bytes)  # cast_scalar will check size!
        raise TypeError(f"Invalid type: {data}. Expected a string.")

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """
        Convert a scalar to a JSON-serializable representation.

        If all elements of the subarray are identical, the scalar
        is encoded using the subdtype's JSON scalar representation. Otherwise,
        the scalar is encoded as a base64-encoded string of its bytes.

        Parameters
        ----------
        data : object
            The scalar to convert.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        JSON
            A JSON representation of the scalar.
        """
        scalar = self.cast_scalar(data)  # Careful, this is a np.ndarray
        single_element = scalar.flatten()[0]
        if np.all(scalar == single_element) or np.isnan(scalar).all():
            return self.subdtype.to_json_scalar(single_element, zarr_format=zarr_format)
        else:
            return bytes_to_json(scalar.tobytes(), zarr_format)

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
