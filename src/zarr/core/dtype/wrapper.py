"""
Wrapper for native array data types.

The `ZDType` class is an abstract base class for wrapping native array data types, e.g. numpy dtypes.
It provides a common interface for working with data types in a way that is independent of the
underlying data type system.

The wrapper class encapsulates a native data type. Instances of the class can be created from a
native data type instance, and a native data type instance can be created from an instance of the
wrapper class.

The wrapper class is responsible for:
- Reversibly serializing a native data type to Zarr V2 or Zarr V3 metadata.
  This ensures that the data type can be properly stored and retrieved from array metadata.
- Reversibly serializing scalar values to Zarr V2 or Zarr V3 metadata. This is important for
  storing a fill value for an array in a manner that is valid for the data type.

To add support for a new data type in Zarr, you should subclass the wrapper class and adapt its methods
to support your native data type. The wrapper class must be added to a data type registry
(defined elsewhere) before ``create_array`` can properly handle the new data type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Generic,
    Literal,
    Self,
    TypeGuard,
    TypeVar,
    overload,
)

import numpy as np

from zarr.core.dtype.common import DataTypeValidationError

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

# This the upper bound for the scalar types we support. It's numpy scalars + str,
# because the new variable-length string dtype in numpy does not have a corresponding scalar type
TBaseScalar = np.generic | str | bytes
# This is the bound for the dtypes that we support. If we support non-numpy dtypes,
# then this bound will need to be widened.
TBaseDType = np.dtype[np.generic]

# These two type parameters are covariant because we want
# x : ZDType[BaseDType, BaseScalar] = ZDType[SubDType, SubScalar]
# to type check
TScalar_co = TypeVar("TScalar_co", bound=TBaseScalar, covariant=True)
TDType_co = TypeVar("TDType_co", bound=TBaseDType, covariant=True)

# These types should include all JSON-serializable types that can be used to represent a data type.
DTypeJSON_V2 = str | Sequence[object]
DTypeJSON_V3 = str | Mapping[str, object]


@dataclass(frozen=True, kw_only=True, slots=True)
class ZDType(Generic[TDType_co, TScalar_co], ABC):
    """
    Abstract base class for wrapping native array data types, e.g. numpy dtypes

    Attributes
    ----------
    dtype_cls : ClassVar[type[TDType]]
        The wrapped dtype class. This is a class variable. Instances of this class cannot set it.
    _zarr_v3_name : ClassVar[str]
        The name given to the wrapped data type by a zarr v3 data type specification. Note that this
        is not necessarily the same name that will appear in metadata documents, as some data types
        have names that depend on their configuration.
    """

    # this class will create a native data type
    # mypy currently disallows class variables to contain type parameters
    # but it seems OK for us to use it here:
    # https://github.com/python/typing/discussions/1424#discussioncomment-7989934
    dtype_cls: ClassVar[type[TDType_co]]  # type: ignore[misc]
    _zarr_v3_name: ClassVar[str]

    @classmethod
    def _check_native_dtype(cls: type[Self], dtype: TBaseDType) -> TypeGuard[TDType_co]:
        """
        Check that a data type matches the dtype_cls class attribute. Used as a type guard.

        Parameters
        ----------
        dtype : TDType
            The dtype to check.

        Returns
        -------
        Bool
            True if the dtype matches, False otherwise.
        """
        return type(dtype) is cls.dtype_cls

    @classmethod
    def from_native_dtype(cls: type[Self], dtype: TBaseDType) -> Self:
        """
        Wrap a dtype object.

        Parameters
        ----------
        dtype : TDType
            The dtype object to wrap.

        Returns
        -------
        Self
            The wrapped dtype.

        Raises
        ------
        TypeError
            If the dtype does not match the dtype_cls class attribute.
        """
        if cls._check_native_dtype(dtype):
            return cls._from_native_dtype_unchecked(dtype)
        raise DataTypeValidationError(
            f"Invalid dtype: {dtype}. Expected an instance of {cls.dtype_cls}."
        )

    @classmethod
    @abstractmethod
    def _from_native_dtype_unchecked(cls: type[Self], dtype: TBaseDType) -> Self:
        """
        Wrap a native dtype without checking.

        Parameters
        ----------
        dtype : TDType
            The native dtype to wrap.

        Returns
        -------
        Self
            The wrapped dtype.
        """
        ...

    @abstractmethod
    def to_native_dtype(self: Self) -> TDType_co:
        """
        Return an instance of the wrapped dtype.

        Returns
        -------
        TDType
            The unwrapped dtype.
        """
        ...

    def cast_scalar(self, data: object) -> TScalar_co:
        """
        Cast a python object to the wrapped scalar type.
        The type of the provided scalar is first checked for compatibility.
        If it's incompatible with the associated scalar type, a ``TypeError`` will be raised.

        Parameters
        ----------
        data : object
            The python object to cast.

        Returns
        -------
        TScalar
            The cast value.
        """
        if self._check_scalar(data):
            return self._cast_scalar_unchecked(data)
        msg = (
            f"The value {data!r} failed a type check. "
            f"It cannot be safely cast to a scalar compatible with {self}. "
            f"Consult the documentation for {self} to determine the possible values that can "
            "be cast to scalars of the wrapped data type."
        )
        raise TypeError(msg)

    @abstractmethod
    def _check_scalar(self, data: object) -> bool:
        """
        Check that an python object is a valid scalar value for the wrapped data type.

        Parameters
        ----------
        data : object
            A value to check.

        Returns
        -------
        Bool
            True if the object is valid, False otherwise.
        """
        ...

    @abstractmethod
    def _cast_scalar_unchecked(self, data: object) -> TScalar_co:
        """
        Cast a python object to the wrapped data type.
        This method should not perform any type checking.

        Parameters
        ----------
        data : object
            The python object to cast.

        Returns
        -------
        TScalar
            The cast value.
        """
        ...

    @abstractmethod
    def default_scalar(self) -> TScalar_co:
        """
        Get the default scalar value for the wrapped data type. This is a method, rather than an attribute,
        because the default value for some data types may depend on parameters that are not known
        until a concrete data type is wrapped. For example, data types parametrized by a length like
        fixed-length strings or bytes will generate scalars consistent with that length.

        Returns
        -------
        TScalar
            The default value for this data type.
        """
        ...

    @classmethod
    @abstractmethod
    def _check_json_v2(
        cls: type[Self], data: JSON, *, object_codec_id: str | None = None
    ) -> TypeGuard[DTypeJSON_V2]:
        """
        Check that a JSON representation of a data type is consistent with the ZDType class.

        Parameters
        ----------
        data : JSON
            The JSON representation of the data type.

        object_codec_id : str | None
            The object codec ID, if applicable. Object codecs are specific numcodecs codecs that
            zarr-python 2.x used to serialize numpy "Object" scalars. For example, a dtype field set
            to ``"|O"`` with an object codec ID of "vlen-utf8" indicates that the data type is a
            variable-length string.

            Zarr V3 has no such logic, so this parameter is only used for Zarr V2 compatibility.

        Returns
        -------
        Bool
            True if the JSON representation matches, False otherwise.
        """
        ...

    @classmethod
    @abstractmethod
    def _check_json_v3(cls: type[Self], data: JSON) -> TypeGuard[DTypeJSON_V3]:
        """
        Check that a JSON representation of a data type matches the dtype_cls class attribute. Used
        as a type guard. This base implementation checks that the input is a dictionary,
        that the key "name" is in that dictionary, and that the value of "name"
        matches the _zarr_v3_name class attribute.

        Parameters
        ----------
        data : JSON
            The JSON representation of the data type.

        Returns
        -------
        Bool
            True if the JSON representation matches, False otherwise.
        """
        ...

    @overload
    def to_json(self, zarr_format: Literal[2]) -> DTypeJSON_V2: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> DTypeJSON_V3: ...

    @abstractmethod
    def to_json(self, zarr_format: ZarrFormat) -> DTypeJSON_V2 | DTypeJSON_V3:
        """
        Convert the wrapped data type to a JSON-serializable form.

        Parameters
        ----------
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        DTypeJSON_V2 | DTypeJSON_V3
            The JSON-serializable representation of the wrapped data type
        """
        ...

    @classmethod
    def from_json_v3(cls: type[Self], data: JSON) -> Self:
        """
        Wrap a Zarr V3 JSON representation of a data type.

        Parameters
        ----------
        data : JSON
            The JSON representation of the data type.

        Returns
        -------
        Self
            The wrapped data type.
        """
        if cls._check_json_v3(data):
            return cls._from_json_unchecked(data, zarr_format=3)
        raise DataTypeValidationError(f"Invalid JSON representation of data type {cls}: {data}")

    @classmethod
    def from_json_v2(cls: type[Self], data: JSON, *, object_codec_id: str | None) -> Self:
        """
        Wrap a Zarr V2 JSON representation of a data type.

        Parameters
        ----------
        data : JSON
            The JSON representation of the data type.

        Returns
        -------
        Self
            The wrapped data type.
        """
        if cls._check_json_v2(data, object_codec_id=object_codec_id):
            return cls._from_json_unchecked(data, zarr_format=2)
        raise DataTypeValidationError(
            f"Invalid JSON representation of data type {cls}: {data!r}, object_codec_id={object_codec_id!r}"
        )

    @classmethod
    @overload
    def _from_json_unchecked(cls, data: DTypeJSON_V2, *, zarr_format: Literal[2]) -> Self: ...
    @classmethod
    @overload
    def _from_json_unchecked(cls, data: DTypeJSON_V3, *, zarr_format: Literal[3]) -> Self: ...

    @classmethod
    @abstractmethod
    def _from_json_unchecked(
        cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
    ) -> Self:
        """
        Create a ZDType instance from a JSON representation of a data type.

        This method should be called after input has been type checked, and so it should not perform
        any input validation.

        Parameters
        ----------
        data : JSON
            The JSON representation of the data type.

        Returns
        -------
        Self
            The wrapped data type.
        """
        ...

    @abstractmethod
    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """
        Convert a single value to JSON-serializable format.

        Parameters
        ----------
        data : object
            The value to convert.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        JSON
            The JSON-serializable form of the scalar.
        """
        ...

    @abstractmethod
    def from_json_scalar(self: Self, data: JSON, *, zarr_format: ZarrFormat) -> TScalar_co:
        """
        Read a JSON-serializable value as a scalar.

        Parameters
        ----------
        data : JSON
            The JSON-serializable value.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        TScalar
            The native scalar value.
        """
        ...
