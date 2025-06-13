"""
Wrapper for native array data types.

The ``ZDType`` class is an abstract base class for wrapping native array data types, e.g. NumPy dtypes.
``ZDType`` provides a common interface for working with data types in a way that is independent of the
underlying data type system.

The wrapper class encapsulates a native data type. Instances of the class can be created from a
native data type instance, and a native data type instance can be created from an instance of the
wrapper class.

The wrapper class is responsible for:
- Serializing and deserializing a native data type to Zarr V2 or Zarr V3 metadata.
  This ensures that the data type can be properly stored and retrieved from array metadata.
- Serializing and deserializing scalar values to Zarr V2 or Zarr V3 metadata. This is important for
  storing a fill value for an array in a manner that is valid for the data type.

You can add support for a new data type in Zarr by subclassing ``ZDType`` wrapper class and adapt its methods
to support your native data type. The wrapper class must be added to a data type registry
(defined elsewhere) before array creation routines or array reading routines can use your new data
type.
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
        The wrapped dtype class. This is a class variable.
    _zarr_v3_name : ClassVar[str]
        The name given to the data type by a Zarr v3 data type specification. This is a
        class variable, and it should generally be unique across different data types.
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
        Check that a native data type matches the dtype_cls class attribute. Used as a type guard.

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
    @abstractmethod
    def from_native_dtype(cls: type[Self], dtype: TBaseDType) -> Self:
        """
        Create a ZDType instance from a native data type. The default implementation first performs
        a type check via ``cls._check_native_dtype``. If that type check succeeds, the ZDType class
        instance is created.

        This method is used when taking a user-provided native data type, like a NumPy data type,
        and creating the corresponding ZDType instance from them.

        Parameters
        ----------
        dtype : TDType
            The native data type object to wrap.

        Returns
        -------
        Self
            The ZDType that wraps the native data type.

        Raises
        ------
        TypeError
            If the native data type is not consistent with the wrapped data type.
        """
        ...

    @abstractmethod
    def to_native_dtype(self: Self) -> TDType_co:
        """
        Return an instance of the wrapped data type. This operation inverts ``from_native_dtype``.

        Returns
        -------
        TDType
            The native data type wrapped by this ZDType.
        """
        ...

    @abstractmethod
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
    def default_scalar(self) -> TScalar_co:
        """
        Get the default scalar value for the wrapped data type. This is a method, rather than an
        attribute, because the default value for some data types depends on parameters that are
        not known until a concrete data type is wrapped. For example, data types parametrized by a
        length like fixed-length strings or bytes will generate scalars consistent with that length.

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
        Check that JSON data matches the Zarr V2 JSON serialization of this ZDType.

        Parameters
        ----------
        data : JSON
            The JSON representation of the data type.

        object_codec_id : str | None
            The string identifier of an object codec, if applicable. Object codecs are specific
            numcodecs codecs that zarr-python 2.x used to serialize numpy "Object" scalars.
            For example, a dtype field set to ``"|O"`` with an object codec ID of "vlen-utf8"
            indicates that the data type is a variable-length string.

            Zarr V3 has no such logic, so this parameter is only used for Zarr V2 compatibility.

        Returns
        -------
        Bool
            True if the JSON representation matches this data type, False otherwise.
        """
        ...

    @classmethod
    @abstractmethod
    def _check_json_v3(cls: type[Self], data: JSON) -> TypeGuard[DTypeJSON_V3]:
        """
        Check that JSON data matches the Zarr V3 JSON serialization of this ZDType.

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
        Serialize this ZDType to JSON.

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
    @abstractmethod
    def from_json_v3(cls: type[Self], data: JSON) -> Self:
        """
        Create an instance of this ZDType from Zarr V3 JSON data.

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

    @classmethod
    @abstractmethod
    def from_json_v2(cls: type[Self], data: JSON, *, object_codec_id: str | None = None) -> Self:
        """
        Create an instance of this ZDType from Zarr V2 JSON data.

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
        Serialize a python object to the JSON representation of a scalar. The value will first be
        cast to the scalar type associated with this ZDType, then serialized to JSON.

        Parameters
        ----------
        data : object
            The value to convert.
        zarr_format : ZarrFormat
            The zarr format version. This is specified because the JSON serialization of scalars
            differs between Zarr V2 and Zarr V3.

        Returns
        -------
        JSON
            The JSON-serialized scalar.
        """
        ...

    @abstractmethod
    def from_json_scalar(self: Self, data: JSON, *, zarr_format: ZarrFormat) -> TScalar_co:
        """
        Read a JSON-serializable value as a scalar.

        Parameters
        ----------
        data : JSON
            A JSON representation of a scalar value.
        zarr_format : ZarrFormat
            The zarr format version. This is specified because the JSON serialization of scalars
            differs between Zarr V2 and Zarr V3.

        Returns
        -------
        TScalar
            The deserialized scalar value.
        """
        ...


def scalar_failed_type_check_msg(
    cls_instance: ZDType[TBaseDType, TBaseScalar], bad_scalar: object
) -> str:
    """
    Generate an error message reporting that a particular value failed a type check when attempting
    to cast that value to a scalar.
    """
    return (
        f"The value {bad_scalar!r} failed a type check. "
        f"It cannot be safely cast to a scalar compatible with {cls_instance}. "
        f"Consult the documentation for {cls_instance} to determine the possible values that can "
        "be cast to scalars of the wrapped data type."
    )
