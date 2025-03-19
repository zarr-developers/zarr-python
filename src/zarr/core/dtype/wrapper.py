from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Self, TypeGuard, TypeVar

import numpy as np

from zarr.core.dtype.common import DataTypeValidationError

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

# This the upper bound for the scalar types we support. It's numpy scalars + str,
# because the new variable-length string dtype in numpy does not have a corresponding scalar type
_BaseScalar = np.generic | str
# This is the bound for the dtypes that we support. If we support non-numpy dtypes,
# then this bound will need to be widened.
_BaseDType = np.dtype[np.generic]
TScalar = TypeVar("TScalar", bound=_BaseScalar)
# TODO: figure out an interface or protocol that non-numpy dtypes can use
TDType = TypeVar("TDType", bound=_BaseDType)


@dataclass(frozen=True, kw_only=True)
class ZDType(Generic[TDType, TScalar], ABC):
    """
    Abstract base class for wrapping native array data types, e.g. numpy dtypes

    Attributes
    ----------
    dtype_cls : ClassVar[type[TDType]]
        The numpy dtype class. This is a class variable. Instances of this class cannot set it.
    _zarr_v3_name : ClassVar[str]
        The name given to the wrapped data type by a zarr v3 data type specification. Note that this
        is not necessarily the same name that will appear in metadata documents, as some data types
        have names that depend on their configuration.
    """

    # this class will create a native data type
    # mypy currently disallows class variables to contain type parameters
    # but it seems OK for us to use it here:
    # https://github.com/python/typing/discussions/1424#discussioncomment-7989934
    dtype_cls: ClassVar[type[TDType]]  # type: ignore[misc]
    _zarr_v3_name: ClassVar[str]

    @classmethod
    def check_dtype(cls: type[Self], dtype: _BaseDType) -> TypeGuard[TDType]:
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
    def from_dtype(cls: type[Self], dtype: TDType) -> Self:
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
        if cls.check_dtype(dtype):
            return cls._from_dtype_unsafe(dtype)
        raise DataTypeValidationError(
            f"Invalid dtype: {dtype}. Expected an instance of {cls.dtype_cls}."
        )

    @classmethod
    @abstractmethod
    def _from_dtype_unsafe(cls: type[Self], dtype: TDType) -> Self:
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
    def to_dtype(self: Self) -> TDType:
        """
        Return an instance of the wrapped dtype.

        Returns
        -------
        TDType
            The unwrapped dtype.
        """
        ...

    @abstractmethod
    def default_value(self) -> TScalar:
        """
        Get the default value for the wrapped data type. This is a method, rather than an attribute,
        because the default value for some data types may depend on parameters that are not known
        until a concrete data type is wrapped.

        Returns
        -------
        TScalar
            The default value for this data type.
        """
        ...

    @classmethod
    @abstractmethod
    def check_json(cls: type[Self], data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        """
        Check that a JSON representation of a data type matches the dtype_cls class attribute. Used
        as a type guard. This base implementation checks that the input is a dictionary,
        that the key "name" is in that dictionary, and that the value of "name"
        matches the _zarr_v3_name class attribute.

        Parameters
        ----------
        data : JSON
            The JSON representation of the data type.

        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        Bool
            True if the JSON representation matches, False otherwise.
        """
        ...

    @abstractmethod
    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        """
        Convert the wrapped data type to a JSON-serializable form.

        Parameters
        ----------
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        JSON
            The JSON-serializable representation of the wrapped data type
        """
        ...

    @classmethod
    def from_json(cls: type[Self], data: JSON, zarr_format: ZarrFormat) -> Self:
        """
        Wrap a JSON representation of a data type.

        Parameters
        ----------
        data : JSON
            The JSON representation of the data type.

        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        Self
            The wrapped data type.
        """
        if cls.check_json(data, zarr_format=zarr_format):
            return cls._from_json_unsafe(data, zarr_format=zarr_format)
        raise DataTypeValidationError(f"Invalid JSON representation of data type {cls}: {data}")

    @classmethod
    @abstractmethod
    def _from_json_unsafe(cls: type[Self], data: JSON, zarr_format: ZarrFormat) -> Self:
        """
        Wrap a JSON representation of a data type.

        Parameters
        ----------
        data : JSON
            The JSON representation of the data type.

        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        Self
            The wrapped data type.
        """
        ...

    @abstractmethod
    def to_json_value(self, data: TScalar, *, zarr_format: ZarrFormat) -> JSON:
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
    def from_json_value(self: Self, data: JSON, *, zarr_format: ZarrFormat) -> TScalar:
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
