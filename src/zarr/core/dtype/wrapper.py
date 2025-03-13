from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeGuard, TypeVar, cast

import numpy as np

from zarr.abc.metadata import Metadata
from zarr.core.dtype.common import DataTypeValidationError

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

TScalar = TypeVar("TScalar", bound=np.generic | str)
# TODO: figure out an interface or protocol that non-numpy dtypes can use
TDType = TypeVar("TDType", bound=np.dtype[Any])


@dataclass(frozen=True, kw_only=True)
class DTypeWrapper(Generic[TDType, TScalar], ABC, Metadata):
    """
    Abstract base class for wrapping numpy dtypes.

    Attributes
    ----------
    dtype_cls : ClassVar[type[TDType]]
        The numpy dtype class. This is a class variable. Instances of this class cannot set it.
    _zarr_v3_name : ClassVar[str]
        The name given to the wrapped data type by a zarr v3 data type specification. Note that this
        is not necessarily the same name that will appear in metadata documents, as some data types
        have names that depend on their configuration.
    """

    # this class will create a numpy dtype
    # mypy currently disallows class variables to contain type parameters
    # but it seems like it should be OK for us to use it here:
    # https://github.com/python/typing/discussions/1424#discussioncomment-7989934
    dtype_cls: ClassVar[type[TDType]]  # type: ignore[misc]
    _zarr_v3_name: ClassVar[str]

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
        raise NotImplementedError

    @abstractmethod
    def to_dtype(self: Self) -> TDType:
        """
        Return an instance of the wrapped dtype.

        Returns
        -------
        TDType
            The unwrapped dtype.
        """
        raise NotImplementedError

    def cast_value(self: Self, value: object) -> TScalar:
        """
        Cast a value to an instance of the scalar type.
        This implementation assumes a numpy-style dtype class that has a
        ``type`` method for casting scalars. Non-numpy dtypes will need to
        override this method.

        Parameters
        ----------
        value : object
            The value to cast.

        Returns
        -------
        TScalar
            The cast value.
        """
        return cast(TScalar, self.to_dtype().type(value))

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
    def check_dtype(cls: type[Self], dtype: TDType) -> TypeGuard[TDType]:
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
    def check_dict(cls: type[Self], data: dict[str, JSON]) -> TypeGuard[dict[str, JSON]]:
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
        return "name" in data and data["name"] == cls._zarr_v3_name

    @abstractmethod
    def to_dict(self) -> dict[str, JSON]:
        """
        Convert the wrapped data type to a dictionary.

        Returns
        -------
        dict[str, JSON]
            The dictionary representation of the wrapped data type
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, JSON]) -> Self:
        """
        Wrap a JSON representation of a data type.

        Parameters
        ----------
        data : dict[str, JSON]
            The JSON representation of the data type.

        Returns
        -------
        Self
            The wrapped data type.
        """
        if cls.check_dict(data):
            return cls._from_dict_unsafe(data)
        raise DataTypeValidationError(f"Invalid JSON representation of data type {cls}.")

    @classmethod
    def _from_dict_unsafe(cls: type[Self], data: dict[str, JSON]) -> Self:
        """
        Wrap a JSON representation of a data type.

        Parameters
        ----------
        data : dict[str, JSON]
            The JSON representation of the data type.

        Returns
        -------
        Self
            The wrapped data type.
        """
        config = data.get("configuration", {})
        return cls(**config)

    def get_name(self, zarr_format: ZarrFormat) -> str:
        """
        Return the name of the wrapped data type.

        Parameters
        ----------
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        str
            The name of the wrapped data type.

        Notes
        -----
        This is a method, rather than an attribute, because the name of the data type may depend on
        parameters that are not known until a concrete data type is wrapped.

        As the names of data types vary between zarr versions, this method takes a ``zarr_format``
        parameter
        """
        if zarr_format == 2:
            return self.to_dtype().str
        return self._zarr_v3_name

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
            The JSON-serializable format.
        """
        raise NotImplementedError

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
            The numpy scalar.
        """
        raise NotImplementedError
