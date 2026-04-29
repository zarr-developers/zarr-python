"""Data type interface definitions (v1).

This module defines the abstract interface for zarr data types.
External data type implementations should subclass ``ZDType`` from this
module. The interface is intentionally minimal and stable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    ClassVar,
    Literal,
    Self,
    TypeGuard,
    overload,
)

import numpy as np

# JSON-like type for serialization
type JSON = str | int | float | bool | dict[str, JSON] | list[JSON] | None
type ZarrFormat = Literal[2, 3]

# Bounds for the scalar and dtype type parameters
type TBaseScalar = np.generic | str | bytes
type TBaseDType = np.dtype[np.generic]

# JSON representations of data types
type DTypeJSON = JSON
type DTypeSpec_V2 = str | list[tuple[str, DTypeJSON]]
type DTypeSpec_V3 = str | dict[str, JSON]


@dataclass(frozen=True, kw_only=True, slots=True)
class ZDType[DType: TBaseDType, Scalar: TBaseScalar](ABC):
    """Abstract base class for wrapping native array data types.

    Subclasses must implement all abstract methods to support serialization,
    deserialization, and scalar handling for their native data type.

    Type Parameters
    ---------------
    DType
        The native data type (e.g. ``np.dtype[np.float64]``).
    Scalar
        The scalar type produced by this data type (e.g. ``np.float64``).
    """

    dtype_cls: ClassVar[type[TBaseDType]]
    _zarr_v3_name: ClassVar[str]

    @classmethod
    def _check_native_dtype(cls: type[Self], dtype: TBaseDType) -> TypeGuard[DType]:
        """Check that a native data type matches ``dtype_cls``."""
        return type(dtype) is cls.dtype_cls

    @classmethod
    @abstractmethod
    def from_native_dtype(cls: type[Self], dtype: TBaseDType) -> Self:
        """Create an instance from a native data type."""
        ...

    @abstractmethod
    def to_native_dtype(self: Self) -> DType:
        """Return the native data type wrapped by this instance."""
        ...

    @classmethod
    @abstractmethod
    def _from_json_v2(cls: type[Self], data: DTypeJSON) -> Self: ...

    @classmethod
    @abstractmethod
    def _from_json_v3(cls: type[Self], data: DTypeJSON) -> Self: ...

    @classmethod
    def from_json(cls: type[Self], data: DTypeJSON, *, zarr_format: ZarrFormat) -> Self:
        """Create an instance from JSON metadata."""
        if zarr_format == 2:
            return cls._from_json_v2(data)
        if zarr_format == 3:
            return cls._from_json_v3(data)
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")

    @overload
    def to_json(self, zarr_format: Literal[2]) -> DTypeSpec_V2: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> DTypeSpec_V3: ...

    @abstractmethod
    def to_json(self, zarr_format: ZarrFormat) -> DTypeSpec_V2 | DTypeSpec_V3:
        """Serialize this data type to JSON."""
        ...

    @abstractmethod
    def _check_scalar(self, data: object) -> bool:
        """Check that a python object is a valid scalar for this data type."""
        ...

    @abstractmethod
    def cast_scalar(self, data: object) -> Scalar:
        """Cast a python object to the scalar type of this data type."""
        ...

    @abstractmethod
    def default_scalar(self) -> Scalar:
        """Return the default scalar value for this data type."""
        ...

    @abstractmethod
    def from_json_scalar(self: Self, data: JSON, *, zarr_format: ZarrFormat) -> Scalar:
        """Deserialize a JSON value to a scalar."""
        ...

    @abstractmethod
    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize a scalar value to JSON."""
        ...
