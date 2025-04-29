from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Self,
    TypeGuard,
    cast,
)

import numpy as np

from zarr.core.common import JSON, ZarrFormat
from zarr.core.dtype.common import HasEndianness
from zarr.core.dtype.npy.common import (
    ComplexLike,
    TComplexDType_co,
    TComplexScalar_co,
    check_json_complex_float,
    complex_float_from_json,
    complex_float_to_json,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
)
from zarr.core.dtype.wrapper import ZDType, _BaseDType

if TYPE_CHECKING:
    from zarr.core.dtype.npy.common import EndiannessNumpy


@dataclass(frozen=True)
class BaseComplex(ZDType[TComplexDType_co, TComplexScalar_co], HasEndianness):
    # This attribute holds the possible zarr v2 JSON names for the data type
    _zarr_v2_names: ClassVar[tuple[str, ...]]

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> TComplexDType_co:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)  # type: ignore[return-value]

    def to_json(self, zarr_format: ZarrFormat) -> str:
        """
        Convert the wrapped data type to a JSON-serializable form.

        Parameters
        ----------
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        str
            The JSON-serializable representation of the wrapped data type
        """
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        """
        Check that the input is a valid JSON representation of this data type.
        """
        if zarr_format == 2:
            return data in cls._zarr_v2_names
        elif zarr_format == 3:
            return data == cls._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def check_value(self, value: object) -> bool:
        return isinstance(value, ComplexLike)

    def _cast_value_unsafe(self, value: object) -> TComplexScalar_co:
        if self.check_value(value):
            return self.to_dtype().type(value)  # type: ignore[arg-type, return-value]
        raise TypeError(f"Invalid type: {value}. Expected a value castable to a complex scalar.")

    def default_value(self) -> TComplexScalar_co:
        """
        Get the default value, which is 0 cast to this dtype

        Returns
        -------
        Int scalar
            The default value.
        """
        return self._cast_value_unsafe(0)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> TComplexScalar_co:
        """
        Read a JSON-serializable value as a numpy float.

        Parameters
        ----------
        data : JSON
            The JSON-serializable value.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        TScalar_co
            The numpy float.
        """
        if check_json_complex_float(data, zarr_format=zarr_format):
            return self._cast_value_unsafe(complex_float_from_json(data, zarr_format=zarr_format))
        raise TypeError(
            f"Invalid type: {data}. Expected a float or a special string encoding of a float."
        )

    def to_json_value(self, data: object, zarr_format: ZarrFormat) -> JSON:
        """
        Convert an object to a JSON-serializable float.

        Parameters
        ----------
        data : _BaseScalar
            The value to convert.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        JSON
            The JSON-serializable form of the complex number, which is a list of two floats,
            each of which is encoding according to a zarr-format-specific encoding.
        """
        return complex_float_to_json(self.cast_value(data), zarr_format=zarr_format)


@dataclass(frozen=True, kw_only=True)
class Complex64(BaseComplex[np.dtypes.Complex64DType, np.complex64]):
    dtype_cls = np.dtypes.Complex64DType
    _zarr_v3_name = "complex64"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">c8", "<c8")


@dataclass(frozen=True, kw_only=True)
class Complex128(BaseComplex[np.dtypes.Complex128DType, np.complex128], HasEndianness):
    dtype_cls = np.dtypes.Complex128DType
    _zarr_v3_name = "complex128"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">c16", "<c16")
