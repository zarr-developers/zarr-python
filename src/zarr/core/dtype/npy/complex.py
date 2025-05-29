from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Self,
    TypeGuard,
    cast,
)

import numpy as np

from zarr.core.common import JSON, ZarrFormat
from zarr.core.dtype.common import HasEndianness, HasItemSize
from zarr.core.dtype.npy.common import (
    ComplexLike,
    TComplexDType_co,
    TComplexScalar_co,
    check_json_complex_float_v2,
    check_json_complex_float_v3,
    complex_float_from_json_v2,
    complex_float_from_json_v3,
    complex_float_to_json_v2,
    complex_float_to_json_v3,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
)
from zarr.core.dtype.wrapper import DTypeJSON_V2, DTypeJSON_V3, TBaseDType, ZDType

if TYPE_CHECKING:
    from zarr.core.dtype.npy.common import EndiannessNumpy


@dataclass(frozen=True)
class BaseComplex(ZDType[TComplexDType_co, TComplexScalar_co], HasEndianness, HasItemSize):
    # This attribute holds the possible zarr v2 JSON names for the data type
    _zarr_v2_names: ClassVar[tuple[str, ...]]

    @classmethod
    def _from_native_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_native_dtype(self) -> TComplexDType_co:
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
            return self.to_native_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unchecked(
        cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
    ) -> Self:
        if zarr_format == 2:
            return cls.from_native_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def check_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> TypeGuard[str]:
        """
        Check that the input is a valid JSON representation of this data type.
        """
        return data in cls._zarr_v2_names

    @classmethod
    def check_json_v3(cls, data: JSON) -> TypeGuard[str]:
        return data == cls._zarr_v3_name

    def check_scalar(self, data: object) -> bool:
        return isinstance(data, ComplexLike)

    def _cast_scalar_unchecked(self, data: object) -> TComplexScalar_co:
        return self.to_native_dtype().type(data)  # type: ignore[arg-type, return-value]

    def default_scalar(self) -> TComplexScalar_co:
        """
        Get the default value, which is 0 cast to this dtype

        Returns
        -------
        Int scalar
            The default value.
        """
        return self._cast_scalar_unchecked(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> TComplexScalar_co:
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
        if zarr_format == 2:
            if check_json_complex_float_v2(data):
                return self._cast_scalar_unchecked(complex_float_from_json_v2(data))
            raise TypeError(
                f"Invalid type: {data}. Expected a float or a special string encoding of a float."
            )
        elif zarr_format == 3:
            if check_json_complex_float_v3(data):
                return self._cast_scalar_unchecked(complex_float_from_json_v3(data))
            raise TypeError(
                f"Invalid type: {data}. Expected a float or a special string encoding of a float."
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
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
        if zarr_format == 2:
            return complex_float_to_json_v2(self.cast_scalar(data))
        elif zarr_format == 3:
            return complex_float_to_json_v3(self.cast_scalar(data))
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover


@dataclass(frozen=True, kw_only=True)
class Complex64(BaseComplex[np.dtypes.Complex64DType, np.complex64]):
    dtype_cls = np.dtypes.Complex64DType
    _zarr_v3_name: ClassVar[Literal["complex64"]] = "complex64"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">c8", "<c8")

    @property
    def item_size(self) -> int:
        return 8


@dataclass(frozen=True, kw_only=True)
class Complex128(BaseComplex[np.dtypes.Complex128DType, np.complex128], HasEndianness):
    dtype_cls = np.dtypes.Complex128DType
    _zarr_v3_name: ClassVar[Literal["complex128"]] = "complex128"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">c16", "<c16")

    @property
    def item_size(self) -> int:
        return 16
