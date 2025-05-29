from dataclasses import dataclass
from typing import ClassVar, Literal, Self, TypeGuard, overload

import numpy as np

from zarr.core.common import JSON, ZarrFormat
from zarr.core.dtype.common import HasObjectCodec
from zarr.core.dtype.wrapper import TBaseDType, ZDType


@dataclass(frozen=True, kw_only=True)
class VariableLengthString(ZDType[np.dtypes.ObjectDType, str], HasObjectCodec):  # type: ignore[no-redef]
    dtype_cls = np.dtypes.ObjectDType
    _zarr_v3_name: ClassVar[Literal["variable_length_bytes"]] = "variable_length_bytes"
    object_codec_id = "vlen-bytes"

    @classmethod
    def _from_native_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
        return cls()

    def to_native_dtype(self) -> np.dtypes.ObjectDType:
        return self.dtype_cls()

    @classmethod
    def check_json_v2(
        cls, data: JSON, *, object_codec_id: str | None = None
    ) -> TypeGuard[Literal["|O"]]:
        """
        Check that the input is a valid JSON representation of a numpy O dtype, and that the
        object codec id is appropriate for variable-length UTF-8 strings.
        """
        return data == "|O" and object_codec_id == cls.object_codec_id

    @classmethod
    def check_json_v3(cls, data: JSON) -> TypeGuard[Literal["variable_length_utf8"]]:
        return data == cls._zarr_v3_name

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Literal["|O"]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["variable_length_utf8"]: ...

    def to_json(self, zarr_format: ZarrFormat) -> Literal["|O", "variable_length_utf8"]:
        if zarr_format == 2:
            return "|O"
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unchecked(
        cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
    ) -> Self:
        return cls()

    def default_scalar(self) -> str:
        return ""

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return data  # type: ignore[return-value]

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
        """
        Strings pass through
        """
        if not check_json_str(data):
            raise TypeError(f"Invalid type: {data}. Expected a string.")
        return data

    def check_scalar(self, data: object) -> bool:
        return isinstance(data, str)

    def _cast_scalar_unchecked(self, data: object) -> str:
        return str(data)
