from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from zarr.abc.codec import ArrayArrayCodec
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
    from zarr.core.metadata.v3 import ChunkGridMetadata


@dataclass(frozen=True)
class ScaleOffset(ArrayArrayCodec):
    """Scale-offset array-to-array codec.

    Encodes values by subtracting an offset and multiplying by a scale factor.
    Decodes by dividing by the scale and adding the offset.

    All arithmetic uses the input array's data type semantics (no implicit promotion).

    Parameters
    ----------
    offset : float
        Value subtracted during encoding. Default is 0.
    scale : float
        Value multiplied during encoding (after offset subtraction). Default is 1.
    """

    is_fixed_size = True

    offset: int | float | str
    scale: int | float | str

    def __init__(self, *, offset: object = 0, scale: object = 1) -> None:
        if not isinstance(offset, int | float | str):
            raise TypeError(f"offset must be a number or string, got {type(offset).__name__}")
        if not isinstance(scale, int | float | str):
            raise TypeError(f"scale must be a number or string, got {type(scale).__name__}")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "scale_offset", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)

    def to_dict(self) -> dict[str, JSON]:
        if self.offset == 0 and self.scale == 1:
            return {"name": "scale_offset"}
        config: dict[str, JSON] = {}
        if self.offset != 0:
            config["offset"] = self.offset
        if self.scale != 1:
            config["scale"] = self.scale
        return {"name": "scale_offset", "configuration": config}

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        native = dtype.to_native_dtype()
        if not np.issubdtype(native, np.integer) and not np.issubdtype(native, np.floating):
            raise ValueError(
                f"scale_offset codec only supports integer and floating-point data types. "
                f"Got {dtype}."
            )
        for name, value in [("offset", self.offset), ("scale", self.scale)]:
            try:
                dtype.from_json_scalar(value, zarr_format=3)
            except (TypeError, ValueError, OverflowError) as e:
                raise ValueError(
                    f"scale_offset {name} value {value!r} is not representable in dtype {native}."
                ) from e

    def _to_scalar(self, value: float | str, dtype: ZDType[TBaseDType, TBaseScalar]) -> TBaseScalar:
        """Convert a JSON-form value to a numpy scalar using the given dtype."""
        return dtype.from_json_scalar(value, zarr_format=3)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        zdtype = chunk_spec.dtype
        fill = chunk_spec.fill_value
        offset = self._to_scalar(self.offset, zdtype)
        scale = self._to_scalar(self.scale, zdtype)
        new_fill = (zdtype.to_native_dtype().type(fill) - offset) * scale  # type: ignore[operator]
        return replace(chunk_spec, fill_value=new_fill)

    def _decode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        arr = chunk_array.as_ndarray_like()
        offset = self._to_scalar(self.offset, chunk_spec.dtype)
        scale = self._to_scalar(self.scale, chunk_spec.dtype)
        if np.issubdtype(arr.dtype, np.integer):
            result = (arr // scale) + offset  # type: ignore[operator]
        else:
            result = (arr / scale) + offset  # type: ignore[operator]
        if result.dtype != arr.dtype:
            raise ValueError(
                f"scale_offset decode changed dtype from {arr.dtype} to {result.dtype}. "
                f"Arithmetic must preserve the data type."
            )
        return chunk_array.__class__.from_ndarray_like(result)

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def _encode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        arr = chunk_array.as_ndarray_like()
        offset = self._to_scalar(self.offset, chunk_spec.dtype)
        scale = self._to_scalar(self.scale, chunk_spec.dtype)
        result = (arr - offset) * scale  # type: ignore[operator]
        if result.dtype != arr.dtype:
            raise ValueError(
                f"scale_offset encode changed dtype from {arr.dtype} to {result.dtype}. "
                f"Arithmetic must preserve the data type."
            )
        return chunk_array.__class__.from_ndarray_like(result)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, _chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length
