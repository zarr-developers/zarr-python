from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, NotRequired

import numpy as np
from typing_extensions import TypedDict

from zarr.abc.codec import ArrayArrayCodec
from zarr.core.common import JSON, NamedConfig

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


class ScaleOffsetConfig(TypedDict, closed=True):  # type: ignore[call-arg]
    scale: NotRequired[JSON]
    offset: NotRequired[JSON]


ScaleOffsetName = Literal["scale_offset"]


class ScaleOffsetJSON(NamedConfig[ScaleOffsetName, ScaleOffsetConfig]):
    """The JSON form(s) of the `scale_offset` codec"""


@dataclass(kw_only=True, frozen=True)
class ScaleOffset(ArrayArrayCodec):
    """Scale-offset array-to-array codec.

    Encodes values by subtracting an offset and multiplying by a scale factor.
    Decodes by dividing by the scale and adding the offset.

    All arithmetic uses the input array's data type semantics.

    Parameters
    ----------
    offset : float
        Value subtracted during encoding. Default is 0.
    scale : float
        Value multiplied during encoding (after offset subtraction). Default is 1.
    """

    is_fixed_size: bool = field(default=True, init=False)

    offset: float = 0
    scale: float = 1

    @classmethod
    def from_dict(cls, data: ScaleOffsetJSON) -> Self:  # type: ignore[override]
        scale: float = data.get("configuration", {}).get("scale", 1)  # type: ignore[assignment]
        offset: float = data.get("configuration", {}).get("offset", 0)  # type: ignore[assignment]
        return cls(scale=scale, offset=offset)

    def to_dict(self) -> ScaleOffsetJSON:  # type: ignore[override]
        if self.offset == 0 and self.scale == 1:
            return {"name": "scale_offset"}
        config: ScaleOffsetConfig = {}  #
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
        chunk_grid: ChunkGrid,
    ) -> None:
        native = dtype.to_native_dtype()
        if not np.issubdtype(native, np.integer) and not np.issubdtype(native, np.floating):
            raise ValueError(
                f"scale_offset codec only supports integer and floating-point data types. "
                f"Got {dtype}."
            )

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """
        Define the effect of this codec on the spec for an array. The only change is to update
        the output fill value by applying the scale + offset transformation.
        """
        native_dtype = chunk_spec.dtype.to_native_dtype()
        fill = chunk_spec.fill_value
        new_fill = (
            np.dtype(native_dtype).type(fill) - native_dtype.type(self.offset)
        ) * native_dtype.type(self.scale)
        return replace(chunk_spec, fill_value=new_fill)

    def _encode_sync(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        arr = chunk_array.as_ndarray_like()
        result = (arr - arr.dtype.type(self.offset)) * arr.dtype.type(self.scale)
        return chunk_array.__class__.from_ndarray_like(result)

    async def _encode_single(
        self,
        chunk_data: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_data, chunk_spec)

    def _decode_sync(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer:
        arr = chunk_array.as_ndarray_like()
        result = (arr / arr.dtype.type(self.scale)) + arr.dtype.type(self.offset)
        return chunk_array.__class__.from_ndarray_like(result)

    async def _decode_single(
        self,
        chunk_data: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._decode_sync(chunk_data, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length
