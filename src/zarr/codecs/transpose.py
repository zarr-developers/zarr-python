from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Self, TypedDict, TypeGuard, cast, overload

import numpy as np
from typing_extensions import ReadOnly

from zarr.abc.codec import ArrayArrayCodec, CodecJSON
from zarr.core.array_spec import ArraySpec
from zarr.core.common import (
    JSON,
    NamedRequiredConfig,
    ZarrFormat,
)
from zarr.errors import CodecValidationError

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.buffer import NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


def parse_transpose_order(data: JSON | Iterable[int]) -> tuple[int, ...]:
    if not isinstance(data, Iterable):
        raise TypeError(f"Expected an iterable. Got {data} instead.")
    if not all(isinstance(a, int) for a in data):
        raise TypeError(f"Expected an iterable of integers. Got {data} instead.")
    return tuple(cast("Iterable[int]", data))


class TransposeConfig(TypedDict):
    order: tuple[int, ...]


class TransposeJSON_V2(TransposeConfig):
    """
    The JSON form of the Transpose codec in Zarr V2.
    """

    id: ReadOnly[Literal["transpose"]]


class TransposeJSON_V3(NamedRequiredConfig[Literal["transpose"], TransposeConfig]):
    """
    The JSON form of the Transpose codec in Zarr V3.
    """


def check_json_v2(data: object) -> TypeGuard[TransposeJSON_V2]:
    return (
        isinstance(data, Mapping)
        and set(data.keys()) == {"id", "configuration"}
        and data["id"] == "transpose"
        and isinstance(data["order"], Sequence)
        and not isinstance(data["order"], str)
    )


def check_json_v3(data: object) -> TypeGuard[TransposeJSON_V3]:
    return (
        isinstance(data, Mapping)
        and set(data.keys()) == {"name", "configuration"}
        and data["name"] == "transpose"
        and isinstance(data["configuration"], Mapping)
        and set(data["configuration"].keys()) == {"order"}
    )


@dataclass(frozen=True)
class TransposeCodec(ArrayArrayCodec):
    is_fixed_size = True

    order: tuple[int, ...]

    def __init__(self, *, order: Iterable[int]) -> None:
        order_parsed = parse_transpose_order(order)

        object.__setattr__(self, "order", order_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls.from_json(data)  # type: ignore[arg-type]

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        if check_json_v2(data):
            return cls(order=data["order"])
        msg = (
            "Invalid Zarr V2 JSON representation of the transpose codec. "
            f"Got {data!r}, expected a Mapping with keys ('id', 'order')"
        )
        raise CodecValidationError(msg)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON) -> Self:
        if check_json_v3(data):
            return cls(order=data["configuration"]["order"])
        msg = (
            "Invalid Zarr V3 JSON representation of the transpose codec. "
            f"Got {data!r}, expected a Mapping with keys ('name', 'configuration')"
            "Where the 'configuration' key is a Mapping with keys ('order')"
        )
        raise CodecValidationError(msg)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "transpose", "configuration": {"order": tuple(self.order)}}

    @overload
    def to_json(self, zarr_format: Literal[2]) -> TransposeJSON_V2: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> TransposeJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> TransposeJSON_V2 | TransposeJSON_V3:
        if zarr_format == 2:
            return {"id": "transpose", "order": self.order}
        elif zarr_format == 3:
            return {"name": "transpose", "configuration": {"order": self.order}}
        raise ValueError(
            f"Unsupported Zarr format {zarr_format}. Expected 2 or 3."
        )  # pragma: no cover

    def validate(
        self,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        if len(self.order) != len(shape):
            raise ValueError(
                f"The `order` tuple must have as many entries as there are dimensions in the array. Got {self.order}."
            )
        if len(self.order) != len(set(self.order)):
            raise ValueError(
                f"There must not be duplicates in the `order` tuple. Got {self.order}."
            )
        if not all(0 <= x < len(shape) for x in self.order):
            raise ValueError(
                f"All entries in the `order` tuple must be between 0 and the number of dimensions in the array. Got {self.order}."
            )

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        ndim = array_spec.ndim
        if len(self.order) != ndim:
            raise ValueError(
                f"The `order` tuple must have as many entries as there are dimensions in the array. Got {self.order}."
            )
        if len(self.order) != len(set(self.order)):
            raise ValueError(
                f"There must not be duplicates in the `order` tuple. Got {self.order}."
            )
        if not all(0 <= x < ndim for x in self.order):
            raise ValueError(
                f"All entries in the `order` tuple must be between 0 and the number of dimensions in the array. Got {self.order}."
            )
        order = tuple(self.order)

        if order != self.order:
            return replace(self, order=order)
        return self

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return ArraySpec(
            shape=tuple(chunk_spec.shape[self.order[i]] for i in range(chunk_spec.ndim)),
            dtype=chunk_spec.dtype,
            fill_value=chunk_spec.fill_value,
            config=chunk_spec.config,
            prototype=chunk_spec.prototype,
        )

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        inverse_order = np.argsort(self.order)
        return chunk_array.transpose(inverse_order)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        return chunk_array.transpose(self.order)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length
