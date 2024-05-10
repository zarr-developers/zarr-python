from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterable, Union, cast

from dataclasses import dataclass, replace

from zarr.common import JSON, ArraySpec, ChunkCoordsLike, parse_named_configuration

if TYPE_CHECKING:
    from zarr.config import RuntimeConfiguration
    from typing import TYPE_CHECKING, Optional, Tuple
    from typing_extensions import Self

import numpy.typing as npt

from zarr.abc.codec import ArrayArrayCodec
from zarr.codecs.registry import register_codec


def parse_transpose_order(data: Union[JSON, Iterable[int]]) -> Tuple[int, ...]:
    if not isinstance(data, Iterable):
        raise TypeError(f"Expected an iterable. Got {data} instead.")
    if not all(isinstance(a, int) for a in data):
        raise TypeError(f"Expected an iterable of integers. Got {data} instead.")
    return tuple(cast(Iterable[int], data))


@dataclass(frozen=True)
class TransposeCodec(ArrayArrayCodec):
    is_fixed_size = True

    order: Tuple[int, ...]

    def __init__(self, *, order: ChunkCoordsLike) -> None:
        order_parsed = parse_transpose_order(order)

        object.__setattr__(self, "order", order_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "transpose")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, JSON]:
        return {"name": "transpose", "configuration": {"order": list(self.order)}}

    def evolve(self, array_spec: ArraySpec) -> Self:
        if len(self.order) != array_spec.ndim:
            raise ValueError(
                "The `order` tuple needs have as many entries as "
                + f"there are dimensions in the array. Got {self.order}."
            )
        if len(self.order) != len(set(self.order)):
            raise ValueError(
                f"There must not be duplicates in the `order` tuple. Got {self.order}."
            )
        if not all(0 <= x < array_spec.ndim for x in self.order):
            raise ValueError(
                "All entries in the `order` tuple must be between 0 and "
                + f"the number of dimensions in the array. Got {self.order}."
            )
        order = tuple(self.order)

        if order != self.order:
            return replace(self, order=order)
        return self

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        from zarr.common import ArraySpec

        return ArraySpec(
            shape=tuple(chunk_spec.shape[self.order[i]] for i in range(chunk_spec.ndim)),
            dtype=chunk_spec.dtype,
            fill_value=chunk_spec.fill_value,
        )

    async def decode(
        self,
        chunk_array: npt.NDArray[Any],
        chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> npt.NDArray[Any]:
        inverse_order = [0] * chunk_spec.ndim
        for x, i in enumerate(self.order):
            inverse_order[x] = i
        chunk_array = chunk_array.transpose(inverse_order)
        return chunk_array

    async def encode(
        self,
        chunk_array: npt.NDArray[Any],
        chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[npt.NDArray[Any]]:
        chunk_array = chunk_array.transpose(self.order)
        return chunk_array

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


register_codec("transpose", TransposeCodec)
