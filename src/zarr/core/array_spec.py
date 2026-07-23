from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, cast

import numpy as np

from zarr.core.common import (
    MemoryOrder,
    parse_bool,
    parse_fill_value,
    parse_int,
    parse_order,
    parse_shapelike,
)
from zarr.core.config import config as zarr_config

if TYPE_CHECKING:
    from typing import NotRequired

    from zarr.core.buffer import BufferPrototype
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


class ArrayConfigParams(TypedDict):
    """
    A TypedDict model of the attributes of an ArrayConfig class, but with no required fields.
    This allows for partial construction of an ArrayConfig, with the assumption that the unset
    keys will be taken from a global configuration.
    """

    order: NotRequired[MemoryOrder]
    write_empty_chunks: NotRequired[bool]
    read_missing_chunks: NotRequired[bool]
    sharding_coalesce_max_gap_bytes: NotRequired[int]
    sharding_coalesce_max_bytes: NotRequired[int]


@dataclass(frozen=True)
class ArrayConfig:
    """
    A model of the runtime configuration of an array.

    Parameters
    ----------
    order : MemoryOrder
        The memory layout of the arrays returned when reading data from the store.
    write_empty_chunks : bool
        If True, empty chunks will be written to the store.
    read_missing_chunks : bool
        If True, missing chunks will be filled with the array's fill value on read.
        If False, reading missing chunks will raise a ``ChunkNotFoundError``.
    sharding_coalesce_max_gap_bytes : int
        When reading multiple chunks from the same shard, nearby byte ranges
        separated by no more than this many bytes are coalesced into a single
        request to the store.
    sharding_coalesce_max_bytes : int
        Requests will not be coalesced if doing so would exceed this byte size.
    """

    order: MemoryOrder
    write_empty_chunks: bool
    read_missing_chunks: bool
    sharding_coalesce_max_gap_bytes: int
    sharding_coalesce_max_bytes: int

    def __init__(
        self,
        order: MemoryOrder,
        write_empty_chunks: bool,
        *,
        read_missing_chunks: bool = True,
        sharding_coalesce_max_gap_bytes: int = 1 << 20,  # 1 MiB
        sharding_coalesce_max_bytes: int = 16 << 20,  # 16 MiB
    ) -> None:
        order_parsed = parse_order(order)
        write_empty_chunks_parsed = parse_bool(write_empty_chunks)
        read_missing_chunks_parsed = parse_bool(read_missing_chunks)
        sharding_coalesce_max_gap_bytes_parsed = parse_int(sharding_coalesce_max_gap_bytes)
        sharding_coalesce_max_bytes_parsed = parse_int(sharding_coalesce_max_bytes)

        object.__setattr__(self, "order", order_parsed)
        object.__setattr__(self, "write_empty_chunks", write_empty_chunks_parsed)
        object.__setattr__(self, "read_missing_chunks", read_missing_chunks_parsed)
        object.__setattr__(
            self, "sharding_coalesce_max_gap_bytes", sharding_coalesce_max_gap_bytes_parsed
        )
        object.__setattr__(self, "sharding_coalesce_max_bytes", sharding_coalesce_max_bytes_parsed)

    @classmethod
    def from_dict(cls, data: ArrayConfigParams) -> Self:
        """
        Create an ArrayConfig from a dict. The keys of that dict are a subset of the
        attributes of the ArrayConfig class. Any keys missing from that dict will be set to the
        the values in the ``array`` namespace of ``zarr.config``.
        """
        kwargs_out: ArrayConfigParams = {}
        for f in fields(ArrayConfig):
            field_name = cast(
                "Literal['order', 'write_empty_chunks', 'read_missing_chunks', 'sharding_coalesce_max_gap_bytes', 'sharding_coalesce_max_bytes']",
                f.name,
            )
            if field_name not in data:
                kwargs_out[field_name] = zarr_config.get(f"array.{field_name}")
            else:
                kwargs_out[field_name] = data[field_name]
        return cls(**kwargs_out)

    def to_dict(self) -> ArrayConfigParams:
        """
        Serialize an instance of this class to a dict.
        """
        return {
            "order": self.order,
            "write_empty_chunks": self.write_empty_chunks,
            "read_missing_chunks": self.read_missing_chunks,
            "sharding_coalesce_max_gap_bytes": self.sharding_coalesce_max_gap_bytes,
            "sharding_coalesce_max_bytes": self.sharding_coalesce_max_bytes,
        }


ArrayConfigLike = ArrayConfig | ArrayConfigParams


def parse_array_config(data: ArrayConfigLike | None) -> ArrayConfig:
    """
    Convert various types of data to an ArrayConfig.
    """
    if data is None:
        return ArrayConfig.from_dict({})
    elif isinstance(data, ArrayConfig):
        return data
    else:
        return ArrayConfig.from_dict(data)


@dataclass(frozen=True, eq=False)
class ArraySpec:
    shape: tuple[int, ...]
    dtype: ZDType[TBaseDType, TBaseScalar]
    fill_value: Any
    config: ArrayConfig
    prototype: BufferPrototype

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        fill_value: Any,
        config: ArrayConfig,
        prototype: BufferPrototype,
    ) -> None:
        shape_parsed = parse_shapelike(shape)
        fill_value_parsed = parse_fill_value(fill_value)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "prototype", prototype)

    def _key(self) -> tuple[object, ...]:
        """Returns the tuple used for equality/hash identity."""
        fill_value = self.fill_value
        if isinstance(fill_value, np.generic):
            # fill_values should be byte-identical, otherwise they correspond to different values in memory / on disk.
            # Importantly, this ensures np.nan == np.nan, NaT == NaT, and -0.0 != 0.0.
            # It also fixes np.void fill_values being unhashable (#3054).
            fill_value = fill_value.tobytes()
        return (self.shape, self.dtype, fill_value, self.config, self.prototype)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArraySpec):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def order(self) -> MemoryOrder:
        return self.config.order
