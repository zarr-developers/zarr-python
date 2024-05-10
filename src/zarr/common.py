from __future__ import annotations

import asyncio
import contextvars
import functools
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Tuple,
    TypeVar,
    Union,
    overload,
)

if TYPE_CHECKING:
    from typing import Any, Awaitable, Callable, Iterator, Optional, Type

import numpy as np

ZARR_JSON = "zarr.json"
ZARRAY_JSON = ".zarray"
ZGROUP_JSON = ".zgroup"
ZATTRS_JSON = ".zattrs"

BytesLike = Union[bytes, bytearray, memoryview]
ChunkCoords = Tuple[int, ...]
ChunkCoordsLike = Iterable[int]
SliceSelection = Tuple[slice, ...]
Selection = Union[slice, SliceSelection]
JSON = Union[str, None, int, float, Enum, Dict[str, "JSON"], List["JSON"], Tuple["JSON", ...]]


def product(tup: ChunkCoords) -> int:
    return functools.reduce(lambda x, y: x * y, tup, 1)


T = TypeVar("T", bound=Tuple[Any, ...])
V = TypeVar("V")


async def concurrent_map(
    items: List[T], func: Callable[..., Awaitable[V]], limit: Optional[int] = None
) -> List[V]:
    if limit is None:
        return await asyncio.gather(*[func(*item) for item in items])

    else:
        sem = asyncio.Semaphore(limit)

        async def run(item):
            async with sem:
                return await func(*item)

        return await asyncio.gather(*[asyncio.ensure_future(run(item)) for item in items])


async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)


E = TypeVar("E", bound=Enum)


def enum_names(enum: Type[E]) -> Iterator[str]:
    for item in enum:
        yield item.name


def parse_enum(data: JSON, cls: Type[E]) -> E:
    if isinstance(data, cls):
        return data
    if not isinstance(data, str):
        raise TypeError(f"Expected str, got {type(data)}")
    if data in enum_names(cls):
        return cls(data)
    raise ValueError(f"Value must be one of {repr(list(enum_names(cls)))}. Got {data} instead.")


@dataclass(frozen=True)
class ArraySpec:
    shape: ChunkCoords
    dtype: np.dtype[Any]
    fill_value: Any
    order: Literal["C", "F"]

    def __init__(
        self, shape: ChunkCoords, dtype: np.dtype[Any], fill_value: Any, order: Literal["C", "F"]
    ) -> None:
        shape_parsed = parse_shapelike(shape)
        dtype_parsed = parse_dtype(dtype)
        fill_value_parsed = parse_fill_value(fill_value)
        order_parsed = parse_order(order)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "dtype", dtype_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "order", order_parsed)

    @property
    def ndim(self) -> int:
        return len(self.shape)


def parse_name(data: JSON, expected: Optional[str] = None) -> str:
    if isinstance(data, str):
        if expected is None or data == expected:
            return data
        raise ValueError(f"Expected '{expected}'. Got {data} instead.")
    else:
        raise TypeError(f"Expected a string, got an instance of {type(data)}.")


def parse_configuration(data: JSON) -> JSON:
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    return data


@overload
def parse_named_configuration(
    data: JSON, expected_name: Optional[str] = None
) -> Tuple[str, Dict[str, JSON]]: ...


@overload
def parse_named_configuration(
    data: JSON, expected_name: Optional[str] = None, *, require_configuration: bool = True
) -> Tuple[str, Optional[Dict[str, JSON]]]: ...


def parse_named_configuration(
    data: JSON, expected_name: Optional[str] = None, *, require_configuration: bool = True
) -> Tuple[str, Optional[JSON]]:
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    if "name" not in data:
        raise ValueError(f"Named configuration does not have a 'name' key. Got {data}.")
    name_parsed = parse_name(data["name"], expected_name)
    if "configuration" in data:
        configuration_parsed = parse_configuration(data["configuration"])
    elif require_configuration:
        raise ValueError(f"Named configuration does not have a 'configuration' key. Got {data}.")
    else:
        configuration_parsed = None
    return name_parsed, configuration_parsed


def parse_shapelike(data: Any) -> Tuple[int, ...]:
    if not isinstance(data, Iterable):
        raise TypeError(f"Expected an iterable. Got {data} instead.")
    data_tuple = tuple(data)
    if len(data_tuple) == 0:
        raise ValueError("Expected at least one element. Got 0.")
    if not all(isinstance(v, int) for v in data_tuple):
        msg = f"Expected an iterable of integers. Got {type(data)} instead."
        raise TypeError(msg)
    if not all(lambda v: v > 0 for v in data_tuple):
        raise ValueError(f"All values must be greater than 0. Got {data}.")
    return data_tuple


def parse_dtype(data: Any) -> np.dtype[Any]:
    # todo: real validation
    return np.dtype(data)


def parse_fill_value(data: Any) -> Any:
    # todo: real validation
    return data


def parse_order(data: Any) -> Literal["C", "F"]:
    if data in ("C", "F"):
        return data
    raise ValueError(f"Expected one of ('C', 'F'), got {data} instead.")
