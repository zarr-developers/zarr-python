from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
import contextvars
from dataclasses import dataclass
from enum import Enum
import functools
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

ZARR_JSON = "zarr.json"
ZARRAY_JSON = ".zarray"
ZGROUP_JSON = ".zgroup"
ZATTRS_JSON = ".zattrs"

BytesLike = Union[bytes, bytearray, memoryview]
ChunkCoords = Tuple[int, ...]
SliceSelection = Tuple[slice, ...]
Selection = Union[slice, SliceSelection]
JSON = Union[str, None, int, float, Dict[str, "JSON"], List["JSON"]]


def product(tup: ChunkCoords) -> int:
    return functools.reduce(lambda x, y: x * y, tup, 1)


T = TypeVar("T", bound=Tuple)
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


def enum_names(enum: Enum) -> Iterator[str]:
    for item in enum:
        yield item.name


E = TypeVar("E", bound=Enum)


def parse_enum(data: JSON, cls: Type[E]) -> E:
    if isinstance(data, cls):
        return data
    if data in enum_names(cls):
        return cls(data)
    msg = f"Value must be one of {repr(list(enum_names(cls)))}, got {data} instead."
    raise ValueError(msg)


class NamedConfig(Protocol):
    @property
    def name(self) -> str:
        pass

    @property
    def configuration(self) -> Dict[str, Any]:
        pass


@dataclass(frozen=True)
class RuntimeConfiguration:
    order: Literal["C", "F"] = "C"
    concurrency: Optional[int] = None
    asyncio_loop: Optional[AbstractEventLoop] = None


@dataclass(frozen=True)
class ArraySpec:
    shape: ChunkCoords
    dtype: np.dtype
    fill_value: Any

    def __init__(self, shape, dtype, fill_value):
        shape_parsed = parse_shapelike(shape)
        dtype_parsed = parse_dtype(dtype)
        fill_value_parsed = parse_fill_value(fill_value)

        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "dtype", dtype_parsed)
        object.__setattr__(self, "fill_value", fill_value_parsed)

    @property
    def ndim(self) -> int:
        return len(self.shape)


def parse_name(data: JSON, expected: str) -> str:
    if data == expected:
        return data
    raise ValueError(f"Expected '{expected}' chunk, got {data} instead.")


def parse_shapelike(data: Any) -> Tuple[int, ...]:
    if not isinstance(data, Iterable):
        raise TypeError(f"Expected an iterable. Got {data} instead.")
    if not all(isinstance(a, int) for a in data):
        raise TypeError(f"Expected an iterable of integers. Got {data} instead.")
    return tuple(data)


def parse_dtype(data: Any) -> np.dtype:
    # todo: real validation
    return np.dtype(data)


def parse_fill_value(data: Any) -> Any:
    # todo: real validation
    return data
