from __future__ import annotations

import asyncio
import contextvars
import functools
import operator
from collections.abc import Iterable
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

import numpy as np
import numpy.typing as npt

ZARR_JSON = "zarr.json"
ZARRAY_JSON = ".zarray"
ZGROUP_JSON = ".zgroup"
ZATTRS_JSON = ".zattrs"

BytesLike = bytes | bytearray | memoryview
ChunkCoords = tuple[int, ...]
ChunkCoordsLike = Iterable[int]
SliceSelection = tuple[slice, ...]
Selection = slice | SliceSelection
ZarrFormat = Literal[2, 3]
JSON = None | str | int | float | Enum | dict[str, "JSON"] | list["JSON"] | tuple["JSON", ...]
MemoryOrder = Literal["C", "F"]
OpenMode = Literal["r", "r+", "a", "w", "w-"]


def product(tup: ChunkCoords) -> int:
    return functools.reduce(operator.mul, tup, 1)


T = TypeVar("T", bound=tuple[Any, ...])
V = TypeVar("V")


async def concurrent_map(
    items: list[T], func: Callable[..., Awaitable[V]], limit: int | None = None
) -> list[V]:
    if limit is None:
        return await asyncio.gather(*[func(*item) for item in items])

    else:
        sem = asyncio.Semaphore(limit)

        async def run(item: tuple[Any]) -> V:
            async with sem:
                return await func(*item)

        return await asyncio.gather(*[asyncio.ensure_future(run(item)) for item in items])


P = ParamSpec("P")
U = TypeVar("U")


async def to_thread(func: Callable[P, U], /, *args: P.args, **kwargs: P.kwargs) -> U:
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)


E = TypeVar("E", bound=Enum)


def enum_names(enum: type[E]) -> Iterator[str]:
    for item in enum:
        yield item.name


def parse_enum(data: JSON, cls: type[E]) -> E:
    if isinstance(data, cls):
        return data
    if not isinstance(data, str):
        raise TypeError(f"Expected str, got {type(data)}")
    if data in enum_names(cls):
        return cls(data)
    raise ValueError(f"Value must be one of {list(enum_names(cls))!r}. Got {data} instead.")


def parse_name(data: JSON, expected: str | None = None) -> str:
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
    data: JSON, expected_name: str | None = None
) -> tuple[str, dict[str, JSON]]: ...


@overload
def parse_named_configuration(
    data: JSON, expected_name: str | None = None, *, require_configuration: bool = True
) -> tuple[str, dict[str, JSON] | None]: ...


def parse_named_configuration(
    data: JSON, expected_name: str | None = None, *, require_configuration: bool = True
) -> tuple[str, JSON | None]:
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


def parse_shapelike(data: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(data, int):
        return (data,)
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


def parse_dtype(data: npt.DTypeLike) -> np.dtype[Any]:
    # todo: real validation
    return np.dtype(data)


def parse_fill_value(data: Any) -> Any:
    # todo: real validation
    return data


def parse_order(data: Any) -> Literal["C", "F"]:
    if data in ("C", "F"):
        return cast(Literal["C", "F"], data)
    raise ValueError(f"Expected one of ('C', 'F'), got {data} instead.")
