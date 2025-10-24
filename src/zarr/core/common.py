from __future__ import annotations

import asyncio
import functools
import math
import operator
import threading
import warnings
import weakref
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from itertools import starmap
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
    NotRequired,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import ReadOnly

from zarr.core.config import config as zarr_config
from zarr.errors import ZarrRuntimeWarning

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator


ZARR_JSON = "zarr.json"
ZARRAY_JSON = ".zarray"
ZGROUP_JSON = ".zgroup"
ZATTRS_JSON = ".zattrs"
ZMETADATA_V2_JSON = ".zmetadata"

BytesLike = bytes | bytearray | memoryview
ShapeLike = Iterable[int] | int
# For backwards compatibility
ChunkCoords = tuple[int, ...]
ZarrFormat = Literal[2, 3]
NodeType = Literal["array", "group"]
JSON = str | int | float | Mapping[str, "JSON"] | Sequence["JSON"] | None
MemoryOrder = Literal["C", "F"]
AccessModeLiteral = Literal["r", "r+", "a", "w", "w-"]
ANY_ACCESS_MODE: Final = "r", "r+", "a", "w", "w-"
DimensionNames = Iterable[str | None] | None

TName = TypeVar("TName", bound=str)
TConfig = TypeVar("TConfig", bound=Mapping[str, object])


class NamedConfig(TypedDict, Generic[TName, TConfig]):
    """
    A typed dictionary representing an object with a name and configuration, where the configuration
    is an optional mapping of string keys to values, e.g. another typed dictionary or a JSON object.

    This class is generic with two type parameters: the type of the name (``TName``) and the type of
    the configuration (``TConfig``).
    """

    name: ReadOnly[TName]
    """The name of the object."""

    configuration: NotRequired[ReadOnly[TConfig]]
    """The configuration of the object. Not required."""


class NamedRequiredConfig(TypedDict, Generic[TName, TConfig]):
    """
    A typed dictionary representing an object with a name and configuration, where the configuration
    is a mapping of string keys to values, e.g. another typed dictionary or a JSON object.

    This class is generic with two type parameters: the type of the name (``TName``) and the type of
    the configuration (``TConfig``).
    """

    name: ReadOnly[TName]
    """The name of the object."""

    configuration: ReadOnly[TConfig]
    """The configuration of the object."""


def product(tup: tuple[int, ...]) -> int:
    return functools.reduce(operator.mul, tup, 1)


def ceildiv(a: float, b: float) -> int:
    if a == 0:
        return 0
    return math.ceil(a / b)


T = TypeVar("T", bound=tuple[Any, ...])
V = TypeVar("V")


# Global semaphore management for per-process concurrency limiting
# Use WeakKeyDictionary to automatically clean up semaphores when event loops are garbage collected
_global_semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)
# Use threading.Lock instead of asyncio.Lock to coordinate across event loops
_global_semaphore_lock = threading.Lock()


def get_global_semaphore() -> asyncio.Semaphore:
    """
    Get the global semaphore for the current event loop.

    This ensures that all concurrent operations across the process share the same
    concurrency limit, preventing excessive concurrent task creation when multiple
    arrays or operations are running simultaneously.

    The semaphore is lazily created per event loop and uses the configured
    `async.concurrency` value from zarr config. The semaphore is cached per event
    loop, so subsequent calls return the same semaphore instance.

    Note: Config changes after the first call will not affect the semaphore limit.
    To apply new config values, use :func:`reset_global_semaphores` to clear the cache.

    Returns
    -------
    asyncio.Semaphore
        The global semaphore for this event loop.

    Raises
    ------
    RuntimeError
        If called outside of an async context (no running event loop).

    See Also
    --------
    reset_global_semaphores : Clear the global semaphore cache
    """
    loop = asyncio.get_running_loop()

    # Acquire lock FIRST to prevent TOCTOU race condition
    with _global_semaphore_lock:
        if loop not in _global_semaphores:
            limit = zarr_config.get("async.concurrency")
            _global_semaphores[loop] = asyncio.Semaphore(limit)
        return _global_semaphores[loop]


def reset_global_semaphores() -> None:
    """
    Clear all cached global semaphores.

    This is useful when you want config changes to take effect, or for testing.
    The next call to :func:`get_global_semaphore` will create a new semaphore
    using the current configuration.

    Warning: This should only be called when no async operations are in progress,
    as it will invalidate all existing semaphore references.

    Examples
    --------
    >>> import zarr
    >>> zarr.config.set({"async.concurrency": 50})
    >>> reset_global_semaphores()  # Apply new config
    """
    with _global_semaphore_lock:
        _global_semaphores.clear()


async def concurrent_map(
    items: Iterable[T],
    func: Callable[..., Awaitable[V]],
    limit: int | None = None,
    *,
    use_global_semaphore: bool = True,
) -> list[V]:
    """
    Execute an async function concurrently over multiple items with concurrency limiting.

    Parameters
    ----------
    items : Iterable[T]
        Items to process, where each item is a tuple of arguments to pass to func.
    func : Callable[..., Awaitable[V]]
        Async function to execute for each item.
    limit : int | None, optional
        If provided and use_global_semaphore is False, creates a local semaphore
        with this limit. If None, no concurrency limiting is applied.
    use_global_semaphore : bool, default True
        If True, uses the global per-process semaphore for concurrency limiting,
        ensuring all concurrent operations share the same limit. If False, uses
        the `limit` parameter for local limiting (legacy behavior).

    Returns
    -------
    list[V]
        Results from executing func on all items.
    """
    if use_global_semaphore:
        if limit is not None:
            raise ValueError(
                "Cannot specify both use_global_semaphore=True and a limit value. "
                "Either use the global semaphore (use_global_semaphore=True, limit=None) "
                "or specify a local limit (use_global_semaphore=False, limit=<int>)."
            )
        # Use the global semaphore for process-wide concurrency limiting
        sem = get_global_semaphore()

        async def run(item: tuple[Any]) -> V:
            async with sem:
                return await func(*item)

        return await asyncio.gather(*[asyncio.ensure_future(run(item)) for item in items])

    elif limit is None:
        # No concurrency limiting
        return await asyncio.gather(*list(starmap(func, items)))

    else:
        # Legacy mode: create local semaphore with specified limit
        sem = asyncio.Semaphore(limit)

        async def run(item: tuple[Any]) -> V:
            async with sem:
                return await func(*item)

        return await asyncio.gather(*[asyncio.ensure_future(run(item)) for item in items])


E = TypeVar("E", bound=Enum)


def enum_names(enum: type[E]) -> Iterator[str]:
    for item in enum:
        yield item.name


def parse_enum(data: object, cls: type[E]) -> E:
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
    data: JSON | NamedConfig[str, Any], expected_name: str | None = None
) -> tuple[str, dict[str, JSON]]: ...


@overload
def parse_named_configuration(
    data: JSON | NamedConfig[str, Any],
    expected_name: str | None = None,
    *,
    require_configuration: bool = True,
) -> tuple[str, dict[str, JSON] | None]: ...


def parse_named_configuration(
    data: JSON | NamedConfig[str, Any],
    expected_name: str | None = None,
    *,
    require_configuration: bool = True,
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


def parse_shapelike(data: ShapeLike) -> tuple[int, ...]:
    if isinstance(data, int):
        if data < 0:
            raise ValueError(f"Expected a non-negative integer. Got {data} instead")
        return (data,)
    try:
        data_tuple = tuple(data)
    except TypeError as e:
        msg = f"Expected an integer or an iterable of integers. Got {data} instead."
        raise TypeError(msg) from e

    if not all(isinstance(v, int) for v in data_tuple):
        msg = f"Expected an iterable of integers. Got {data} instead."
        raise TypeError(msg)
    if not all(v > -1 for v in data_tuple):
        msg = f"Expected all values to be non-negative. Got {data} instead."
        raise ValueError(msg)
    return data_tuple


def parse_fill_value(data: Any) -> Any:
    # todo: real validation
    return data


def parse_order(data: Any) -> Literal["C", "F"]:
    if data in ("C", "F"):
        return cast("Literal['C', 'F']", data)
    raise ValueError(f"Expected one of ('C', 'F'), got {data} instead.")


def parse_bool(data: Any) -> bool:
    if isinstance(data, bool):
        return data
    raise ValueError(f"Expected bool, got {data} instead.")


def _warn_write_empty_chunks_kwarg() -> None:
    # TODO: link to docs page on array configuration in this message
    msg = (
        "The `write_empty_chunks` keyword argument is deprecated and will be removed in future versions. "
        "To control whether empty chunks are written to storage, either use the `config` keyword "
        "argument, as in `config={'write_empty_chunks': True}`,"
        "or change the global 'array.write_empty_chunks' configuration variable."
    )
    warnings.warn(msg, ZarrRuntimeWarning, stacklevel=2)


def _warn_order_kwarg() -> None:
    # TODO: link to docs page on array configuration in this message
    msg = (
        "The `order` keyword argument has no effect for Zarr format 3 arrays. "
        "To control the memory layout of the array, either use the `config` keyword "
        "argument, as in `config={'order': 'C'}`,"
        "or change the global 'array.order' configuration variable."
    )
    warnings.warn(msg, ZarrRuntimeWarning, stacklevel=2)


def _default_zarr_format() -> ZarrFormat:
    """Return the default zarr_version"""
    return cast("ZarrFormat", int(zarr_config.get("default_zarr_format", 3)))
