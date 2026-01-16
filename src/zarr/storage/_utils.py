from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable, Coroutine, Iterable, Mapping

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer

P = ParamSpec("P")
T_co = TypeVar("T_co", covariant=True)


def with_concurrency_limit(
    semaphore_attr: str = "_semaphore",
) -> Callable[[Callable[P, Coroutine[Any, Any, T_co]]], Callable[P, Coroutine[Any, Any, T_co]]]:
    """
    Decorator that applies a semaphore-based concurrency limit to an async method.

    This decorator is designed for Store methods that need to limit concurrent operations.
    The store instance should have a `_semaphore` attribute (or custom attribute name)
    that is either an asyncio.Semaphore or None (for unlimited concurrency).

    Parameters
    ----------
    semaphore_attr : str, optional
        Name of the semaphore attribute on the class instance. Default is "_semaphore".

    Returns
    -------
    Callable
        The decorated async function with concurrency limiting applied.

    Examples
    --------
    ```python
    class MyStore(Store):
        def __init__(self, concurrency_limit: int = 100):
            self._semaphore = asyncio.Semaphore(concurrency_limit) if concurrency_limit else None

        @with_concurrency_limit()
        async def get(self, key: str) -> Buffer | None:
            # This will only run when semaphore permits
            return await expensive_io_operation(key)
    ```
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, T_co]],
    ) -> Callable[P, Coroutine[Any, Any, T_co]]:
        """
        This decorator wraps the invocation of `func` in an `async with semaphore` context manager.
        The semaphore object is resolved by getting the `semaphor_attr` attribute from the first
        argument to func. When this decorator is used on a method of a class, that first argument
        is a reference to the class instance (`self`).
        """

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T_co:
            # First arg should be 'self'
            if not args:
                raise TypeError(f"{func.__name__} requires at least one argument (self)")

            self = args[0]

            semaphore: asyncio.Semaphore = getattr(self, semaphore_attr)

            # Apply concurrency limit
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def normalize_path(path: str | bytes | Path | None) -> str:
    if path is None:
        result = ""
    elif isinstance(path, bytes):
        result = str(path, "ascii")

    # handle pathlib.Path
    elif isinstance(path, Path):
        result = str(path)

    elif isinstance(path, str):
        result = path

    else:
        raise TypeError(f'Object {path} has an invalid type for "path": {type(path).__name__}')

    # convert backslash to forward slash
    result = result.replace("\\", "/")

    # remove leading and trailing slashes
    result = result.strip("/")

    # collapse any repeated slashes
    pat = re.compile(r"//+")
    result = pat.sub("/", result)

    # disallow path segments with just '.' or '..'
    segments = result.split("/")
    if any(s in {".", ".."} for s in segments):
        raise ValueError(
            f"The path {path!r} is invalid because its string representation contains '.' or '..' segments."
        )

    return result


def _normalize_byte_range_index(data: Buffer, byte_range: ByteRequest | None) -> tuple[int, int]:
    """
    Convert a ByteRequest into an explicit start and stop
    """
    if byte_range is None:
        start = 0
        stop = len(data) + 1
    elif isinstance(byte_range, RangeByteRequest):
        start = byte_range.start
        stop = byte_range.end
    elif isinstance(byte_range, OffsetByteRequest):
        start = byte_range.offset
        stop = len(data) + 1
    elif isinstance(byte_range, SuffixByteRequest):
        start = len(data) - byte_range.suffix
        stop = len(data) + 1
    else:
        raise ValueError(f"Unexpected byte_range, got {byte_range}.")
    return (start, stop)


def _join_paths(paths: Iterable[str]) -> str:
    """
    Filter out instances of '' and join the remaining strings with '/'.

    Parameters
    ----------
    paths : Iterable[str]

    Returns
    -------
    str

    Examples
    --------
    ```python
    from zarr.storage._utils import _join_paths
    _join_paths(["", "a", "b"])
    # 'a/b'
    _join_paths(["a", "b", "c"])
    # 'a/b/c'
    ```
    """
    return "/".join(filter(lambda v: v != "", paths))


def _relativize_path(*, path: str, prefix: str) -> str:
    """
    Make a "/"-delimited path relative to some prefix. If the prefix is '', then the path is
    returned as-is. Otherwise, the prefix is removed from the path as well as the separator
    string "/".

    If ``prefix`` is not the empty string and ``path`` does not start with ``prefix``
    followed by a "/" character, then an error is raised.

    This function assumes that the prefix does not end with "/".

    Parameters
    ----------
    path : str
        The path to make relative to the prefix.
    prefix : str
        The prefix to make the path relative to.

    Returns
    -------
    str

    Examples
    --------
    ```python
    from zarr.storage._utils import _relativize_path
    _relativize_path(path="a/b", prefix="")
    # 'a/b'
    _relativize_path(path="a/b/c", prefix="a/b")
    # 'c'
    ```
    """
    if prefix == "":
        return path
    else:
        _prefix = prefix + "/"
        if not path.startswith(_prefix):
            raise ValueError(f"The first component of {path} does not start with {prefix}.")
        return path.removeprefix(f"{prefix}/")


def _normalize_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """
    Normalize the input paths according to the normalization scheme used for zarr node paths.
    If any two paths normalize to the same value, raise a ValueError.
    """
    path_map: dict[str, str] = {}
    for path in paths:
        parsed = normalize_path(path)
        if parsed in path_map:
            msg = (
                f"After normalization, the value '{path}' collides with '{path_map[parsed]}'. "
                f"Both '{path}' and '{path_map[parsed]}' normalize to the same value: '{parsed}'. "
                f"You should use either '{path}' or '{path_map[parsed]}', but not both."
            )
            raise ValueError(msg)
        path_map[parsed] = path
    return tuple(path_map.keys())


T = TypeVar("T")


def _normalize_path_keys(data: Mapping[str, T]) -> dict[str, T]:
    """
    Normalize the keys of the input dict according to the normalization scheme used for zarr node
    paths. If any two keys in the input normalize to the same value, raise a ValueError.
    Returns a dict where the keys are the elements of the input and the values are the
    normalized form of each key.
    """
    parsed_keys = _normalize_paths(data.keys())
    return dict(zip(parsed_keys, data.values(), strict=True))
