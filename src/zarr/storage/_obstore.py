from __future__ import annotations

import asyncio
import pickle
from collections import defaultdict
from itertools import chain
from operator import itemgetter
from typing import TYPE_CHECKING, Literal, Self, TypedDict

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.common import concurrent_map
from zarr.core.config import config
from zarr.storage._utils import _relativize_path

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Iterable, Sequence
    from collections.abc import Buffer as BufferLike
    from typing import Any, Protocol

    from obspec import (
        DeleteAsync,
        GetAsync,
        GetRangeAsync,
        GetRangesAsync,
        HeadAsync,
        ListAsync,
        ListResult,
        ListWithDelimiterAsync,
        ObjectMeta,
        OffsetRange,
        PutAsync,
        SuffixRange,
    )

    from zarr.core.buffer import Buffer, BufferPrototype

    class ObspecInput(
        DeleteAsync,
        GetAsync,
        GetRangeAsync,
        GetRangesAsync,
        HeadAsync,
        ListAsync,
        ListWithDelimiterAsync,
        PutAsync,
        Protocol,
    ):
        """The union of the async obspec protocols that ``ObjectStore`` relies on.

        Any object with these methods can back an ``ObjectStore``; there is no
        requirement to inherit from anything. Keep ``_OBSPEC_METHODS`` in sync with
        the protocols listed here.
        """


__all__ = ["ObjectStore"]

# The methods of the ``ObspecInput`` protocol, checked structurally at runtime in
# ``ObjectStore.__init__``. obspec is only imported for type checking, so its
# protocol classes cannot be used for an ``isinstance`` check here.
_OBSPEC_METHODS: tuple[str, ...] = (
    "delete_async",
    "get_async",
    "get_range_async",
    "get_ranges_async",
    "head_async",
    "list_async",
    "list_with_delimiter_async",
    "put_async",
)

_ALLOWED_EXCEPTIONS: tuple[type[Exception], ...] = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
)

_ObspecErrorName = Literal["AlreadyExistsError", "NotFoundError", "NotSupportedError"]


def _is_obspec_error(exc: Exception, name: _ObspecErrorName) -> bool:
    """Check whether ``exc`` is the obspec exception called ``name``.

    obspec uses structural typing everywhere except for exceptions, which cannot be
    matched structurally. Instead, implementations raise exceptions with well-known
    class names and ``obspec.exceptions.map_exception`` resolves those names to the
    obspec exception classes. The builtin ``FileNotFoundError`` maps to
    ``NotFoundError``.

    obspec is imported lazily so that importing ``zarr.storage`` does not require it.
    """
    from obspec import exceptions

    return isinstance(exceptions.map_exception(exc), getattr(exceptions, name))


class ObjectStore[T_Store: "ObspecInput"](Store):
    """
    Store that reads and writes through any object store implementing the
    [obspec](https://developmentseed.org/obspec/) async protocols, such as the
    [obstore](https://developmentseed.org/obstore/) stores for AWS S3, Google
    Cloud Storage and Azure Blob Storage, or a wrapper (a cache, a request logger)
    around one of them.

    Parameters
    ----------
    store : ObspecInput
        Any object implementing the ``DeleteAsync``, ``GetAsync``, ``GetRangeAsync``,
        ``GetRangesAsync``, ``HeadAsync``, ``ListAsync``, ``ListWithDelimiterAsync``
        and ``PutAsync`` obspec protocols, set up with the proper credentials. The
        check is structural: the object must have those methods, but it does not
        have to inherit from anything.
    read_only : bool
        Whether to open the store in read-only mode.

    Warnings
    --------
    ObjectStore is experimental and subject to API changes without notice. Please
    raise an issue with any comments/concerns about the store.
    """

    store: T_Store
    """The underlying obspec-compatible store instance."""

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ObjectStore):
            return False

        if not self.read_only == value.read_only:
            return False

        return self.store == value.store  # type: ignore[no-any-return]

    def __init__(self, store: T_Store, *, read_only: bool = False) -> None:
        missing = [name for name in _OBSPEC_METHODS if not callable(getattr(store, name, None))]
        if missing:
            raise TypeError(
                f"expected an object implementing the obspec async store protocols, got "
                f"{store!r}, which is missing the following methods: {', '.join(missing)}"
            )
        try:
            import obspec  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ObjectStore requires the obspec package. Install it with "
                "'pip install obspec' or 'pip install zarr[remote]'."
            ) from e
        super().__init__(read_only=read_only)
        self.store = store

    def with_read_only(self, read_only: bool = False) -> Self:
        # docstring inherited
        return type(self)(
            store=self.store,
            read_only=read_only,
        )

    def __str__(self) -> str:
        return f"object_store://{self.store}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        state["store"] = pickle.dumps(self.store)
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        state["store"] = pickle.loads(state["store"])
        self.__dict__.update(state)

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        # docstring inherited
        if byte_range is not None and not isinstance(
            byte_range, RangeByteRequest | OffsetByteRequest | SuffixByteRequest
        ):
            raise ValueError(f"Unexpected byte_range, got {byte_range}")
        try:
            if byte_range is None:
                resp = await self.store.get_async(key)
                return prototype.buffer.from_bytes(await resp.buffer_async())  # type: ignore[arg-type]
            elif isinstance(byte_range, RangeByteRequest):
                bytes = await self.store.get_range_async(
                    key, start=byte_range.start, end=byte_range.end
                )
                return prototype.buffer.from_bytes(bytes)  # type: ignore[arg-type]
            elif isinstance(byte_range, OffsetByteRequest):
                resp = await self.store.get_async(
                    key, options={"range": {"offset": byte_range.offset}}
                )
                return prototype.buffer.from_bytes(await resp.buffer_async())  # type: ignore[arg-type]
            else:
                buffer = await _get_suffix(self.store, key, byte_range.suffix)
                return prototype.buffer.from_bytes(buffer)  # type: ignore[arg-type]
        except Exception as e:
            if isinstance(e, _ALLOWED_EXCEPTIONS) or _is_obspec_error(e, "NotFoundError"):
                return None
            raise

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # docstring inherited
        return await _get_partial_values(self.store, prototype=prototype, key_ranges=key_ranges)

    async def exists(self, key: str) -> bool:
        # docstring inherited
        try:
            await self.store.head_async(key)
        except Exception as e:
            if _is_obspec_error(e, "NotFoundError"):
                return False
            raise
        return True

    @property
    def supports_writes(self) -> bool:
        # docstring inherited
        return True

    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        self._check_writable()

        buf = value.as_buffer_like()
        await self.store.put_async(key, buf)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        # docstring inherited
        self._check_writable()
        buf = value.as_buffer_like()
        try:
            await self.store.put_async(key, buf, mode="create")
        except Exception as e:
            if not _is_obspec_error(e, "AlreadyExistsError"):
                raise

    @property
    def supports_deletes(self) -> bool:
        # docstring inherited
        return True

    async def delete(self, key: str) -> None:
        # docstring inherited
        self._check_writable()

        # Some stores such as local filesystems, GCP and Azure raise an error
        # when deleting a non-existent key, while others such as S3 and in-memory do
        # not. We suppress the error to make the behavior consistent across all
        # stores. This is also in line with the behavior of the other Zarr store adapters.
        try:
            await self.store.delete_async(key)
        except Exception as e:
            if not _is_obspec_error(e, "NotFoundError"):
                raise

    async def delete_dir(self, prefix: str) -> None:
        # docstring inherited
        self._check_writable()
        if prefix != "" and not prefix.endswith("/"):
            prefix += "/"

        keys = [(obj["path"],) async for obj in self._list(prefix)]
        await concurrent_map(keys, self.delete, limit=config.get("async.concurrency"))

    @property
    def supports_listing(self) -> bool:
        # docstring inherited
        return True

    async def _list(self, prefix: str | None = None) -> AsyncGenerator[ObjectMeta, None]:
        async for batch in self.store.list_async(prefix=prefix):
            for item in batch:
                yield item

    def list(self) -> AsyncGenerator[str, None]:
        # docstring inherited
        return (obj["path"] async for obj in self._list())

    def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        return (obj["path"] async for obj in self._list(prefix))

    def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        coroutine = self.store.list_with_delimiter_async(prefix=prefix)
        return _transform_list_dir(coroutine, prefix)

    async def getsize(self, key: str) -> int:
        # docstring inherited
        try:
            resp = await self.store.head_async(key)
        except Exception as e:
            # The Store contract is a FileNotFoundError; a store's own NotFoundError
            # may not derive from it.
            if not isinstance(e, FileNotFoundError) and _is_obspec_error(e, "NotFoundError"):
                raise FileNotFoundError(key) from e
            raise
        return resp["size"]

    async def getsize_prefix(self, prefix: str) -> int:
        # docstring inherited
        sizes = [obj["size"] async for obj in self._list(prefix=prefix)]
        return sum(sizes)


async def _get_suffix(store: ObspecInput, path: str, suffix: int) -> BufferLike:
    """Fetch the last ``suffix`` bytes of ``path``.

    Some object stores (Azure) don't support suffix requests. In this case, our
    workaround is to first get the length of the object and then manually request
    the byte range at the end.
    """
    try:
        resp = await store.get_async(path, options={"range": {"suffix": suffix}})
        return await resp.buffer_async()
    except Exception as e:
        if not _is_obspec_error(e, "NotSupportedError"):
            raise
    head_resp = await store.head_async(path)
    file_size = head_resp["size"]
    return await store.get_range_async(path, start=file_size - suffix, length=suffix)


async def _transform_list_dir(
    list_result_coroutine: Coroutine[Any, Any, ListResult[Sequence[ObjectMeta]]], prefix: str
) -> AsyncGenerator[str, None]:
    """
    Transform the result of list_with_delimiter into an async generator of paths.
    """
    list_result = await list_result_coroutine

    # We assume that the underlying object-store implementation correctly handles the
    # prefix, so we don't double-check that the returned results actually start with the
    # given prefix.
    prefix = prefix.rstrip("/")
    for path in chain(
        list_result["common_prefixes"], map(itemgetter("path"), list_result["objects"])
    ):
        if prefix != "" and path == prefix:
            continue
        relpath = _relativize_path(path=path, prefix=prefix)
        if relpath:
            yield relpath


class _BoundedRequest(TypedDict):
    """Range request with a known start and end byte.

    These requests can be multiplexed natively on the Rust side with
    `obstore.get_ranges_async`.
    """

    original_request_index: int
    """The positional index in the original key_ranges input"""

    start: int
    """Start byte offset."""

    end: int
    """End byte offset."""


class _OtherRequest(TypedDict):
    """Offset or suffix range requests.

    These requests cannot be concurrent on the Rust side, and each need their own call
    to `obstore.get_async`, passing in the `range` parameter.
    """

    original_request_index: int
    """The positional index in the original key_ranges input"""

    path: str
    """The path to request from."""

    range: OffsetRange | None
    # Note: suffix requests are handled separately because some object stores (Azure)
    # don't support them
    """The range request type."""


class _SuffixRequest(TypedDict):
    """Offset or suffix range requests.

    These requests cannot be concurrent on the Rust side, and each need their own call
    to `obstore.get_async`, passing in the `range` parameter.
    """

    original_request_index: int
    """The positional index in the original key_ranges input"""

    path: str
    """The path to request from."""

    range: SuffixRange
    """The suffix range."""


class _Response(TypedDict):
    """A response buffer associated with the original index that it should be restored to."""

    original_request_index: int
    """The positional index in the original key_ranges input"""

    buffer: Buffer
    """The buffer returned from obstore's range request."""


async def _make_bounded_requests(
    store: ObspecInput,
    path: str,
    requests: list[_BoundedRequest],
    prototype: BufferPrototype,
    semaphore: asyncio.Semaphore,
) -> list[_Response]:
    """Make all bounded requests for a specific file.

    `obstore.get_ranges_async` allows for making concurrent requests for multiple ranges
    within a single file, and will e.g. merge concurrent requests. This only uses one
    single Python coroutine.
    """
    starts = [r["start"] for r in requests]
    ends = [r["end"] for r in requests]
    async with semaphore:
        responses = await store.get_ranges_async(path=path, starts=starts, ends=ends)

    buffer_responses: list[_Response] = []
    for request, response in zip(requests, responses, strict=True):
        buffer_responses.append(
            {
                "original_request_index": request["original_request_index"],
                "buffer": prototype.buffer.from_bytes(response),  # type: ignore[arg-type]
            }
        )

    return buffer_responses


async def _make_other_request(
    store: ObspecInput,
    request: _OtherRequest,
    prototype: BufferPrototype,
    semaphore: asyncio.Semaphore,
) -> list[_Response]:
    """Make offset or full-file requests.

    We return a `list[_Response]` for symmetry with `_make_bounded_requests` so that all
    futures can be gathered together.
    """
    async with semaphore:
        if request["range"] is None:
            resp = await store.get_async(request["path"])
        else:
            resp = await store.get_async(request["path"], options={"range": request["range"]})
        buffer = await resp.buffer_async()

    return [
        {
            "original_request_index": request["original_request_index"],
            "buffer": prototype.buffer.from_bytes(buffer),  # type: ignore[arg-type]
        }
    ]


async def _make_suffix_request(
    store: ObspecInput,
    request: _SuffixRequest,
    prototype: BufferPrototype,
    semaphore: asyncio.Semaphore,
) -> list[_Response]:
    """Make suffix requests.

    This is separated out from `_make_other_request` because some object stores (Azure)
    don't support suffix requests; see `_get_suffix` for the workaround.

    We return a `list[_Response]` for symmetry with `_make_bounded_requests` so that all
    futures can be gathered together.
    """
    async with semaphore:
        buffer = await _get_suffix(store, request["path"], request["range"]["suffix"])

    return [
        {
            "original_request_index": request["original_request_index"],
            "buffer": prototype.buffer.from_bytes(buffer),  # type: ignore[arg-type]
        }
    ]


async def _get_partial_values(
    store: ObspecInput,
    prototype: BufferPrototype,
    key_ranges: Iterable[tuple[str, ByteRequest | None]],
) -> list[Buffer | None]:
    """Make multiple range requests.

    ObjectStore has a `get_ranges` method that will additionally merge nearby ranges,
    but it's _per_ file. So we need to split these key_ranges into **per-file** key
    ranges, and then reassemble the results in the original order.

    We separate into different requests:

    - One call to `obstore.get_ranges_async` **per target file**
    - One call to `obstore.get_async` for each other request.
    """
    key_ranges = list(key_ranges)
    per_file_bounded_requests: dict[str, list[_BoundedRequest]] = defaultdict(list)
    other_requests: list[_OtherRequest] = []
    suffix_requests: list[_SuffixRequest] = []

    for idx, (path, byte_range) in enumerate(key_ranges):
        if byte_range is None:
            other_requests.append(
                {
                    "original_request_index": idx,
                    "path": path,
                    "range": None,
                }
            )
        elif isinstance(byte_range, RangeByteRequest):
            per_file_bounded_requests[path].append(
                {"original_request_index": idx, "start": byte_range.start, "end": byte_range.end}
            )
        elif isinstance(byte_range, OffsetByteRequest):
            other_requests.append(
                {
                    "original_request_index": idx,
                    "path": path,
                    "range": {"offset": byte_range.offset},
                }
            )
        elif isinstance(byte_range, SuffixByteRequest):
            suffix_requests.append(
                {
                    "original_request_index": idx,
                    "path": path,
                    "range": {"suffix": byte_range.suffix},
                }
            )
        else:
            raise ValueError(f"Unsupported range input: {byte_range}")

    semaphore = asyncio.Semaphore(config.get("async.concurrency"))

    futs: list[Coroutine[Any, Any, list[_Response]]] = []
    for path, bounded_ranges in per_file_bounded_requests.items():
        futs.append(
            _make_bounded_requests(store, path, bounded_ranges, prototype, semaphore=semaphore)
        )

    for request in other_requests:
        futs.append(_make_other_request(store, request, prototype, semaphore=semaphore))  # noqa: PERF401

    for suffix_request in suffix_requests:
        futs.append(_make_suffix_request(store, suffix_request, prototype, semaphore=semaphore))  # noqa: PERF401

    buffers: list[Buffer | None] = [None] * len(key_ranges)

    for responses in await asyncio.gather(*futs):
        for resp in responses:
            buffers[resp["original_request_index"]] = resp["buffer"]

    return buffers
