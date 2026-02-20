from __future__ import annotations

import asyncio
import contextlib
import pickle
from collections import defaultdict
from itertools import chain
from operator import itemgetter
from typing import TYPE_CHECKING, Generic, Self, TypeVar

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.storage._utils import ConcurrencyLimiter, _relativize_path, with_concurrency_limit

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Iterable, Sequence
    from typing import Any

    from obstore import ListResult, ListStream, ObjectMeta
    from obstore.store import ObjectStore as _UpstreamObjectStore

    from zarr.core.buffer import Buffer, BufferPrototype

__all__ = ["ObjectStore"]

_ALLOWED_EXCEPTIONS: tuple[type[Exception], ...] = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
)


T_Store = TypeVar("T_Store", bound="_UpstreamObjectStore")


class ObjectStore(Store, ConcurrencyLimiter, Generic[T_Store]):
    """
    Store that uses obstore for fast read/write from AWS, GCP, Azure.

    Parameters
    ----------
    store : obstore.store.ObjectStore
        An obstore store instance that is set up with the proper credentials.
    read_only : bool
        Whether to open the store in read-only mode.
    concurrency_limit : int, optional
        Maximum number of concurrent I/O operations. Default is 50.
        Set to None for unlimited concurrency.

    Warnings
    --------
    ObjectStore is experimental and subject to API changes without notice. Please
    raise an issue with any comments/concerns about the store.
    """

    store: T_Store
    """The underlying obstore instance."""

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ObjectStore):
            return False

        if not self.read_only == value.read_only:
            return False

        return self.store == value.store  # type: ignore[no-any-return]

    def __init__(
        self,
        store: T_Store,
        *,
        read_only: bool = False,
        concurrency_limit: int | None = 50,
    ) -> None:
        if not store.__class__.__module__.startswith("obstore"):
            raise TypeError(f"expected ObjectStore class, got {store!r}")
        Store.__init__(self, read_only=read_only)
        ConcurrencyLimiter.__init__(self, concurrency_limit)
        self.store = store

    def with_read_only(self, read_only: bool = False) -> Self:
        # docstring inherited
        return type(self)(
            store=self.store,
            read_only=read_only,
            concurrency_limit=self.concurrency_limit,
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

    @with_concurrency_limit
    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        # docstring inherited
        import obstore as obs

        try:
            return await self._get_impl(key, prototype, byte_range, obs)
        except _ALLOWED_EXCEPTIONS:
            return None

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # docstring inherited
        # We override to:
        # 1. Avoid deadlock from calling the decorated get() method
        # 2. Batch RangeByteRequests per-file using get_ranges_async for performance
        import obstore as obs

        key_ranges = list(key_ranges)
        # Group bounded range requests by path for batched fetching
        per_file_bounded: dict[str, list[tuple[int, RangeByteRequest]]] = defaultdict(list)
        other_requests: list[tuple[int, str, ByteRequest | None]] = []

        for idx, (path, byte_range) in enumerate(key_ranges):
            if isinstance(byte_range, RangeByteRequest):
                per_file_bounded[path].append((idx, byte_range))
            else:
                other_requests.append((idx, path, byte_range))

        buffers: list[Buffer | None] = [None] * len(key_ranges)

        async def _fetch_ranges(path: str, requests: list[tuple[int, RangeByteRequest]]) -> None:
            """Batch multiple range requests for the same file using get_ranges_async."""
            starts = [r.start for _, r in requests]
            ends = [r.end for _, r in requests]
            async with self._limit():
                responses = await obs.get_ranges_async(
                    self.store, path=path, starts=starts, ends=ends
                )
            for (idx, _), response in zip(requests, responses, strict=True):
                buffers[idx] = prototype.buffer.from_bytes(response)  # type: ignore[arg-type]

        async def _fetch_one(idx: int, path: str, byte_range: ByteRequest | None) -> None:
            """Fetch a single non-range request with semaphore limiting."""
            try:
                async with self._limit():
                    buffers[idx] = await self._get_impl(path, prototype, byte_range, obs)
            except _ALLOWED_EXCEPTIONS:
                pass  # buffers[idx] stays None

        futs: list[Coroutine[Any, Any, None]] = []
        for path, requests in per_file_bounded.items():
            futs.append(_fetch_ranges(path, requests))
        for idx, path, byte_range in other_requests:
            futs.append(_fetch_one(idx, path, byte_range))

        await asyncio.gather(*futs)
        return buffers

    async def _get_impl(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None, obs: Any
    ) -> Buffer:
        """Implementation of get without semaphore decoration."""
        if byte_range is None:
            resp = await obs.get_async(self.store, key)
            return prototype.buffer.from_bytes(await resp.bytes_async())
        elif isinstance(byte_range, RangeByteRequest):
            bytes = await obs.get_range_async(
                self.store, key, start=byte_range.start, end=byte_range.end
            )
            return prototype.buffer.from_bytes(bytes)
        elif isinstance(byte_range, OffsetByteRequest):
            resp = await obs.get_async(
                self.store, key, options={"range": {"offset": byte_range.offset}}
            )
            return prototype.buffer.from_bytes(await resp.bytes_async())
        elif isinstance(byte_range, SuffixByteRequest):
            try:
                resp = await obs.get_async(
                    self.store, key, options={"range": {"suffix": byte_range.suffix}}
                )
                return prototype.buffer.from_bytes(await resp.bytes_async())
            except obs.exceptions.NotSupportedError:
                head_resp = await obs.head_async(self.store, key)
                file_size = head_resp["size"]
                suffix_len = byte_range.suffix
                buffer = await obs.get_range_async(
                    self.store,
                    key,
                    start=file_size - suffix_len,
                    length=suffix_len,
                )
                return prototype.buffer.from_bytes(buffer)
        else:
            raise ValueError(f"Unexpected byte_range, got {byte_range}")

    async def exists(self, key: str) -> bool:
        # docstring inherited
        import obstore as obs

        try:
            await obs.head_async(self.store, key)
        except FileNotFoundError:
            return False
        else:
            return True

    @property
    def supports_writes(self) -> bool:
        # docstring inherited
        return True

    @with_concurrency_limit
    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        import obstore as obs

        self._check_writable()

        buf = value.as_buffer_like()
        await obs.put_async(self.store, key, buf)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        # Override to avoid deadlock from calling decorated set() method
        import obstore as obs

        self._check_writable()

        async def _set_with_limit(key: str, value: Buffer) -> None:
            buf = value.as_buffer_like()
            async with self._limit():
                await obs.put_async(self.store, key, buf)

        await asyncio.gather(*[_set_with_limit(key, value) for key, value in values])

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        # docstring inherited
        # Not decorated to avoid deadlock when called in batch via gather()
        import obstore as obs

        self._check_writable()
        buf = value.as_buffer_like()
        async with self._limit():
            with contextlib.suppress(obs.exceptions.AlreadyExistsError):
                await obs.put_async(self.store, key, buf, mode="create")

    @property
    def supports_deletes(self) -> bool:
        # docstring inherited
        return True

    @with_concurrency_limit
    async def delete(self, key: str) -> None:
        # docstring inherited
        import obstore as obs

        self._check_writable()

        # Some obstore stores such as local filesystems, GCP and Azure raise an error
        # when deleting a non-existent key, while others such as S3 and in-memory do
        # not. We suppress the error to make the behavior consistent across all obstore
        # stores. This is also in line with the behavior of the other Zarr store adapters.
        with contextlib.suppress(FileNotFoundError):
            await obs.delete_async(self.store, key)

    async def delete_dir(self, prefix: str) -> None:
        # docstring inherited
        import obstore as obs

        self._check_writable()
        if prefix != "" and not prefix.endswith("/"):
            prefix += "/"

        metas = await obs.list(self.store, prefix).collect_async()

        async def _delete_with_limit(path: str) -> None:
            async with self._limit():
                with contextlib.suppress(FileNotFoundError):
                    await obs.delete_async(self.store, path)

        await asyncio.gather(*[_delete_with_limit(m["path"]) for m in metas])

    @property
    def supports_listing(self) -> bool:
        # docstring inherited
        return True

    async def _list(self, prefix: str | None = None) -> AsyncGenerator[ObjectMeta, None]:
        import obstore as obs

        objects: ListStream[Sequence[ObjectMeta]] = obs.list(self.store, prefix=prefix)
        async for batch in objects:
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
        import obstore as obs

        coroutine = obs.list_with_delimiter_async(self.store, prefix=prefix)
        return _transform_list_dir(coroutine, prefix)

    async def getsize(self, key: str) -> int:
        # docstring inherited
        import obstore as obs

        resp = await obs.head_async(self.store, key)
        return resp["size"]

    async def getsize_prefix(self, prefix: str) -> int:
        # docstring inherited
        sizes = [obj["size"] async for obj in self._list(prefix=prefix)]
        return sum(sizes)


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
        yield _relativize_path(path=path, prefix=prefix)
