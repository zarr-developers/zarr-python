from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypedDict

import obstore as obs

from zarr.abc.store import ByteRangeRequest, Store
from zarr.core.buffer import Buffer
from zarr.core.buffer.core import BufferPrototype

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Iterable
    from typing import Any

    from obstore import ListStream, ObjectMeta, OffsetRange, SuffixRange
    from obstore.store import ObjectStore as _ObjectStore

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import BytesLike


class ObjectStore(Store):
    store: _ObjectStore

    def __init__(self, store: _ObjectStore) -> None:
        self.store = store

    def __str__(self) -> str:
        return f"object://{self.store}"

    def __repr__(self) -> str:
        return f"ObjectStore({self!r})"

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRangeRequest | None = None
    ) -> Buffer:
        if byte_range is None:
            resp = await obs.get_async(self.store, key)
            return prototype.buffer.from_buffer(memoryview(await resp.bytes_async()))  # type: ignore not assignable to buffer

        start, end = byte_range
        if start is not None and end is not None:
            resp = await obs.get_range_async(self.store, key, start=start, end=end)
            return prototype.buffer.from_buffer(memoryview(await resp.bytes_async()))  # type: ignore not assignable to buffer
        elif start is not None:
            if start >= 0:
                # Offset request
                resp = await obs.get_async(self.store, key, options={"range": {"offset": start}})
            else:
                resp = await obs.get_async(self.store, key, options={"range": {"suffix": start}})

            return prototype.buffer.from_buffer(memoryview(await resp.bytes_async()))  # type: ignore not assignable to buffer
        else:
            raise ValueError(f"Unexpected input to `get`: {start=}, {end=}")

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRangeRequest]],
    ) -> list[Buffer | None]:
        return await _get_partial_values(self.store, prototype=prototype, key_ranges=key_ranges)

    async def exists(self, key: str) -> bool:
        try:
            await obs.head_async(self.store, key)
        except FileNotFoundError:
            return False
        else:
            return True

    @property
    def supports_writes(self) -> bool:
        return True

    async def set(self, key: str, value: Buffer) -> None:
        buf = value.to_bytes()
        await obs.put_async(self.store, key, buf)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        buf = value.to_bytes()
        await obs.put_async(self.store, key, buf, mode="create")

    @property
    def supports_deletes(self) -> bool:
        return True

    async def delete(self, key: str) -> None:
        await obs.delete_async(self.store, key)

    @property
    def supports_partial_writes(self) -> bool:
        return False

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, BytesLike]]
    ) -> None:
        raise NotImplementedError

    @property
    def supports_listing(self) -> bool:
        return True

    def list(self) -> AsyncGenerator[str, None]:
        objects: ListStream[list[ObjectMeta]] = obs.list(self.store)
        return _transform_list(objects)

    def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        objects: ListStream[list[ObjectMeta]] = obs.list(self.store, prefix=prefix)
        return _transform_list(objects)

    def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        objects: ListStream[list[ObjectMeta]] = obs.list(self.store, prefix=prefix)
        return _transform_list_dir(objects, prefix)


async def _transform_list(
    list_stream: AsyncGenerator[list[ObjectMeta], None],
) -> AsyncGenerator[str, None]:
    async for batch in list_stream:
        for item in batch:
            yield item["path"]


async def _transform_list_dir(
    list_stream: AsyncGenerator[list[ObjectMeta], None], prefix: str
) -> AsyncGenerator[str, None]:
    # We assume that the underlying object-store implementation correctly handles the
    # prefix, so we don't double-check that the returned results actually start with the
    # given prefix.
    prefix_len = len(prefix)
    async for batch in list_stream:
        for item in batch:
            # Yield this item if "/" does not exist after the prefix.
            if "/" not in item["path"][prefix_len:]:
                yield item["path"]


class BoundedRequest(TypedDict):
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


class OtherRequest(TypedDict):
    """Offset or suffix range requests.

    These requests cannot be concurrent on the Rust side, and each need their own call
    to `obstore.get_async`, passing in the `range` parameter.
    """

    original_request_index: int
    """The positional index in the original key_ranges input"""

    path: str
    """The path to request from."""

    range: OffsetRange | SuffixRange
    """The range request type."""


class Response(TypedDict):
    """A response buffer associated with the original index that it should be restored to."""

    original_request_index: int
    """The positional index in the original key_ranges input"""

    buffer: Buffer
    """The buffer returned from obstore's range request."""


async def _make_bounded_requests(
    store: obs.store.ObjectStore,
    path: str,
    requests: list[BoundedRequest],
    prototype: BufferPrototype,
) -> list[Response]:
    """Make all bounded requests for a specific file.

    `obstore.get_ranges_async` allows for making concurrent requests for multiple ranges
    within a single file, and will e.g. merge concurrent requests. This only uses one
    single Python coroutine.
    """

    starts = [r["start"] for r in requests]
    ends = [r["end"] for r in requests]
    responses = await obs.get_ranges_async(store, path=path, starts=starts, ends=ends)

    buffer_responses: list[Response] = []
    for request, response in zip(requests, responses, strict=True):
        buffer_responses.append(
            {
                "original_request_index": request["original_request_index"],
                "buffer": prototype.buffer.from_bytes(memoryview(response)),
            }
        )

    return buffer_responses


async def _make_other_request(
    store: obs.store.ObjectStore,
    request: OtherRequest,
    prototype: BufferPrototype,
) -> list[Response]:
    """Make suffix or offset requests.

    We return a `list[Response]` for symmetry with `_make_bounded_requests` so that all
    futures can be gathered together.
    """
    resp = await obs.get_async(store, request["path"], options={"range": request["range"]})
    buffer = await resp.bytes_async()
    return [
        {
            "original_request_index": request["original_request_index"],
            "buffer": prototype.buffer.from_bytes(buffer),
        }
    ]


async def _get_partial_values(
    store: obs.store.ObjectStore,
    prototype: BufferPrototype,
    key_ranges: Iterable[tuple[str, ByteRangeRequest]],
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
    per_file_bounded_requests: dict[str, list[BoundedRequest]] = defaultdict(list)
    other_requests: list[OtherRequest] = []

    for idx, (path, (start, end)) in enumerate(key_ranges):
        if start is None:
            raise ValueError("Cannot pass `None` for the start of the range request.")

        if end is not None:
            # This is a bounded request with known start and end byte.
            per_file_bounded_requests[path].append(
                {"original_request_index": idx, "start": start, "end": end}
            )
        elif end is None and start < 0:
            # Suffix request from the end
            other_requests.append(
                {"original_request_index": idx, "path": path, "range": {"suffix": abs(start)}}
            )
        elif end is None and start > 0:
            # Offset request to the end
            other_requests.append(
                {"original_request_index": idx, "path": path, "range": {"offset": start}}
            )
        else:
            raise ValueError(f"Unsupported range input: {start=}, {end=}")

    futs: list[Coroutine[Any, Any, list[Response]]] = []
    for path, bounded_ranges in per_file_bounded_requests.items():
        futs.append(_make_bounded_requests(store, path, bounded_ranges, prototype))

    for request in other_requests:
        futs.append(_make_other_request(store, request, prototype))  # noqa: PERF401

    buffers: list[Buffer | None] = [None] * len(key_ranges)

    # TODO: this gather a list of list of Response; not sure if there's a way to
    # unpack these lists inside of an `asyncio.gather`?
    for responses in await asyncio.gather(*futs):
        for resp in responses:
            buffers[resp["original_request_index"]] = resp["buffer"]

    return buffers
