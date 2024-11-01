from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import obstore as obs

from zarr.abc.store import ByteRangeRequest, Store
from zarr.core.buffer import Buffer
from zarr.core.buffer.core import BufferPrototype

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Iterable
    from typing import Any

    from obstore import Buffer as ObjectStoreBuffer
    from obstore import ListStream, ObjectMeta
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
        # TODO: this is a bit hacky and untested. ObjectStore has a `get_ranges` method
        # that will additionally merge nearby ranges, but it's _per_ file. So we need to
        # split these key_ranges into **per-file** key ranges, and then reassemble the
        # results in the original order.
        key_ranges = list(key_ranges)

        per_file_requests: dict[str, list[tuple[int | None, int | None, int]]] = defaultdict(list)
        for idx, (path, range_) in enumerate(key_ranges):
            per_file_requests[path].append((range_[0], range_[1], idx))

        futs: list[Coroutine[Any, Any, list[ObjectStoreBuffer]]] = []
        for path, ranges in per_file_requests.items():
            starts = [r[0] for r in ranges]
            ends = [r[1] for r in ranges]
            fut = obs.get_ranges_async(self.store, path, starts=starts, ends=ends)
            futs.append(fut)

        result = await asyncio.gather(*futs)

        output_buffers: list[type[BufferPrototype]] = [b""] * len(key_ranges)
        for per_file_request, buffers in zip(per_file_requests.items(), result, strict=True):
            path, ranges = per_file_request
            for buffer, ranges_ in zip(buffers, ranges, strict=True):
                initial_index = ranges_[2]
                output_buffers[initial_index] = prototype.buffer.from_buffer(memoryview(buffer))

        return output_buffers

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
