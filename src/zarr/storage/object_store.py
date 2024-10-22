from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import object_store_rs as obs

from zarr.abc.store import ByteRangeRequest, Store
from zarr.core.buffer import Buffer
from zarr.core.buffer.core import BufferPrototype

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Iterable
    from typing import Any

    from object_store_rs.store import ObjectStore as _ObjectStore

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
            return await resp.bytes_async()

            pass

        raise NotImplementedError

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

        futs: list[Coroutine[Any, Any, list[bytes]]] = []
        for path, ranges in per_file_requests.items():
            offsets = [r[0] for r in ranges]
            lengths = [r[1] - r[0] for r in ranges]
            fut = obs.get_ranges_async(self.store, path, offsets=offsets, lengths=lengths)
            futs.append(fut)

        result = await asyncio.gather(*futs)

        output_buffers: list[bytes] = [b""] * len(key_ranges)
        for per_file_request, buffers in zip(per_file_requests.items(), result, strict=True):
            path, ranges = per_file_request
            for buffer, ranges_ in zip(buffers, ranges, strict=True):
                initial_index = ranges_[2]
                output_buffers[initial_index] = buffer

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

    # TODO:
    # async def set_if_not_exists(self, key: str, value: Buffer) -> None:

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
        # object-store-rs does not yet support list results as an async generator
        # https://github.com/apache/arrow-rs/issues/6587
        objects = obs.list(self.store)
        paths = [object["path"] for object in objects]
        # Not sure how to convert list to async generator
        return paths

    def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        # object-store-rs does not yet support list results as an async generator
        # https://github.com/apache/arrow-rs/issues/6587
        objects = obs.list(self.store, prefix=prefix)
        paths = [object["path"] for object in objects]
        # Not sure how to convert list to async generator
        return paths

    def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        # object-store-rs does not yet support list results as an async generator
        # https://github.com/apache/arrow-rs/issues/6587
        objects = obs.list_with_delimiter(self.store, prefix=prefix)
        paths = [object["path"] for object in objects["objects"]]
        # Not sure how to convert list to async generator
        return paths
