from __future__ import annotations

from typing import TYPE_CHECKING, Self

from zarr.abc.store import ByteRangeRequest, Store
from zarr.core.buffer import Buffer, gpu
from zarr.core.common import concurrent_map
from zarr.storage._utils import _normalize_interval_index

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable, MutableMapping

    from zarr.core.buffer import BufferPrototype
    from zarr.core.common import AccessModeLiteral


# TODO: this store could easily be extended to wrap any MutableMapping store from v2
# When that is done, the `MemoryStore` will just be a store that wraps a dict.
class MemoryStore(Store):
    supports_writes: bool = True
    supports_deletes: bool = True
    supports_partial_writes: bool = True
    supports_listing: bool = True

    _store_dict: MutableMapping[str, Buffer]

    def __init__(
        self,
        store_dict: MutableMapping[str, Buffer] | None = None,
        *,
        mode: AccessModeLiteral = "r",
    ) -> None:
        super().__init__(mode=mode)
        if store_dict is None:
            store_dict = {}
        self._store_dict = store_dict

    async def empty(self) -> bool:
        return not self._store_dict

    async def clear(self) -> None:
        self._store_dict.clear()

    def with_mode(self, mode: AccessModeLiteral) -> Self:
        return type(self)(store_dict=self._store_dict, mode=mode)

    def __str__(self) -> str:
        return f"memory://{id(self._store_dict)}"

    def __repr__(self) -> str:
        return f"MemoryStore({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._store_dict == other._store_dict
            and self.mode == other.mode
        )

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        if not self._is_open:
            await self._open()
        assert isinstance(key, str)
        try:
            value = self._store_dict[key]
            start, length = _normalize_interval_index(value, byte_range)
            return prototype.buffer.from_buffer(value[start : start + length])
        except KeyError:
            return None

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRangeRequest]],
    ) -> list[Buffer | None]:
        # All the key-ranges arguments goes with the same prototype
        async def _get(key: str, byte_range: ByteRangeRequest) -> Buffer | None:
            return await self.get(key, prototype=prototype, byte_range=byte_range)

        return await concurrent_map(key_ranges, _get, limit=None)

    async def exists(self, key: str) -> bool:
        return key in self._store_dict

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        self._check_writable()
        await self._ensure_open()
        assert isinstance(key, str)
        if not isinstance(value, Buffer):
            raise TypeError(f"Expected Buffer. Got {type(value)}.")

        if byte_range is not None:
            buf = self._store_dict[key]
            buf[byte_range[0] : byte_range[1]] = value
            self._store_dict[key] = buf
        else:
            self._store_dict[key] = value

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._check_writable()
        await self._ensure_open()
        self._store_dict.setdefault(key, value)

    async def delete(self, key: str) -> None:
        self._check_writable()
        try:
            del self._store_dict[key]
        except KeyError:
            pass  # Q(JH): why not raise?

    async def set_partial_values(self, key_start_values: Iterable[tuple[str, int, bytes]]) -> None:
        raise NotImplementedError

    async def list(self) -> AsyncGenerator[str, None]:
        for key in self._store_dict:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        for key in self._store_dict:
            if key.startswith(prefix):
                yield key.removeprefix(prefix)

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        """
        Retrieve all keys in the store that begin with a given prefix. Keys are returned with the
        common leading prefix removed.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncGenerator[str, None]
        """
        if prefix.endswith("/"):
            prefix = prefix[:-1]

        if prefix == "":
            keys_unique = {k.split("/")[0] for k in self._store_dict}
        else:
            # Our dictionary doesn't contain directory markers, but we want to include
            # a pseudo directory when there's a nested item and we're listing an
            # intermediate level.
            keys_unique = {
                key.removeprefix(prefix + "/").split("/")[0]
                for key in self._store_dict
                if key.startswith(prefix + "/") and key != prefix
            }

        for key in keys_unique:
            yield key


class GpuMemoryStore(MemoryStore):
    """A GPU only memory store that stores every chunk in GPU memory irrespective
    of the original location.

    The dictionary of buffers to initialize this memory store with *must* be
    GPU Buffers.

    Writing data to this store through ``.set`` will move the buffer to the GPU
    if necessary.

    Parameters
    ----------
    store_dict: MutableMapping, optional
        A mutable mapping with string keys and :class:`zarr.core.buffer.gpu.Buffer`
        values.
    """

    _store_dict: MutableMapping[str, gpu.Buffer]  # type: ignore[assignment]

    def __init__(
        self,
        store_dict: MutableMapping[str, gpu.Buffer] | None = None,
        *,
        mode: AccessModeLiteral = "r",
    ) -> None:
        super().__init__(store_dict=store_dict, mode=mode)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return f"gpumemory://{id(self._store_dict)}"

    def __repr__(self) -> str:
        return f"GpuMemoryStore({str(self)!r})"

    @classmethod
    def from_dict(cls, store_dict: MutableMapping[str, Buffer]) -> Self:
        """
        Create a GpuMemoryStore from a dictionary of buffers at any location.

        The dictionary backing the newly created ``GpuMemoryStore`` will not be
        the same as ``store_dict``.

        Parameters
        ----------
        store_dict: mapping
            A mapping of strings keys to arbitrary Buffers. The buffer data
            will be moved into a :class:`gpu.Buffer`.

        Returns
        -------
        GpuMemoryStore
        """
        gpu_store_dict = {k: gpu.Buffer.from_buffer(v) for k, v in store_dict.items()}
        return cls(gpu_store_dict)

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        self._check_writable()
        assert isinstance(key, str)
        if not isinstance(value, Buffer):
            raise TypeError(f"Expected Buffer. Got {type(value)}.")

        # Convert to gpu.Buffer
        gpu_value = value if isinstance(value, gpu.Buffer) else gpu.Buffer.from_buffer(value)
        await super().set(key, gpu_value, byte_range=byte_range)
