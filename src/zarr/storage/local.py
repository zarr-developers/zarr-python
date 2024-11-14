from __future__ import annotations

import asyncio
import io
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from zarr.abc.store import ByteRangeRequest, Store
from zarr.core.buffer import Buffer
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.common import concurrent_map

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from zarr.core.buffer import BufferPrototype


def _get(
    path: Path, prototype: BufferPrototype, byte_range: tuple[int | None, int | None] | None
) -> Buffer:
    if byte_range is not None:
        if byte_range[0] is None:
            start = 0
        else:
            start = byte_range[0]

        end = (start + byte_range[1]) if byte_range[1] is not None else None
    else:
        return prototype.buffer.from_bytes(path.read_bytes())
    with path.open("rb") as f:
        size = f.seek(0, io.SEEK_END)
        if start is not None:
            if start >= 0:
                f.seek(start)
            else:
                f.seek(max(0, size + start))
        if end is not None:
            if end < 0:
                end = size + end
            return prototype.buffer.from_bytes(f.read(end - f.tell()))
        return prototype.buffer.from_bytes(f.read())


def _put(
    path: Path,
    value: Buffer,
    start: int | None = None,
    exclusive: bool = False,
) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if start is not None:
        with path.open("r+b") as f:
            f.seek(start)
            f.write(value.as_numpy_array().tobytes())
        return None
    else:
        view = memoryview(value.as_numpy_array().tobytes())
        if exclusive:
            mode = "xb"
        else:
            mode = "wb"
        with path.open(mode=mode) as f:
            return f.write(view)


class LocalStore(Store):
    """
    Local file system store.

    Parameters
    ----------
    root : str or Path
        Directory to use as root of store.
    read_only : bool
        Whether the store is read-only

    Attributes
    ----------
    supports_writes
    supports_deletes
    supports_partial_writes
    supports_listing
    root
    """

    supports_writes: bool = True
    supports_deletes: bool = True
    supports_partial_writes: bool = True
    supports_listing: bool = True

    root: Path

    def __init__(self, root: Path | str, *, read_only: bool = False) -> None:
        super().__init__(read_only=read_only)
        if isinstance(root, str):
            root = Path(root)
        if not isinstance(root, Path):
            raise TypeError(
                f'"root" must be a string or Path instance. Got an object with type {type(root)} instead.'
            )
        self.root = root

    async def _open(self) -> None:
        if not self.read_only:
            self.root.mkdir(parents=True, exist_ok=True)
        return await super()._open()

    async def clear(self) -> None:
        # docstring inherited
        self._check_writable()
        shutil.rmtree(self.root)
        self.root.mkdir()

    def __str__(self) -> str:
        return f"file://{self.root.as_posix()}"

    def __repr__(self) -> str:
        return f"LocalStore('{self}')"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.root == other.root

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        # docstring inherited
        if prototype is None:
            prototype = default_buffer_prototype()
        if not self._is_open:
            await self._open()
        assert isinstance(key, str)
        path = self.root / key

        try:
            return await asyncio.to_thread(_get, path, prototype, byte_range)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRangeRequest]],
    ) -> list[Buffer | None]:
        # docstring inherited
        args = []
        for key, byte_range in key_ranges:
            assert isinstance(key, str)
            path = self.root / key
            args.append((_get, path, prototype, byte_range))
        return await concurrent_map(args, asyncio.to_thread, limit=None)  # TODO: fix limit

    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        return await self._set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        # docstring inherited
        try:
            return await self._set(key, value, exclusive=True)
        except FileExistsError:
            pass

    async def _set(self, key: str, value: Buffer, exclusive: bool = False) -> None:
        if not self._is_open:
            await self._open()
        self._check_writable()
        assert isinstance(key, str)
        if not isinstance(value, Buffer):
            raise TypeError("LocalStore.set(): `value` must a Buffer instance")
        path = self.root / key
        await asyncio.to_thread(_put, path, value, start=None, exclusive=exclusive)

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, bytes | bytearray | memoryview]]
    ) -> None:
        # docstring inherited
        self._check_writable()
        args = []
        for key, start, value in key_start_values:
            assert isinstance(key, str)
            path = self.root / key
            args.append((_put, path, value, start))
        await concurrent_map(args, asyncio.to_thread, limit=None)  # TODO: fix limit

    async def delete(self, key: str) -> None:
        # docstring inherited
        self._check_writable()
        path = self.root / key
        if path.is_dir():  # TODO: support deleting directories? shutil.rmtree?
            shutil.rmtree(path)
        else:
            await asyncio.to_thread(path.unlink, True)  # Q: we may want to raise if path is missing

    async def exists(self, key: str) -> bool:
        # docstring inherited
        path = self.root / key
        return await asyncio.to_thread(path.is_file)

    async def list(self) -> AsyncIterator[str]:
        # docstring inherited
        to_strip = self.root.as_posix() + "/"
        for p in list(self.root.rglob("*")):
            if p.is_file():
                yield p.as_posix().replace(to_strip, "")

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        to_strip = self.root.as_posix() + "/"
        prefix = prefix.rstrip("/")
        for p in (self.root / prefix).rglob("*"):
            if p.is_file():
                yield p.as_posix().replace(to_strip, "")

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        base = self.root / prefix
        try:
            key_iter = base.iterdir()
            for key in key_iter:
                yield key.relative_to(base).as_posix()
        except (FileNotFoundError, NotADirectoryError):
            pass

    async def getsize(self, key: str) -> int:
        return os.path.getsize(self.root / key)
