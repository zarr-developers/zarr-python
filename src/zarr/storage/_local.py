from __future__ import annotations

import asyncio
import io
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.common import concurrent_map

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from zarr.core.buffer import BufferPrototype


def _get(path: Path, prototype: BufferPrototype, byte_range: ByteRequest | None) -> Buffer:
    if byte_range is None:
        return prototype.buffer.from_bytes(path.read_bytes())
    with path.open("rb") as f:
        size = f.seek(0, io.SEEK_END)
        if isinstance(byte_range, RangeByteRequest):
            f.seek(byte_range.start)
            return prototype.buffer.from_bytes(f.read(byte_range.end - f.tell()))
        elif isinstance(byte_range, OffsetByteRequest):
            f.seek(byte_range.offset)
        elif isinstance(byte_range, SuffixByteRequest):
            f.seek(max(0, size - byte_range.suffix))
        else:
            raise TypeError(f"Unexpected byte_range, got {byte_range}.")
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
            # write takes any object supporting the buffer protocol
            f.write(value.as_buffer_like())
        return None
    else:
        view = value.as_buffer_like()
        if exclusive:
            mode = "xb"
        else:
            mode = "wb"
        with path.open(mode=mode) as f:
            # write takes any object supporting the buffer protocol
            return f.write(view)


class LocalStore(Store):
    """
    Store for the local file system.

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
                f"'root' must be a string or Path instance. Got an instance of {type(root)} instead."
            )
        self.root = root

    def with_read_only(self, read_only: bool = False) -> LocalStore:
        # docstring inherited
        return type(self)(
            root=self.root,
            read_only=read_only,
        )

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
        byte_range: ByteRequest | None = None,
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
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
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
            raise TypeError(
                f"LocalStore.set(): `value` must be a Buffer instance. Got an instance of {type(value)} instead."
            )
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
        """
        Remove a key from the store.

        Parameters
        ----------
        key : str

        Notes
        -----
        If ``key`` is a directory within this store, the entire directory
        at ``store.root / key`` is deleted.
        """
        # docstring inherited
        self._check_writable()
        path = self.root / key
        if path.is_dir():  # TODO: support deleting directories? shutil.rmtree?
            shutil.rmtree(path)
        else:
            await asyncio.to_thread(path.unlink, True)  # Q: we may want to raise if path is missing

    async def delete_dir(self, prefix: str) -> None:
        # docstring inherited
        self._check_writable()
        path = self.root / prefix
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            raise ValueError(f"delete_dir was passed a {prefix=!r} that is a file")
        else:
            # Non-existent directory
            # This path is tested by test_group:test_create_creates_parents for one
            pass

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

    async def move(self, dest_root: Path | str) -> None:
        """
        Move the store to another path. The old root directory is deleted.
        """
        if isinstance(dest_root, str):
            dest_root = Path(dest_root)
        os.makedirs(dest_root.parent, exist_ok=True)
        if os.path.exists(dest_root):
            raise FileExistsError(f"Destination root {dest_root} already exists.")
        shutil.move(self.root, dest_root)
        self.root = dest_root

    async def getsize(self, key: str) -> int:
        return os.path.getsize(self.root / key)
