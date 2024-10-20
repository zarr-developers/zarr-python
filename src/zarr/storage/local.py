from __future__ import annotations

import asyncio
import io
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from zarr.abc.store import Store
from zarr.core.buffer import Buffer
from zarr.core.common import concurrent_map

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable
    from typing import Self

    from zarr.abc.store import ByteRangeRequest
    from zarr.core.buffer import BufferPrototype
    from zarr.core.common import AccessModeLiteral


def _get(path: Path, prototype: BufferPrototype, byte_range: ByteRangeRequest | None) -> Buffer:
    """
    Fetch a contiguous region of bytes from a file.

    Parameters
    ----------

    path : Path
        The file to read bytes from.
    prototype : BufferPrototype
        The buffer prototype to use when reading the bytes.
    byte_range : tuple[int | None, int | None] | None = None
        The range of bytes to read. If `byte_range` is `None`, then the entire file will be read.
        If `byte_range` is a tuple, the first value specifies the index of the first byte to read,
        and the second value specifies the total number of bytes to read. If the total value is
        `None`, then the entire file after the first byte will be read.
    """
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
    path : str or Path
        Directory to use as root of store.
    mode : str
        Mode in which to open the store. Either 'r', 'r+', 'a', 'w', 'w-'.

    Attributes
    ----------
    supports_writes
    supports_deletes
    supports_partial_writes
    supports_listing
    path
    """

    supports_writes: bool = True
    supports_deletes: bool = True
    supports_partial_writes: bool = True
    supports_listing: bool = True
    _path: Path

    def __init__(self, path: Path | str, *, mode: AccessModeLiteral = "r") -> None:
        super().__init__(mode=mode, path=str(path))
        self._path = Path(self.path)

    async def _open(self) -> None:
        if not self.mode.readonly:
            self._path.mkdir(parents=True, exist_ok=True)
        return await super()._open()

    async def clear(self) -> None:
        # docstring inherited
        self._check_writable()
        shutil.rmtree(self.path)
        os.mkdir(self.path)

    async def empty(self) -> bool:
        # docstring inherited
        try:
            with os.scandir(self.path) as it:
                for entry in it:
                    if entry.is_file():
                        # stop once a file is found
                        return False
        except FileNotFoundError:
            return True
        else:
            return True

    def with_mode(self, mode: AccessModeLiteral) -> Self:
        # docstring inherited
        return type(self)(path=self.path, mode=mode)

    def __str__(self) -> str:
        return f"file://{self.path}"

    def __repr__(self) -> str:
        return f"LocalStore({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.path == other.path

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        # docstring inherited
        if not self._is_open:
            await self._open()

        path = self._path / key

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
            path = self._path / key
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
        path = self._path / key
        await asyncio.to_thread(_put, path, value, start=None, exclusive=exclusive)

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, bytes | bytearray | memoryview]]
    ) -> None:
        # docstring inherited
        self._check_writable()
        args = []
        for key, start, value in key_start_values:
            assert isinstance(key, str)
            path = os.path.join(self.path, key)
            args.append((_put, path, value, start))
        await concurrent_map(args, asyncio.to_thread, limit=None)  # TODO: fix limit

    async def delete(self, key: str) -> None:
        # docstring inherited
        self._check_writable()
        path = self._path / key
        if path.is_dir():  # TODO: support deleting directories? shutil.rmtree?
            shutil.rmtree(path)
        else:
            await asyncio.to_thread(path.unlink, True)  # Q: we may want to raise if path is missing

    async def exists(self, key: str) -> bool:
        # docstring inherited
        path = self._path / key
        return await asyncio.to_thread(path.is_file)

    async def list(self) -> AsyncGenerator[str, None]:
        """Retrieve all keys in the store.

        Returns
        -------
        AsyncGenerator[str, None]
        """
        # TODO: just invoke list_prefix with the prefix "/"
        to_strip = self.path + "/"
        for p in self._path.rglob("*"):
            if p.is_file():
                yield str(p.relative_to(to_strip))

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        to_strip = os.path.join(self.path, prefix)
        for p in (self._path / prefix).rglob("*"):
            if p.is_file():
                yield str(p.relative_to(to_strip))

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        base = os.path.join(self.path, prefix)
        to_strip = base + "/"

        try:
            key_iter = Path(base).iterdir()
            for key in key_iter:
                yield str(key.relative_to(to_strip))
        except (FileNotFoundError, NotADirectoryError):
            pass
