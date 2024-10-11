from __future__ import annotations

import asyncio
import io
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Self

from zarr.abc.store import ByteRangeRequest, Store
from zarr.core.buffer import Buffer
from zarr.core.common import concurrent_map

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable

    from zarr.core.buffer import BufferPrototype
    from zarr.core.common import AccessModeLiteral


def _get(
    path: Path, prototype: BufferPrototype, byte_range: tuple[int | None, int | None] | None
) -> Buffer:
    """
    Fetch a contiguous region of bytes from a file.

    Parameters
    ----------
    path: Path
        The file to read bytes from.
    byte_range: tuple[int, int | None] | None = None
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
    supports_writes: bool = True
    supports_deletes: bool = True
    supports_partial_writes: bool = True
    supports_listing: bool = True

    root: Path

    def __init__(self, root: Path | str, *, mode: AccessModeLiteral = "r") -> None:
        super().__init__(mode=mode)
        if isinstance(root, str):
            root = Path(root)
        assert isinstance(root, Path)
        self.root = root

    async def _open(self) -> None:
        if not self.mode.readonly:
            self.root.mkdir(parents=True, exist_ok=True)
        return await super()._open()

    async def clear(self) -> None:
        self._check_writable()
        shutil.rmtree(self.root)
        self.root.mkdir()

    async def empty(self) -> bool:
        try:
            with os.scandir(self.root) as it:
                for entry in it:
                    if entry.is_file():
                        # stop once a file is found
                        return False
        except FileNotFoundError:
            return True
        else:
            return True

    def with_mode(self, mode: AccessModeLiteral) -> Self:
        return type(self)(root=self.root, mode=mode)

    def __str__(self) -> str:
        return f"file://{self.root}"

    def __repr__(self) -> str:
        return f"LocalStore({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.root == other.root

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
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
        """
        Read byte ranges from multiple keys.

        Parameters
        ----------
        key_ranges: List[Tuple[str, Tuple[int, int]]]
            A list of (key, (start, length)) tuples. The first element of the tuple is the name of
            the key in storage to fetch bytes from. The second element the tuple defines the byte
            range to retrieve. These values are arguments to `get`, as this method wraps
            concurrent invocation of `get`.
        """
        args = []
        for key, byte_range in key_ranges:
            assert isinstance(key, str)
            path = self.root / key
            args.append((_get, path, prototype, byte_range))
        return await concurrent_map(args, asyncio.to_thread, limit=None)  # TODO: fix limit

    async def set(self, key: str, value: Buffer) -> None:
        return await self._set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
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
        self._check_writable()
        args = []
        for key, start, value in key_start_values:
            assert isinstance(key, str)
            path = self.root / key
            args.append((_put, path, value, start))
        await concurrent_map(args, asyncio.to_thread, limit=None)  # TODO: fix limit

    async def delete(self, key: str) -> None:
        self._check_writable()
        path = self.root / key
        if path.is_dir():  # TODO: support deleting directories? shutil.rmtree?
            shutil.rmtree(path)
        else:
            await asyncio.to_thread(path.unlink, True)  # Q: we may want to raise if path is missing

    async def exists(self, key: str) -> bool:
        path = self.root / key
        return await asyncio.to_thread(path.is_file)

    async def list(self) -> AsyncGenerator[str, None]:
        """Retrieve all keys in the store.

        Returns
        -------
        AsyncGenerator[str, None]
        """
        to_strip = str(self.root) + "/"
        for p in list(self.root.rglob("*")):
            if p.is_file():
                yield str(p).replace(to_strip, "")

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
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
        to_strip = os.path.join(str(self.root / prefix))
        for p in (self.root / prefix).rglob("*"):
            if p.is_file():
                yield str(p.relative_to(to_strip))

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        """
        Retrieve all keys and prefixes with a given prefix and which do not contain the character
        “/” after the given prefix.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncGenerator[str, None]
        """

        base = self.root / prefix
        to_strip = str(base) + "/"

        try:
            key_iter = base.iterdir()
            for key in key_iter:
                yield str(key).replace(to_strip, "")
        except (FileNotFoundError, NotADirectoryError):
            pass
