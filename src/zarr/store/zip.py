from __future__ import annotations

import os
import threading
import time
import zipfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal

from zarr.abc.store import Store
from zarr.buffer import Buffer, BufferPrototype

ZipStoreAccessModeLiteral = Literal["r", "w", "a"]


class ZipStore(Store):
    supports_writes: bool = True
    supports_deletes: bool = False
    supports_partial_writes: bool = False
    supports_listing: bool = True

    path: Path
    compression: int
    allowZip64: bool

    def __init__(
        self,
        path: Path | str,
        *,
        mode: ZipStoreAccessModeLiteral = "r",
        compression: int = zipfile.ZIP_STORED,
        allowZip64: bool = True,
    ):
        super().__init__(mode=mode)

        if isinstance(path, str):
            path = Path(path)
        assert isinstance(path, Path)
        self.path = path  # root?

        self.compression = compression
        self.allowZip64 = allowZip64

        self._lock = threading.RLock()  # TODO: evaluate if this is the lock we want or if we want an asyncio.Lock or something like that

        self._zf = zipfile.ZipFile(path, mode=mode, compression=compression, allowZip64=allowZip64)

    def close(self) -> None:
        self._is_open = False
        with self._lock:
            self._zf.close()

    async def clear(self) -> None:
        with self._lock:
            self._check_writable()
            self._zf.close()
            os.remove(self.path)
            self._zf = zipfile.ZipFile(
                self.path, mode="w", compression=self.compression, allowZip64=self.allowZip64
            )

    async def empty(self) -> bool:
        async for _ in self.list():
            return False
        return True

    def __str__(self) -> str:
        return f"zip://{self.path}"

    def __repr__(self) -> str:
        return f"ZipStore({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.path == other.path

    def _get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        try:
            with self._zf.open(key) as f:  # will raise KeyError
                if byte_range is None:
                    return prototype.buffer.from_bytes(f.read())
                start, length = byte_range
                if start:
                    if start < 0:
                        start = f.seek(start, os.SEEK_END) + start
                    else:
                        start = f.seek(start, os.SEEK_SET)
                if length:
                    return prototype.buffer.from_bytes(f.read(length))
                else:
                    return prototype.buffer.from_bytes(f.read())
        except KeyError:
            return None

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        assert isinstance(key, str)

        with self._lock:
            return self._get(key, prototype=prototype, byte_range=byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: list[tuple[str, tuple[int | None, int | None]]],
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
        out = []
        with self._lock:
            for key, byte_range in key_ranges:
                out.append(self._get(key, prototype=prototype, byte_range=byte_range))
        return out

    def _set(self, key: str, value: Buffer) -> None:
        # generally, this should be called inside a lock
        keyinfo = zipfile.ZipInfo(filename=key, date_time=time.localtime(time.time())[:6])
        keyinfo.compress_type = self.compression
        if keyinfo.filename[-1] == os.sep:
            keyinfo.external_attr = 0o40775 << 16  # drwxrwxr-x
            keyinfo.external_attr |= 0x10  # MS-DOS directory flag
        else:
            keyinfo.external_attr = 0o644 << 16  # ?rw-r--r--
        self._zf.writestr(keyinfo, value.to_bytes())

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        assert isinstance(key, str)
        if not isinstance(value, Buffer):
            raise TypeError("ZipStore.set(): `value` must a Buffer instance")
        with self._lock:
            self._set(key, value)

    async def set_partial_values(self, key_start_values: list[tuple[str, int, bytes]]) -> None:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        with self._lock:
            try:
                self._zf.getinfo(key)
            except KeyError:
                return False
            else:
                return True

    async def list(self) -> AsyncGenerator[str, None]:
        """Retrieve all keys in the store.

        Returns
        -------
        AsyncGenerator[str, None]
        """
        with self._lock:
            for key in self._zf.namelist():
                yield key

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        """Retrieve all keys in the store with a given prefix.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncGenerator[str, None]
        """

        async for key in self.list():
            if key.startswith(prefix):
                yield key

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

        if prefix.endswith("/"):
            prefix = prefix[:-1]

        keys = self._zf.namelist()
        seen = set()
        if prefix == "":
            keys_unique = set(k.split("/")[0] for k in keys)
            for key in keys_unique:
                if key not in seen:
                    seen.add(key)
                    yield key
        else:
            for key in keys:
                if key.startswith(prefix + "/") and key != prefix:
                    k = key.removeprefix(prefix + "/").split("/")[0]
                    if k not in seen:
                        seen.add(k)
                        yield k
