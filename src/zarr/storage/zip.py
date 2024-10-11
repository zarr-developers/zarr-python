from __future__ import annotations

import os
import threading
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from zarr.abc.store import ByteRangeRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable

ZipStoreAccessModeLiteral = Literal["r", "w", "a"]


class ZipStore(Store):
    """
    Storage class using a ZIP file.

    Parameters
    ----------
    path : str
        Location of file.
    mode : str, optional
        One of 'r' to read an existing file, 'w' to truncate and write a new
        file, 'a' to append to an existing file, or 'x' to exclusively create
        and write a new file.
    compression : int, optional
        Compression method to use when writing to the archive.
    allowZip64 : bool, optional
        If True (the default) will create ZIP files that use the ZIP64
        extensions when the zipfile is larger than 2 GiB. If False
        will raise an exception when the ZIP file would require ZIP64
        extensions.
    """

    supports_writes: bool = True
    supports_deletes: bool = False
    supports_partial_writes: bool = False
    supports_listing: bool = True

    path: Path
    compression: int
    allowZip64: bool

    _zf: zipfile.ZipFile
    _lock: threading.RLock

    def __init__(
        self,
        path: Path | str,
        *,
        mode: ZipStoreAccessModeLiteral = "r",
        compression: int = zipfile.ZIP_STORED,
        allowZip64: bool = True,
    ) -> None:
        super().__init__(mode=mode)

        if isinstance(path, str):
            path = Path(path)
        assert isinstance(path, Path)
        self.path = path  # root?

        self._zmode = mode
        self.compression = compression
        self.allowZip64 = allowZip64

    def _sync_open(self) -> None:
        if self._is_open:
            raise ValueError("store is already open")

        self._lock = threading.RLock()

        self._zf = zipfile.ZipFile(
            self.path,
            mode=self._zmode,
            compression=self.compression,
            allowZip64=self.allowZip64,
        )

        self._is_open = True

    async def _open(self) -> None:
        self._sync_open()

    def __getstate__(self) -> tuple[Path, ZipStoreAccessModeLiteral, int, bool]:
        return self.path, self._zmode, self.compression, self.allowZip64

    def __setstate__(self, state: Any) -> None:
        self.path, self._zmode, self.compression, self.allowZip64 = state
        self._is_open = False
        self._sync_open()

    def close(self) -> None:
        super().close()
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
        with self._lock:
            return not self._zf.namelist()

    def with_mode(self, mode: ZipStoreAccessModeLiteral) -> Self:  # type: ignore[override]
        raise NotImplementedError("ZipStore cannot be reopened with a new mode.")

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
        byte_range: ByteRangeRequest | None = None,
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
        byte_range: ByteRangeRequest | None = None,
    ) -> Buffer | None:
        assert isinstance(key, str)

        with self._lock:
            return self._get(key, prototype=prototype, byte_range=byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRangeRequest]],
    ) -> list[Buffer | None]:
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

    async def set_partial_values(self, key_start_values: Iterable[tuple[str, int, bytes]]) -> None:
        raise NotImplementedError

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._check_writable()
        with self._lock:
            members = self._zf.namelist()
            if key not in members:
                self._set(key, value)

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
        with self._lock:
            for key in self._zf.namelist():
                yield key

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
        async for key in self.list():
            if key.startswith(prefix):
                yield key.removeprefix(prefix)

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        if prefix.endswith("/"):
            prefix = prefix[:-1]

        keys = self._zf.namelist()
        seen = set()
        if prefix == "":
            keys_unique = {k.split("/")[0] for k in keys}
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
