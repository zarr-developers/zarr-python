from __future__ import annotations

import os
import threading
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from zarr.abc.store import ByteRangeRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable
    from typing import Self

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

    Attributes
    ----------
    allowed_exceptions
    supports_writes
    supports_deletes
    supports_partial_writes
    supports_listing
    path
    compression
    allowZip64
    """

    supports_writes: bool = True
    supports_deletes: bool = False
    supports_partial_writes: bool = False
    supports_listing: bool = True

    file_path: Path
    compression: int
    allowZip64: bool

    _zf: zipfile.ZipFile
    _lock: threading.RLock

    def __init__(
        self,
        file_path: Path | str,
        *,
        path: str = "",
        mode: ZipStoreAccessModeLiteral = "r",
        compression: int = zipfile.ZIP_STORED,
        allowZip64: bool = True,
    ) -> None:
        super().__init__(mode=mode, path=path)

        if isinstance(file_path, str):
            file_path = Path(file_path)
        self.file_path = file_path

        self._zmode = mode
        self.compression = compression
        self.allowZip64 = allowZip64

    def _sync_open(self) -> None:
        if self._is_open:
            raise ValueError("store is already open")

        self._lock = threading.RLock()

        self._zf = zipfile.ZipFile(
            self.file_path,
            mode=self._zmode,
            compression=self.compression,
            allowZip64=self.allowZip64,
        )

        self._is_open = True

    async def _open(self) -> None:
        self._sync_open()

    def __getstate__(self) -> tuple[str, Path, ZipStoreAccessModeLiteral, int, bool]:
        return self.path, self.file_path, self._zmode, self.compression, self.allowZip64

    def __setstate__(self, state: tuple[str, Path, ZipStoreAccessModeLiteral, int, bool]) -> None:
        self.path, self.file_path, self._zmode, self.compression, self.allowZip64 = state
        self._is_open = False
        self._sync_open()

    def close(self) -> None:
        # docstring inherited
        super().close()
        with self._lock:
            self._zf.close()

    async def clear(self) -> None:
        # docstring inherited
        with self._lock:
            self._check_writable()
            self._zf.close()
            os.remove(self.file_path)
            self._zf = zipfile.ZipFile(
                self.file_path, mode="w", compression=self.compression, allowZip64=self.allowZip64
            )

    async def empty(self) -> bool:
        # docstring inherited
        with self._lock:
            return not self._zf.namelist()

    def with_mode(self, mode: ZipStoreAccessModeLiteral) -> Self:  # type: ignore[override]
        # docstring inherited
        raise NotImplementedError("ZipStore cannot be reopened with a new mode.")

    def __str__(self) -> str:
        # lets try https://github.com/zarr-developers/zeps/pull/48/files
        return f"file://{self.file_path}|zip://{self.path}"

    def __repr__(self) -> str:
        return f"ZipStore({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.file_path == other.file_path
            and self.path == other.path
            and self._zmode == other._zmode
        )

    def _get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRangeRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        # assume the key has already been made absolute
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
        # docstring inherited
        assert isinstance(key, str)

        with self._lock:
            return self._get(self.resolve_key(key), prototype=prototype, byte_range=byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRangeRequest]],
    ) -> list[Buffer | None]:
        # docstring inherited
        out = []
        with self._lock:
            for key, byte_range in key_ranges:
                out.append(
                    self._get(self.resolve_key(key), prototype=prototype, byte_range=byte_range)
                )
        return out

    def _set(self, key: str, value: Buffer) -> None:
        # generally, this should be called inside a lock
        # assume that the key has already been made absolute
        keyinfo = zipfile.ZipInfo(filename=key, date_time=time.localtime(time.time())[:6])
        keyinfo.compress_type = self.compression
        if keyinfo.filename[-1] == os.sep:
            keyinfo.external_attr = 0o40775 << 16  # drwxrwxr-x
            keyinfo.external_attr |= 0x10  # MS-DOS directory flag
        else:
            keyinfo.external_attr = 0o644 << 16  # ?rw-r--r--
        self._zf.writestr(keyinfo, value.to_bytes())

    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        self._check_writable()
        assert isinstance(key, str)
        if not isinstance(value, Buffer):
            raise TypeError("ZipStore.set(): `value` must a Buffer instance")
        with self._lock:
            self._set(self.resolve_key(key), value)

    async def set_partial_values(self, key_start_values: Iterable[tuple[str, int, bytes]]) -> None:
        raise NotImplementedError

    async def set_if_not_exists(self, key: str, default: Buffer) -> None:
        key_abs = self.resolve_key(key)
        self._check_writable()
        with self._lock:
            members = self._zf.namelist()
            if key_abs not in members:
                self._set(key_abs, default)

    async def delete(self, key: str) -> None:
        # docstring inherited
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        # docstring inherited
        with self._lock:
            try:
                self._zf.getinfo(self.resolve_key(key))
            except KeyError:
                return False
            else:
                return True

    async def list(self) -> AsyncGenerator[str, None]:
        # docstring inherited
        with self._lock:
            for key in self._zf.namelist():
                yield key.lstrip("/")

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        async for key in self.list():
            if key.startswith(prefix):
                yield key.removeprefix(prefix)

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        prefix_abs = self.resolve_key(prefix)

        keys = self._zf.namelist()
        seen = set()
        if prefix_abs == "":
            keys_unique = {k.split("/")[0] for k in keys}
            for key in keys_unique:
                if key not in seen:
                    seen.add(key)
                    yield key
        else:
            for key in keys:
                if key.startswith(prefix_abs + "/") and key != prefix_abs:
                    k = key.removeprefix(prefix_abs + "/").split("/")[0]
                    if k not in seen:
                        seen.add(k)
                        yield k
