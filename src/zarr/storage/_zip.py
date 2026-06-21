from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable

ZipStoreAccessModeLiteral = Literal["r", "w", "a"]


class ZipStore(Store):
    """
    Store using a ZIP file.

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
    supports_listing
    path
    compression
    allowZip64
    """

    supports_writes: bool = True
    supports_deletes: bool = True
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
        read_only: bool | None = None,
        compression: int = zipfile.ZIP_STORED,
        allowZip64: bool = True,
    ) -> None:
        if read_only is None:
            read_only = mode == "r"

        super().__init__(read_only=read_only)

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

    def __getstate__(self) -> dict[str, Any]:
        # We need a copy to not modify the state of the original store
        state = self.__dict__.copy()
        for attr in ["_zf", "_lock"]:
            state.pop(attr, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
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
            os.remove(self.path)
            self._zf = zipfile.ZipFile(
                self.path, mode="w", compression=self.compression, allowZip64=self.allowZip64
            )

    def __str__(self) -> str:
        return f"zip://{self.path}"

    def __repr__(self) -> str:
        return f"ZipStore('{self}')"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.path == other.path

    def _get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if not self._is_open:
            self._sync_open()
        # docstring inherited
        try:
            with self._zf.open(key) as f:  # will raise KeyError
                if byte_range is None:
                    return prototype.buffer.from_bytes(f.read())
                elif isinstance(byte_range, RangeByteRequest):
                    f.seek(byte_range.start)
                    return prototype.buffer.from_bytes(f.read(byte_range.end - f.tell()))
                size = f.seek(0, os.SEEK_END)
                if isinstance(byte_range, OffsetByteRequest):
                    f.seek(byte_range.offset)
                elif isinstance(byte_range, SuffixByteRequest):
                    f.seek(max(0, size - byte_range.suffix))
                else:
                    raise TypeError(f"Unexpected byte_range, got {byte_range}.")
                return prototype.buffer.from_bytes(f.read())
        except KeyError:
            return None

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        assert isinstance(key, str)

        with self._lock:
            return self._get(key, prototype=prototype, byte_range=byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # docstring inherited
        out = []
        with self._lock:
            for key, byte_range in key_ranges:
                out.append(self._get(key, prototype=prototype, byte_range=byte_range))
        return out

    def _set(self, key: str, value: Buffer) -> None:
        if not self._is_open:
            self._sync_open()
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
        # docstring inherited
        self._check_writable()
        if not self._is_open:
            self._sync_open()
        assert isinstance(key, str)
        if not isinstance(value, Buffer):
            raise TypeError(
                f"ZipStore.set(): `value` must be a Buffer instance. Got an instance of {type(value)} instead."
            )
        with self._lock:
            self._set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._check_writable()
        with self._lock:
            members = self._zf.namelist()
            if key not in members:
                self._set(key, value)

    def _rewrite_without(self, should_delete: Callable[[str], bool]) -> None:
        # Rewrite the archive, dropping every member for which ``should_delete``
        # returns True. ZIP files do not support in-place deletion, so the only
        # way to remove an entry is to copy the surviving entries into a fresh
        # archive (see issue #828). Duplicate members (created when a chunk is
        # overwritten via ``writestr``) are compacted to their most recent value
        # as a side effect, since that is what reads already return.
        #
        # This must be called while holding ``self._lock``.
        members: dict[str, zipfile.ZipInfo] = {}
        for info in self._zf.infolist():
            members[info.filename] = info  # keep the last entry for each name

        to_delete = [name for name in members if should_delete(name)]
        if not to_delete:
            # nothing matched; leave the archive untouched
            return

        fd, tmp_path = tempfile.mkstemp(dir=self.path.parent)
        os.close(fd)
        try:
            with zipfile.ZipFile(
                tmp_path, mode="w", compression=self.compression, allowZip64=self.allowZip64
            ) as new_zf:
                for name, info in members.items():
                    if should_delete(name):
                        continue
                    new_zf.writestr(info, self._zf.read(name))
            self._zf.close()
            os.replace(tmp_path, self.path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        # Reopen in append mode so subsequent writes preserve the archive
        # (the original mode may be "w"/"x", which would truncate or fail).
        self._zf = zipfile.ZipFile(
            self.path, mode="a", compression=self.compression, allowZip64=self.allowZip64
        )

    async def delete_dir(self, prefix: str) -> None:
        # docstring inherited
        self._check_writable()
        if not self._is_open:
            self._sync_open()
        if prefix != "" and not prefix.endswith("/"):
            prefix += "/"
        with self._lock:
            self._rewrite_without(lambda name: name.startswith(prefix))

    async def delete(self, key: str) -> None:
        # docstring inherited
        # deleting a missing key is a no-op, matching the other stores
        self._check_writable()
        if not self._is_open:
            self._sync_open()
        with self._lock:
            self._rewrite_without(lambda name: name == key)

    async def exists(self, key: str) -> bool:
        # docstring inherited
        if not self._is_open:
            self._sync_open()
        with self._lock:
            try:
                self._zf.getinfo(key)
            except KeyError:
                return False
            else:
                return True

    async def list(self) -> AsyncIterator[str]:
        # docstring inherited
        if not self._is_open:
            self._sync_open()
        with self._lock:
            for key in self._zf.namelist():
                yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        if not self._is_open:
            self._sync_open()
        prefix = prefix.rstrip("/")

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
                if key.startswith(f"{prefix}/") and key.strip("/") != prefix:
                    k = key.removeprefix(f"{prefix}/").split("/")[0]
                    if k not in seen:
                        seen.add(k)
                        yield k

    async def move(self, path: Path | str) -> None:
        """
        Move the store to another path.
        """
        if isinstance(path, str):
            path = Path(path)
        self.close()
        os.makedirs(path.parent, exist_ok=True)
        shutil.move(self.path, path)
        self.path = path
        await self._open()
