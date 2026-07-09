from __future__ import annotations

import os
import shutil
import threading
import time
import warnings
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
    from collections.abc import AsyncIterator, Iterable

ZipStoreAccessModeLiteral = Literal["r", "w", "a"]

# Marker used to flag a zip entry as soft-deleted.
#
# IMPORTANT: this is stored in the ZipInfo *comment* field (per-entry
# central-directory metadata), not in the entry's actual data content.
# An earlier version of this fix wrote a sentinel *value* as the entry's
# data (first b"", later a long "unique" byte string). Both are unsafe:
# Zarr payload bytes are arbitrary, so a legitimate value can always
# equal whatever sentinel is chosen -- Hypothesis's stateful property
# tests proved this by finding a falsifying example that set a key's
# value to the literal sentinel bytes, which made list()/exists()/get()
# incorrectly report a live key as deleted.
#
# Using ZipInfo.comment instead sidesteps the problem entirely: it is
# metadata the store never exposes through set()/get(), so it cannot
# collide with any byte string a caller stores as data, no matter what
# that data is. It's also persisted to the zip's central directory, so
# soft-deletes made in one process are still visible after the archive
# is closed and reopened elsewhere.
_SOFT_DELETE_MARKER = b"__zarr_zipstore_soft_delete__"


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

    Notes
    -----
    Deletion is implemented as a soft-delete: since the ZIP format does not
    support removing entries in place, `delete()` appends a new, empty entry
    under the same name and flags it via the ZipInfo *comment* field (not its
    data). All read, exists, and list operations check that flag and treat
    matching entries as absent. Because deletions are recorded in the zip's
    central directory rather than kept only in memory, they remain in effect
    even if the store is closed and the same file reopened later. The
    original, physical bytes for a soft-deleted entry stay on disk (the
    archive is never rewritten), but are unreachable through the store API.
    """

    supports_writes: bool = True
    supports_deletes: bool = True  # soft-delete via a flagged empty entry
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

    def _is_soft_deleted(self, key: str) -> bool:
        """Return True if `key`'s most recent entry is a soft-delete marker."""
        try:
            info = self._zf.getinfo(key)
        except KeyError:
            return False
        return info.comment == _SOFT_DELETE_MARKER

    def _get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if not self._is_open:
            self._sync_open()
        # docstring inherited
        if self._is_soft_deleted(key):
            return None
        try:
            with self._zf.open(key) as f:
                data = f.read()
        except KeyError:
            return None

        if byte_range is None:
            return prototype.buffer.from_bytes(data)
        elif isinstance(byte_range, RangeByteRequest):
            return prototype.buffer.from_bytes(data[byte_range.start : byte_range.end])
        elif isinstance(byte_range, OffsetByteRequest):
            return prototype.buffer.from_bytes(data[byte_range.offset :])
        elif isinstance(byte_range, SuffixByteRequest):
            return prototype.buffer.from_bytes(data[max(0, len(data) - byte_range.suffix) :])
        else:
            raise TypeError(f"Unexpected byte_range, got {byte_range}.")

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
        # keyinfo.comment defaults to b"", so a fresh write is never
        # mistaken for a soft-delete marker regardless of `value`.
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

    async def delete(self, key: str) -> None:
        # docstring inherited
        # Soft-delete: append an empty entry under this name, flagged via
        # the ZipInfo comment field (see _SOFT_DELETE_MARKER above). The
        # comment -- not the data -- is what read/list/exists check, so
        # this can never be confused with a real value, including a real
        # value that happens to be empty bytes.
        self._check_writable()
        with self._lock:
            if key in self._zf.namelist():
                keyinfo = zipfile.ZipInfo(filename=key, date_time=time.localtime(time.time())[:6])
                keyinfo.compress_type = self.compression
                keyinfo.external_attr = 0o644 << 16  # ?rw-r--r--
                keyinfo.comment = _SOFT_DELETE_MARKER
                # zipfile.writestr() warns "Duplicate name" whenever a name
                # is written more than once, which is exactly what
                # soft-delete does on purpose. That warning is expected and
                # harmless here (unlike a real overwrite via set(), which
                # intentionally keeps warning -- see test_api_integration),
                # so it is suppressed rather than allowed to propagate under
                # this project's filterwarnings = "error" pytest config.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self._zf.writestr(keyinfo, b"")

    async def delete_dir(self, prefix: str) -> None:
        # docstring inherited
        # Collect all live keys under the prefix first, then soft-delete each.
        self._check_writable()
        if prefix != "" and not prefix.endswith("/"):
            prefix += "/"
        keys_to_delete = [key async for key in self.list_prefix(prefix)]
        for key in keys_to_delete:
            await self.delete(key)

    async def exists(self, key: str) -> bool:
        # docstring inherited
        if not self._is_open:
            self._sync_open()
        with self._lock:
            return not self._is_soft_deleted(key) and key in self._zf.namelist()

    async def list(self) -> AsyncIterator[str]:
        # docstring inherited
        if not self._is_open:
            self._sync_open()
        with self._lock:
            seen: set[str] = set()
            for key in self._zf.namelist():
                if key in seen:
                    continue
                seen.add(key)
                if not self._is_soft_deleted(key):
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
        keys = [k async for k in self.list()]
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
        """Move the store to another path."""
        if isinstance(path, str):
            path = Path(path)
        self.close()
        os.makedirs(path.parent, exist_ok=True)
        shutil.move(self.path, path)
        self.path = path
        await self._open()
