from __future__ import annotations

import io
import os
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal

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

ZipStoreAccessModeLiteral = Literal["r", "w", "a", "x"]


class _RawReaderAdapter(io.RawIOBase):
    """
    Adapt a minimal seekable reader to the `io` interface `zipfile` needs.

    Some file-like objects (e.g. `obstore.ReadableFile`) implement
    `read`/`seek`/`tell` but are not `io.IOBase` instances, and their
    `read` may return a buffer-protocol object rather than `bytes`.
    Wrapping in this adapter plus `io.BufferedReader` yields real `bytes`.

    Reads are clamped to the bytes remaining before EOF: some readers
    (obstore < 0.6) raise on short reads rather than returning fewer bytes.
    The size is cached, which is safe because the adapter is only used for
    read-only access.
    """

    def __init__(self, fileobj: IO[bytes]) -> None:
        self._fileobj = fileobj
        self._size: int | None = None

    def _get_size(self) -> int:
        if self._size is None:
            pos = self._fileobj.tell()
            self._size = self._fileobj.seek(0, os.SEEK_END)
            self._fileobj.seek(pos)
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, pos: int, whence: int = 0) -> int:
        return self._fileobj.seek(pos, whence)

    def tell(self) -> int:
        return self._fileobj.tell()

    def readinto(self, b: Any) -> int:
        n_requested = min(len(b), self._get_size() - self._fileobj.tell())
        if n_requested <= 0:
            return 0
        data = self._fileobj.read(n_requested)
        n = len(data)
        b[:n] = memoryview(data)
        return n


class ZipStore(Store):
    """
    Store using a ZIP file.

    Parameters
    ----------
    path : str, Path, or IO[bytes]
        Location of file, or an open binary file object. A file object must
        support `read`, `seek`, and `tell`; objects that are not `io.IOBase`
        instances (e.g. an `obstore` reader) are adapted automatically but
        can only be used for reading (`mode="r"`). The file object must stay
        open for the lifetime of the store, and operations that require a
        filesystem location (`clear`, `move`, pickling) are not supported.
        Using the store again after `close()` reopens the archive, which
        requires a file object that is readable and seekable; otherwise it
        raises `io.UnsupportedOperation`.
    mode : str, optional
        One of 'r' to read an existing file, 'w' to truncate and write a new
        file, 'a' to append to an existing file, or 'x' to exclusively create
        and write a new file. 'w' and 'x' apply to the first open only; the
        store reopens its archive with 'a' after `close()`, `move()`, or
        unpickling, so the entries it already wrote are kept. If `close()`
        raises, the archive may be incomplete, and every later read or write
        raises `RuntimeError` instead of reopening it; `clear()` replaces the
        archive and makes the store usable again.
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
    supports_deletes: bool = False
    supports_listing: bool = True

    path: Path | None
    compression: int
    allowZip64: bool

    _zf: zipfile.ZipFile
    _lock: threading.RLock
    _fileobj: IO[bytes] | None

    def __init__(
        self,
        path: Path | str | IO[bytes],
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
        if isinstance(path, Path):
            self.path = path  # root?
            self._fileobj = None
        else:
            self.path = None
            if not isinstance(path, io.IOBase):
                if not all(
                    callable(getattr(path, attr, None)) for attr in ("read", "seek", "tell")
                ):
                    raise TypeError(
                        f"expected a path or an open binary file object supporting "
                        f"read/seek/tell, got {type(path).__name__}"
                    )
                if mode != "r":
                    raise TypeError(
                        f"a file object that is not an io.IOBase instance can only be "
                        f"opened for reading (mode='r', got mode={mode!r})"
                    )
                # e.g. an obstore ReadableFile: readable and seekable, but
                # not an io object and reads may not return bytes
                path = io.BufferedReader(_RawReaderAdapter(path))
            self._fileobj = path

        self._zmode = mode
        self.compression = compression
        self.allowZip64 = allowZip64
        self._lock = threading.RLock()
        self._was_opened = False
        self._close_failed = False

    def _sync_open(self) -> None:
        if self._is_open:
            raise ValueError("store is already open")
        if self._close_failed:
            # the central directory may be missing; appending to such a file
            # makes zipfile start a new archive and drop the earlier entries
            raise RuntimeError(
                f"closing the archive of {self!r} failed, so it may be incomplete; "
                "the store will not reopen it, but clear() replaces it"
            )
        if (
            self.path is None
            and self._fileobj is not None
            and self._was_opened
            and not (self._fileobj.readable() and self._fileobj.seekable())
        ):
            # reopening appends, which needs to read the archive back; on a
            # write-only file zipfile would start a new archive and drop the
            # earlier entries
            raise io.UnsupportedOperation(
                "a ZipStore backed by a file object that is not readable and "
                "seekable cannot be used again after close(), because the "
                "archive it wrote cannot be read back"
            )

        self._zf = zipfile.ZipFile(
            self.path if self.path is not None else self._fileobj,  # type: ignore[arg-type]
            mode=self._zmode,
            compression=self.compression,
            allowZip64=self.allowZip64,
        )
        # "w" truncates and "x" refuses an existing file. Both apply only to the
        # first open: reopening after close(), move(), or unpickling must keep
        # the entries already written.
        if self._zmode in ("w", "x"):
            self._zmode = "a"

        self._was_opened = True
        self._is_open = True

    def _zipfile(self) -> zipfile.ZipFile:
        """Return the archive, opening it on first use."""
        with self._lock:
            if not self._is_open:
                self._sync_open()
            return self._zf

    async def _open(self) -> None:
        with self._lock:
            self._sync_open()

    async def _ensure_open(self) -> None:
        # the base class checks _is_open outside the lock
        self._zipfile()

    def __getstate__(self) -> dict[str, Any]:
        if self.path is None:
            # A path-backed store pickles its path and reopens the file on
            # unpickling; an open file object cannot be serialized that way.
            raise TypeError(
                "cannot pickle a ZipStore backed by a file-like object; "
                "construct the store from a path instead"
            )
        # We need a copy to not modify the state of the original store. The
        # lock keeps it from catching _sync_open between opening with "w" and
        # switching to "a", which would make unpickling truncate the archive.
        with self._lock:
            state = self.__dict__.copy()
        for attr in ["_zf", "_lock"]:
            state.pop(attr, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
        self.__dict__.setdefault("_close_failed", False)
        self._lock = threading.RLock()
        self._is_open = False
        self._zipfile()

    def close(self) -> None:
        # docstring inherited
        # hold the lock until the archive is closed: a thread that reopened it
        # before its central directory was written would lose the entries
        with self._lock:
            if not self._is_open:
                return
            try:
                self._zf.close()
            except BaseException:
                self._close_failed = True
                raise
            finally:
                super().close()

    async def clear(self) -> None:
        # docstring inherited
        with self._lock:
            self._check_writable()
            if self.path is None:
                raise NotImplementedError(
                    "clear() is not supported for a ZipStore backed by a file-like object"
                )
            if self._close_failed:
                # replacing the file cannot drop entries, so clear() is the one
                # way to recover a store whose close() failed
                self._close_failed = False
            else:
                # opening first keeps mode "x" from deleting a file it may not claim
                self._zipfile()
                self.close()
            os.remove(self.path)
            # if this open fails the store stays closed, and the next use
            # creates the archive again
            self._zmode = "w"
            self._sync_open()

    def __str__(self) -> str:
        if self.path is None:
            return f"zip://{self._fileobj!r}"
        return f"zip://{self.path}"

    def __repr__(self) -> str:
        return f"ZipStore('{self}')"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.path == other.path
            and self._fileobj is other._fileobj
        )

    def _get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        try:
            with self._zipfile().open(key) as f:  # will raise KeyError
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
        # generally, this should be called inside a lock
        keyinfo = zipfile.ZipInfo(filename=key, date_time=time.localtime(time.time())[:6])
        keyinfo.compress_type = self.compression
        if keyinfo.filename[-1] == os.sep:
            keyinfo.external_attr = 0o40775 << 16  # drwxrwxr-x
            keyinfo.external_attr |= 0x10  # MS-DOS directory flag
        else:
            keyinfo.external_attr = 0o644 << 16  # ?rw-r--r--
        self._zipfile().writestr(keyinfo, value.to_bytes())

    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        self._check_writable()
        if not isinstance(value, Buffer):
            raise TypeError(
                f"ZipStore.set(): `value` must be a Buffer instance. Got an instance of {type(value)} instead."
            )
        with self._lock:
            self._set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._check_writable()
        with self._lock:
            members = self._zipfile().namelist()
            if key not in members:
                self._set(key, value)

    async def delete_dir(self, prefix: str) -> None:
        # only raise NotImplementedError if any keys are found
        self._check_writable()
        if prefix != "" and not prefix.endswith("/"):
            prefix += "/"
        async for _ in self.list_prefix(prefix):
            raise NotImplementedError

    async def delete(self, key: str) -> None:
        # docstring inherited
        # we choose to only raise NotImplementedError here if the key exists
        # this allows the array/group APIs to avoid the overhead of existence checks
        self._check_writable()
        if await self.exists(key):
            raise NotImplementedError

    async def exists(self, key: str) -> bool:
        # docstring inherited
        with self._lock:
            try:
                self._zipfile().getinfo(key)
            except KeyError:
                return False
            else:
                return True

    async def list(self) -> AsyncIterator[str]:
        # docstring inherited
        # namelist() is a copy; holding the lock across yield would block
        # other threads for as long as the caller iterates
        for key in self._zipfile().namelist():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        prefix = prefix.rstrip("/")

        keys = self._zipfile().namelist()
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
        if self.path is None:
            raise NotImplementedError(
                "move() is not supported for a ZipStore backed by a file-like object"
            )
        if isinstance(path, str):
            path = Path(path)
        # hold the lock so that no thread reopens the old path mid-move
        with self._lock:
            # opening first keeps mode "x" from moving a file it may not claim
            self._zipfile()
            self.close()
            os.makedirs(path.parent, exist_ok=True)
            shutil.move(self.path, path)
            self.path = path
            self._sync_open()
