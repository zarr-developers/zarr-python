"""
The read-only `zip:` URL pipeline adapter.

Per the URL pipeline specification (`schemes/zip.md`), `zip:` addresses an
entry or directory within the ZIP archive that the preceding pipeline
resolves to:

    s3://bucket/archive.zip|zip:path/within/archive|zarr3:

The archive is read through the preceding store with async byte-range
requests: the central directory is fetched from the end of the archive once,
and each member is fetched (and decompressed) on demand. This works the same
way for local, fsspec, in-memory and nested (`zip:` inside `zip:`) archives,
and never blocks zarr's I/O event loop on file I/O.
"""

from __future__ import annotations

import asyncio
import bz2
import dataclasses
import io
import struct
import zipfile
import zlib
from typing import TYPE_CHECKING

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.abc.url_pipeline import AdapterResolution, URLPipelineAdapter
from zarr.core.buffer import default_buffer_prototype
from zarr.errors import URLPipelineError
from zarr.storage._utils import normalize_path

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from zarr.abc.url_pipeline import PipelineContext, PipelineSegment
    from zarr.core.buffer import Buffer, BufferPrototype

__all__ = ["ZipAdapter", "ZipReaderStore"]

# The end-of-central-directory record is 22 bytes plus a comment of at most
# 65535 bytes, so this suffix always contains it.
_EOCD_SEARCH_BYTES = 22 + 0xFFFF
# Smallest range fetched when the central directory extends past the suffix.
_MIN_FETCH_BYTES = 1 << 16
# Members whose compressed size exceeds this are decompressed in a thread.
_THREAD_DECOMPRESS_BYTES = 1 << 20

_LOCAL_HEADER = struct.Struct("<4s2B4HL2L2H")
_LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"
_WRITE_MODES = frozenset({"w", "w-", "r+"})


async def _read_resource_range(
    store: Store, key: str, byte_range: ByteRequest | None = None
) -> bytes:
    """
    Read bytes of the single-object resource at `key` in `store`.

    This is the one place that relies on a `file`-kind resource being readable
    as a store key: an archive that is an entry of another store (e.g. a nested
    `zip:`) is the value at `key`; an archive that *is* the preceding root (a
    local file, an fsspec object) is read with the empty key, which `LocalStore`
    and `FsspecStore` happen to resolve to the root object itself.
    """
    value = await store.get(key, prototype=default_buffer_prototype(), byte_range=byte_range)
    if value is None:
        raise FileNotFoundError(f"{key!r} in {store}" if key else str(store))
    return value.to_bytes()


async def _resource_size(store: Store, key: str) -> int:
    """The size in bytes of the single-object resource at `key` in `store`."""
    try:
        return await store.getsize(key)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{key!r} in {store}" if key else str(store)) from exc


class _MissingRangeError(Exception):
    def __init__(self, start: int, stop: int) -> None:
        super().__init__(start, stop)
        self.start = start
        self.stop = stop


class _SparseFile(io.RawIOBase):
    """
    A read-only, seekable view of a file of known size of which only some
    byte ranges have been fetched. Reading outside them raises
    `_MissingRangeError`, so the caller can fetch the range and retry.
    """

    def __init__(self, size: int, segments: dict[int, bytes]) -> None:
        self._size = size
        self._segments = segments
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = pos
        elif whence == io.SEEK_CUR:
            self._pos += pos
        else:
            self._pos = self._size + pos
        return self._pos

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        for start, data in self._segments.items():
            if start <= self._pos and self._pos + n <= start + len(data):
                offset = self._pos - start
                b[:n] = data[offset : offset + n]
                self._pos += n
                return n
        raise _MissingRangeError(self._pos, self._pos + n)


def _parse_central_directory(size: int, segments: dict[int, bytes]) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(_SparseFile(size, segments)) as zf:
        return zf.infolist()


def _decompress(info: zipfile.ZipInfo, data: bytes) -> bytes:
    if info.compress_type == zipfile.ZIP_STORED:
        result = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        result = zlib.decompress(data, -15)
    elif info.compress_type == zipfile.ZIP_BZIP2:
        result = bz2.decompress(data)
    else:
        raise NotImplementedError(
            f"ZIP member {info.filename!r} uses compression method {info.compress_type}, "
            "which zip: does not support (supported: stored, deflate, bzip2)"
        )
    if zlib.crc32(result) != info.CRC:
        raise OSError(f"bad CRC-32 for ZIP member {info.filename!r}")
    return result


def _slice(data: bytes, byte_range: ByteRequest | None) -> bytes:
    if byte_range is None:
        return data
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        return data[max(0, len(data) - byte_range.suffix) :]
    raise TypeError(f"Unexpected byte_range, got {byte_range}.")


class ZipReaderStore(Store):
    """
    A read-only store over a ZIP archive that is itself a resource of another store.

    The archive is the value at `key` in `source` (the empty key addresses a
    store rooted at a single object, such as a local file). It is read with
    async range requests only. Use `open_archive` to create one; the store owns
    `source` and closes it on `close`.

    Parameters
    ----------
    source : Store
        The store holding the archive.
    key : str
        The key of the archive in `source`.
    size : int
        The archive size in bytes.
    infos : Iterable[zipfile.ZipInfo]
        The archive's central directory entries.
    """

    supports_writes: bool = False
    supports_deletes: bool = False
    supports_listing: bool = True

    def __init__(
        self, source: Store, key: str, size: int, infos: Iterable[zipfile.ZipInfo]
    ) -> None:
        super().__init__(read_only=True)
        self._source = source
        self._key = key
        self._size = size
        # zarr keys are files: directory entries carry no data
        self._infos = {info.filename: info for info in infos if not info.is_dir()}
        self._data_offsets: dict[str, int] = {}
        self._is_open = True

    @classmethod
    async def open_archive(cls, source: Store, key: str = "") -> ZipReaderStore:
        """
        Read the central directory of the archive at `key` in `source`.

        Parameters
        ----------
        source : Store
            The store holding the archive.
        key : str, optional
            The key of the archive in `source`; empty for a store rooted at
            the archive itself.

        Returns
        -------
        ZipReaderStore
            An open, read-only store over the archive.

        Raises
        ------
        zipfile.BadZipFile
            If the resource is not a ZIP archive.
        FileNotFoundError
            If there is no resource at `key`.
        """
        size = await _resource_size(source, key)
        tail = min(size, _EOCD_SEARCH_BYTES)
        segments = {
            size - tail: await _read_resource_range(
                source, key, RangeByteRequest(size - tail, size)
            )
        }
        while True:
            try:
                # parsing a large central directory is CPU-bound
                infos = await asyncio.to_thread(_parse_central_directory, size, segments)
            except _MissingRangeError as missing:
                stop = min(size, max(missing.stop, missing.start + _MIN_FETCH_BYTES))
                segments[missing.start] = await _read_resource_range(
                    source, key, RangeByteRequest(missing.start, stop)
                )
                continue
            return cls(source, key, size, infos)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._source == other._source
            and self._key == other._key
        )

    def __str__(self) -> str:
        return f"{self._source}|zip:" if not self._key else f"{self._source}/{self._key}|zip:"

    def __repr__(self) -> str:
        return f"ZipReaderStore('{self}')"

    def close(self) -> None:
        # docstring inherited
        super().close()
        self._source.close()

    async def _data_offset(self, info: zipfile.ZipInfo) -> int:
        """The archive offset of a member's data, after its local header."""
        offset = self._data_offsets.get(info.filename)
        if offset is None:
            header = await _read_resource_range(
                self._source,
                self._key,
                RangeByteRequest(info.header_offset, info.header_offset + _LOCAL_HEADER.size),
            )
            fields = _LOCAL_HEADER.unpack(header)
            if fields[0] != _LOCAL_HEADER_SIGNATURE:
                raise zipfile.BadZipFile(f"bad local file header for {info.filename!r}")
            name_length, extra_length = fields[-2], fields[-1]
            offset = info.header_offset + _LOCAL_HEADER.size + name_length + extra_length
            self._data_offsets[info.filename] = offset
        return offset

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        info = self._infos.get(key)
        if info is None:
            return None
        if info.flag_bits & 0x1:
            raise NotImplementedError(f"ZIP member {key!r} is encrypted")
        start = await self._data_offset(info)
        if info.compress_type == zipfile.ZIP_STORED and byte_range is not None:
            # uncompressed members support true range reads
            length = info.file_size
            if isinstance(byte_range, RangeByteRequest):
                lo, hi = min(byte_range.start, length), min(byte_range.end, length)
            elif isinstance(byte_range, OffsetByteRequest):
                lo, hi = min(byte_range.offset, length), length
            elif isinstance(byte_range, SuffixByteRequest):
                lo, hi = max(0, length - byte_range.suffix), length
            else:
                raise TypeError(f"Unexpected byte_range, got {byte_range}.")
            if hi <= lo:
                return prototype.buffer.from_bytes(b"")
            data = await _read_resource_range(
                self._source, self._key, RangeByteRequest(start + lo, start + hi)
            )
            return prototype.buffer.from_bytes(data)
        raw = await _read_resource_range(
            self._source, self._key, RangeByteRequest(start, start + info.compress_size)
        )
        if info.compress_size > _THREAD_DECOMPRESS_BYTES:
            data = await asyncio.to_thread(_decompress, info, raw)
        else:
            data = _decompress(info, raw)
        return prototype.buffer.from_bytes(_slice(data, byte_range))

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # docstring inherited
        return list(
            await asyncio.gather(
                *(self.get(key, prototype, byte_range) for key, byte_range in key_ranges)
            )
        )

    async def getsize(self, key: str) -> int:
        # docstring inherited
        info = self._infos.get(key)
        if info is None:
            raise FileNotFoundError(key)
        return info.file_size

    async def exists(self, key: str) -> bool:
        # docstring inherited
        return key in self._infos

    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        self._check_writable()

    async def delete(self, key: str) -> None:
        # docstring inherited
        self._check_writable()

    async def list(self) -> AsyncIterator[str]:
        # docstring inherited
        for key in self._infos:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        for key in self._infos:
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        prefix = prefix.rstrip("/")
        prefix = f"{prefix}/" if prefix else ""
        seen: set[str] = set()
        for key in self._infos:
            if key.startswith(prefix):
                child = key[len(prefix) :].split("/", 1)[0]
                if child and child not in seen:
                    seen.add(child)
                    yield child


def _parse_zip_path(segment: PipelineSegment) -> str:
    """The path within the archive addressed by a `zip:` segment."""
    if segment.query is not None:
        raise URLPipelineError(f"'zip:' pipeline segments do not accept a query: {segment.raw!r}")
    body = segment.body
    if body.startswith("//"):
        raise URLPipelineError(
            f"invalid 'zip:' path in {segment.raw!r}: more than one leading '/' is not allowed"
        )
    try:
        return normalize_path(body.removeprefix("/"))
    except ValueError as exc:
        raise URLPipelineError(f"invalid path in pipeline segment {segment.raw!r}: {exc}") from exc


class ZipAdapter(URLPipelineAdapter):
    """
    The `zip:` adapter: an entry or directory within a ZIP archive (read-only).
    """

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> AdapterResolution:
        """
        Resolve a `zip:` segment.

        Parameters
        ----------
        segment : PipelineSegment
            The `zip:` segment. Its body is a path within the archive.
        context : PipelineContext
            The pipeline to the left of the segment, which must resolve to the archive.

        Returns
        -------
        AdapterResolution
            A read-only store over the archive, with the segment path as the residual path.

        Raises
        ------
        URLPipelineError
            If the mode requires writing, the segment is malformed, or the preceding
            resource is not a readable ZIP archive.
        """
        if context.mode in _WRITE_MODES:
            raise URLPipelineError(
                f"zip: is read-only until writable ZIP support lands; cannot open "
                f"{segment.raw!r} with mode={context.mode!r}. Use mode='r' (or 'a' to open "
                "an existing archive), or write the archive with zarr.storage.ZipStore."
            )
        path = _parse_zip_path(segment)
        preceding = await context.resolve_preceding(mode="r")
        try:
            store = await ZipReaderStore.open_archive(
                preceding.store, normalize_path(preceding.path)
            )
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            preceding.store.close()
            raise URLPipelineError(
                f"could not open the ZIP archive {context.preceding_url!r} for "
                f"{segment.raw!r}: {exc}"
            ) from exc
        except BaseException:
            preceding.store.close()
            raise
        return dataclasses.replace(preceding, store=store, path=path)
