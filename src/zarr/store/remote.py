from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import fsspec

from zarr.abc.store import Store
from zarr.buffer import Buffer, BufferPrototype
from zarr.common import OpenMode
from zarr.store.core import _dereference_path

if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem
    from upath import UPath

    from zarr.buffer import Buffer
    from zarr.common import BytesLike


class RemoteStore(Store):
    # based on FSSpec
    supports_writes: bool = True
    supports_partial_writes: bool = False
    supports_listing: bool = True

    _fs: AsyncFileSystem
    _url: str
    path: str
    allowed_exceptions: tuple[type[Exception], ...]

    def __init__(
        self,
        url: UPath | str,
        mode: OpenMode = "r",
        allowed_exceptions: tuple[type[Exception], ...] = (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
        ),
        **storage_options: Any,
    ):
        """
        Parameters
        ----------
        url: root of the datastore. In fsspec notation, this is usually like "protocol://path/to".
            Can also be a upath.UPath instance/
        allowed_exceptions: when fetching data, these cases will be deemed to correspond to missing
            keys, rather than some other IO failure
        storage_options: passed on to fsspec to make the filesystem instance. If url is a UPath,
            this must not be used.
        """

        super().__init__(mode=mode)
        if isinstance(url, str):
            self._url = url.rstrip("/")
            self._fs, _path = fsspec.url_to_fs(url, **storage_options)
            self.path = _path.rstrip("/")
        elif hasattr(url, "protocol") and hasattr(url, "fs"):
            # is UPath-like - but without importing
            if storage_options:
                raise ValueError(
                    "If constructed with a UPath object, no additional "
                    "storage_options are allowed"
                )
            # n.b. UPath returns the url and path attributes with a trailing /, at least for s3
            # that trailing / must be removed to compose with the store interface
            self._url = str(url).rstrip("/")
            self.path = url.path.rstrip("/")
            self._fs = url.fs
        else:
            raise ValueError(f"URL not understood, {url}")
        self.allowed_exceptions = allowed_exceptions
        # test instantiate file system
        if not self._fs.async_impl:
            raise TypeError("FileSystem needs to support async operations")

    def __str__(self) -> str:
        return f"{self._url}"

    def __repr__(self) -> str:
        return f"<RemoteStore({type(self._fs).__name__}, {self.path})>"

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        path = _dereference_path(self.path, key)

        try:
            if byte_range:
                # fsspec uses start/end, not start/length
                start, length = byte_range
                if start is not None and length is not None:
                    end = start + length
                elif length is not None:
                    end = length
                else:
                    end = None
            value = prototype.buffer.from_bytes(
                await (
                    self._fs._cat_file(path, start=byte_range[0], end=end)
                    if byte_range
                    else self._fs._cat_file(path)
                )
            )
            return value

        except self.allowed_exceptions:
            return None
        except OSError as e:
            if "not satisfiable" in str(e):
                # this is an s3-specific condition we probably don't want to leak
                return prototype.buffer.from_bytes(b"")
            raise

    async def set(
        self,
        key: str,
        value: Buffer,
        byte_range: tuple[int, int] | None = None,
    ) -> None:
        self._check_writable()
        path = _dereference_path(self.path, key)
        # write data
        if byte_range:
            raise NotImplementedError
        await self._fs._pipe_file(path, value.to_bytes())

    async def delete(self, key: str) -> None:
        self._check_writable()
        path = _dereference_path(self.path, key)
        try:
            await self._fs._rm(path)
        except FileNotFoundError:
            pass
        except self.allowed_exceptions:
            pass

    async def exists(self, key: str) -> bool:
        path = _dereference_path(self.path, key)
        exists: bool = await self._fs._exists(path)
        return exists

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: list[tuple[str, tuple[int | None, int | None]]],
    ) -> list[Buffer | None]:
        if key_ranges:
            paths, starts, stops = zip(
                *(
                    (
                        _dereference_path(self.path, k[0]),
                        k[1][0],
                        ((k[1][0] or 0) + k[1][1]) if k[1][1] is not None else None,
                    )
                    for k in key_ranges
                ),
                strict=False,
            )
        else:
            return []
        # TODO: expectations for exceptions or missing keys?
        res = await self._fs._cat_ranges(list(paths), starts, stops, on_error="return")
        # the following is an s3-specific condition we probably don't want to leak
        res = [b"" if (isinstance(r, OSError) and "not satisfiable" in str(r)) else r for r in res]
        for r in res:
            if isinstance(r, Exception) and not isinstance(r, self.allowed_exceptions):
                raise r

        return [None if isinstance(r, Exception) else prototype.buffer.from_bytes(r) for r in res]

    async def set_partial_values(self, key_start_values: list[tuple[str, int, BytesLike]]) -> None:
        raise NotImplementedError

    async def list(self) -> AsyncGenerator[str, None]:
        allfiles = await self._fs._find(self.path, detail=False, withdirs=False)
        for onefile in (a.replace(self.path + "/", "") for a in allfiles):
            yield onefile

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        prefix = f"{self.path}/{prefix.rstrip('/')}"
        try:
            allfiles = await self._fs._ls(prefix, detail=False)
        except FileNotFoundError:
            return
        for onefile in (a.replace(prefix + "/", "") for a in allfiles):
            yield onefile

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        for onefile in await self._fs._ls(prefix, detail=False):
            yield onefile
