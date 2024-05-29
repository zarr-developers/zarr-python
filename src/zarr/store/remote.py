from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import fsspec

from zarr.abc.store import Store
from zarr.store.core import _dereference_path

if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem
    from upath import UPath

    from zarr.buffer import Buffer, BytesLike


class RemoteStore(Store):
    # based on FSSpec
    supports_writes: bool = True
    supports_partial_writes: bool = False
    supports_listing: bool = True

    _fs: AsyncFileSystem
    exceptions = tuple[type[Exception], ...]

    def __init__(
        self,
        url: UPath | str,
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
        allowed_exceptions: when fetching data, these cases will be deemed to correspond to missinf
            keys, rather than some other IO failure
        storage_options: passed on to fsspec to make the filesystem instance. If url is a UPath,
            this must not be used.
        """

        if isinstance(url, str):
            self._fs, self.path = fsspec.url_to_fs(url, **storage_options)
        elif hasattr(url, "protocol") and hasattr(url, "fs"):
            # is UPath-like - but without importing
            if storage_options:
                raise ValueError(
                    "If constructed with a UPath object, no additional "
                    "storage_options are allowed"
                )
            self.path = url.path
            self._fs = url._fs
        else:
            raise ValueError("URL not understood, %s", url)
        # dear mypy: these have the same type annotations
        self.exceptions = allowed_exceptions  # type: ignore
        # test instantiate file system
        if not self._fs.async_impl:
            raise TypeError("FileSystem needs to support async operations")

    def __str__(self) -> str:
        return f"Remote fsspec store: {self.path}"

    def __repr__(self) -> str:
        return f"<FsspecStore({self.path})>"

    async def get(
        self, key: str, byte_range: tuple[int | None, int | None] | None | None = None
    ) -> Buffer | None:
        path = _dereference_path(self.path, key)

        try:
            return await (
                self._fs._cat_file(path, start=byte_range[0], end=byte_range[1])
                if byte_range
                else self._fs._cat_file(path)
            )
        # dear mypy: this is indeed defined as a tuple of exceptions
        except self.exceptions:  # type: ignore
            return None

    async def set(
        self,
        key: str,
        value: Buffer,
        byte_range: tuple[int, int] | None = None,
    ) -> None:
        path = _dereference_path(self.path, key)
        # write data
        if byte_range:
            raise NotImplementedError
        await self._fs._pipe_file(path, value)

    async def delete(self, key: str) -> None:
        path = _dereference_path(self.path, key)
        try:
            await self._fs._rm(path)
        # dear mypy: yes, I can add a tuple to a tuple
        except (FileNotFoundError,) + self.exceptions:  # type: ignore
            pass

    async def exists(self, key: str) -> bool:
        path = _dereference_path(self.path, key)
        return await self._fs._exists(path)

    async def get_partial_values(
        self, key_ranges: list[tuple[str, tuple[int | None, int | None]]]
    ) -> list[Buffer | None]:
        paths, starts, stops = zip(
            *((_dereference_path(self.path, k[0]), k[1][0], k[1][1]) for k in key_ranges),
            strict=False,
        )
        # TODO: expectations for exceptions or missing keys?
        res = await self._fs._cat_ranges(list(paths), starts, stops, on_error="return")
        for r in res:
            if isinstance(r, Exception) and not isinstance(r, self.exceptions):
                raise r

        return [None if isinstance(r, Exception) else r for r in res]

    async def set_partial_values(self, key_start_values: list[tuple[str, int, BytesLike]]) -> None:
        raise NotImplementedError

    async def list(self) -> AsyncGenerator[str, None]:
        allfiles = await self._fs._find(self.path, detail=False, withdirs=False)
        for onefile in (a.replace(self.path + "/", "") for a in allfiles):
            yield onefile

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        prefix = prefix.rstrip("/")
        allfiles = await self._fs._ls(prefix, detail=False)
        for onefile in (a.replace(prefix + "/", "") for a in allfiles):
            yield onefile

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        for onefile in await self._fs._ls(prefix, detail=False):
            yield onefile
