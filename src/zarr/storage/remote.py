from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import fsspec

from zarr.abc.store import ByteRangeRequest, Store
from zarr.storage.common import _dereference_path

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable

    from fsspec.asyn import AsyncFileSystem

    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.core.common import AccessModeLiteral, BytesLike


ALLOWED_EXCEPTIONS: tuple[type[Exception], ...] = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
)


class RemoteStore(Store):
    """
    A remote Store based on FSSpec

    Parameters
    ----------
    fs : AsyncFileSystem
        The Async FSSpec filesystem to use with this store.
    mode : AccessModeLiteral
        The access mode to use.
    path : str
        The root path of the store.
    allowed_exceptions : tuple[type[Exception], ...]
        When fetching data, these cases will be deemed to correspond to missing keys.

    Attributes
    ----------
    fs
    allowed_exceptions
    supports_writes
    supports_deletes
    supports_partial_writes
    supports_listing
    """

    # based on FSSpec
    supports_writes: bool = True
    supports_deletes: bool = True
    supports_partial_writes: bool = False
    supports_listing: bool = True

    fs: AsyncFileSystem
    allowed_exceptions: tuple[type[Exception], ...]

    def __init__(
        self,
        fs: AsyncFileSystem,
        mode: AccessModeLiteral = "r",
        path: str = "/",
        allowed_exceptions: tuple[type[Exception], ...] = ALLOWED_EXCEPTIONS,
    ) -> None:
        super().__init__(mode=mode)
        self.fs = fs
        self.path = path
        self.allowed_exceptions = allowed_exceptions

        if not self.fs.async_impl:
            raise TypeError("Filesystem needs to support async operations.")

    @classmethod
    def from_upath(
        cls,
        upath: Any,
        mode: AccessModeLiteral = "r",
        allowed_exceptions: tuple[type[Exception], ...] = ALLOWED_EXCEPTIONS,
    ) -> RemoteStore:
        """
        Create a RemoteStore from an upath object.

        Parameters
        ----------
        upath : UPath
            The upath to the root of the store.
        mode : str, optional
            The mode of the store. Defaults to "r".
        allowed_exceptions : tuple, optional
            The exceptions that are allowed to be raised when accessing the
            store. Defaults to ALLOWED_EXCEPTIONS.

        Returns
        -------
        RemoteStore
        """
        return cls(
            fs=upath.fs,
            path=upath.path.rstrip("/"),
            mode=mode,
            allowed_exceptions=allowed_exceptions,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        storage_options: dict[str, Any] | None = None,
        mode: AccessModeLiteral = "r",
        allowed_exceptions: tuple[type[Exception], ...] = ALLOWED_EXCEPTIONS,
    ) -> RemoteStore:
        """
        Create a RemoteStore from a URL.

        Parameters
        ----------
        url : str
            The URL to the root of the store.
        storage_options : dict, optional
            The options to pass to fsspec when creating the filesystem.
        mode : str, optional
            The mode of the store. Defaults to "r".
        allowed_exceptions : tuple, optional
            The exceptions that are allowed to be raised when accessing the
            store. Defaults to ALLOWED_EXCEPTIONS.

        Returns
        -------
        RemoteStore
        """
        fs, path = fsspec.url_to_fs(url, **storage_options)
        return cls(fs=fs, path=path, mode=mode, allowed_exceptions=allowed_exceptions)

    async def clear(self) -> None:
        # docstring inherited
        try:
            for subpath in await self.fs._find(self.path, withdirs=True):
                if subpath != self.path:
                    await self.fs._rm(subpath, recursive=True)
        except FileNotFoundError:
            pass

    async def empty(self) -> bool:
        # docstring inherited

        # TODO: it would be nice if we didn't have to list all keys here
        # it should be possible to stop after the first key is discovered
        try:
            return not await self.fs._ls(self.path)
        except FileNotFoundError:
            return True

    def with_mode(self, mode: AccessModeLiteral) -> Self:
        # docstring inherited
        return type(self)(
            fs=self.fs,
            mode=mode,
            path=self.path,
            allowed_exceptions=self.allowed_exceptions,
        )

    def __repr__(self) -> str:
        return f"<RemoteStore({type(self.fs).__name__}, {self.path})>"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.path == other.path
            and self.mode == other.mode
            and self.fs == other.fs
        )

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRangeRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        if not self._is_open:
            await self._open()
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
                    self.fs._cat_file(path, start=byte_range[0], end=end)
                    if byte_range
                    else self.fs._cat_file(path)
                )
            )

        except self.allowed_exceptions:
            return None
        except OSError as e:
            if "not satisfiable" in str(e):
                # this is an s3-specific condition we probably don't want to leak
                return prototype.buffer.from_bytes(b"")
            raise
        else:
            return value

    async def set(
        self,
        key: str,
        value: Buffer,
        byte_range: tuple[int, int] | None = None,
    ) -> None:
        # docstring inherited
        if not self._is_open:
            await self._open()
        self._check_writable()
        path = _dereference_path(self.path, key)
        # write data
        if byte_range:
            raise NotImplementedError
        await self.fs._pipe_file(path, value.to_bytes())

    async def delete(self, key: str) -> None:
        # docstring inherited
        self._check_writable()
        path = _dereference_path(self.path, key)
        try:
            await self.fs._rm(path)
        except FileNotFoundError:
            pass
        except self.allowed_exceptions:
            pass

    async def exists(self, key: str) -> bool:
        # docstring inherited
        path = _dereference_path(self.path, key)
        exists: bool = await self.fs._exists(path)
        return exists

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRangeRequest]],
    ) -> list[Buffer | None]:
        # docstring inherited
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
        res = await self.fs._cat_ranges(list(paths), starts, stops, on_error="return")
        # the following is an s3-specific condition we probably don't want to leak
        res = [b"" if (isinstance(r, OSError) and "not satisfiable" in str(r)) else r for r in res]
        for r in res:
            if isinstance(r, Exception) and not isinstance(r, self.allowed_exceptions):
                raise r

        return [None if isinstance(r, Exception) else prototype.buffer.from_bytes(r) for r in res]

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, BytesLike]]
    ) -> None:
        # docstring inherited
        raise NotImplementedError

    async def list(self) -> AsyncGenerator[str, None]:
        # docstring inherited
        allfiles = await self.fs._find(self.path, detail=False, withdirs=False)
        for onefile in (a.replace(self.path + "/", "") for a in allfiles):
            yield onefile

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        prefix = f"{self.path}/{prefix.rstrip('/')}"
        try:
            allfiles = await self.fs._ls(prefix, detail=False)
        except FileNotFoundError:
            return
        for onefile in (a.replace(prefix + "/", "") for a in allfiles):
            yield onefile.removeprefix(self.path).removeprefix("/")

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        find_str = f"{self.path}/{prefix}"
        for onefile in await self.fs._find(find_str, detail=False, maxdepth=None, withdirs=False):
            yield onefile.removeprefix(find_str)
