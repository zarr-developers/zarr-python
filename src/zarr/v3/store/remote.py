from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, Union, List

import fsspec
from zarr.v3.abc.store import Store
from zarr.v3.store.core import _dereference_path
from zarr.v3.common import BytesLike


if TYPE_CHECKING:
    from upath import UPath
    from fsspec.asyn import AsyncFileSystem


class FsspecStore(Store):
    supports_writes: bool = True
    supports_partial_writes: bool = False
    supports_listing: bool = True

    _fs: AsyncFileSystem
    exceptions = Tuple[Type[Exception], ...]

    def __init__(
        self,
        url: Union[UPath, str],
        allowed_exceptions: Tuple[Type[Exception], ...] = (
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
            assert len(storage_options) == 0, (
                "If constructed with a UPath object, no additional "
                + "storage_options are allowed."
            )
            self.path = url.path
            self._fs = url._fs
        else:
            raise ValueError("URL not understood, %s", url)
        # dear mypy: these have the same type annotations
        self.exceptions = allowed_exceptions  # type: ignore
        # test instantiate file system
        assert self._fs.async_impl, "FileSystem needs to support async operations."

    def __str__(self) -> str:
        return f"Remote fsspec store: {self.path}"

    def __repr__(self) -> str:
        return f"<FsspecStore({self.path})>"

    async def get(
        self, key: str, byte_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> Optional[BytesLike]:
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
        self, key: str, value: BytesLike, byte_range: Optional[Tuple[int, int]] = None
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

    async def list(self) -> List[str]:
        allfiles = await self._fs._find(self.path, detail=False, withdirs=False)
        return [a.replace(self.path + "/", "") for a in allfiles]

    async def list_dir(self, prefix: str) -> List[str]:
        prefix = prefix.rstrip("/")
        allfiles = await self._fs._ls(prefix, detail=False)
        return [a.replace(prefix + "/", "") for a in allfiles]

    async def list_prefix(self, prefix: str) -> List[str]:
        return await self._fs._ls(prefix, detail=False)

    async def set_partial_values(self, key_start_values: List[Tuple[str, int, bytes]]) -> None:
        raise NotImplementedError

    async def get_partial_values(
        self, key_ranges: List[Tuple[str, Tuple[int, int]]]
    ) -> List[bytes]:
        paths, starts, stops = [
            (_dereference_path(self.path, k[0]), k[1][0], k[1][1]) for k in key_ranges
        ]
        # TODO: expectations for exceptions or missing keys?
        return await self._fs._cat_ranges(paths, starts, stops, on_error="return")
