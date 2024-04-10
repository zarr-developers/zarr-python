from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, List

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

    root: UPath

    def __init__(self, url: Union[UPath, str], **storage_options: Dict[str, Any]):
        from upath import UPath

        if isinstance(url, str):
            self.root = UPath(url, **storage_options)
        else:
            assert len(storage_options) == 0, (
                "If constructed with a UPath object, no additional "
                + "storage_options are allowed."
            )
            self.root = url
        self.path = self.root.path
        # test instantiate file system
        assert self.root.fs.async_impl, "FileSystem needs to support async operations."

    @property
    def _fs(self):
        return self.root.fs

    def __str__(self) -> str:
        return f"Remote fsspec store: {self.root}"

    def __repr__(self) -> str:
        return f"<FsspecStore({self})>"

    async def get(
        self, key: str, byte_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> Optional[BytesLike]:
        assert isinstance(key, str)
        path = _dereference_path(self.path, key)

        try:
            value = await (
                self._fs._cat_file(path, start=byte_range[0], end=byte_range[1])
                if byte_range
                else self._fs._cat_file(path)
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

        return value

    async def set(
        self, key: str, value: BytesLike, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        assert isinstance(key, str)
        path = _dereference_path(self.path, key)

        # write data
        if byte_range:
            with self._fs._open(path, "r+b") as f:
                f.seek(byte_range[0])
                f.write(value)
        else:
            await self._fs._pipe_file(path, value)

    async def delete(self, key: str) -> None:
        path = _dereference_path(self.path, key)
        if await self._fs._exists(path):
            await self._fs._rm(path)

    async def exists(self, key: str) -> bool:
        path = _dereference_path(self.path, key)
        return await self._fs._exists(path)

    async def get_partial_values(self, key_ranges: List[Tuple[str, Tuple[int, int]]]) -> List[bytes]:
        paths, starts, stops = [
            (_dereference_path(self.path, k[0]), k[1][0], k[1][1])
            for k in key_ranges
        ]
        return await self._cat_ranges(paths, starts, stops)
