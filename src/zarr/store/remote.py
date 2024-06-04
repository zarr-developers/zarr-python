from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr.abc.store import Store
from zarr.buffer import Buffer, BufferPrototype
from zarr.common import OpenMode
from zarr.store.core import _dereference_path

if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem
    from upath import UPath


class RemoteStore(Store):
    supports_writes: bool = True
    supports_partial_writes: bool = False
    supports_listing: bool = True

    root: UPath

    def __init__(
        self, url: UPath | str, *, mode: OpenMode = "r", **storage_options: dict[str, Any]
    ):
        import fsspec
        from upath import UPath

        super().__init__(mode=mode)

        if isinstance(url, str):
            self.root = UPath(url, **storage_options)
        else:
            assert (
                len(storage_options) == 0
            ), "If constructed with a UPath object, no additional storage_options are allowed."
            self.root = url.rstrip("/")

        # test instantiate file system
        fs, _ = fsspec.core.url_to_fs(str(self.root), asynchronous=True, **self.root._kwargs)
        assert fs.__class__.async_impl, "FileSystem needs to support async operations."

    def __str__(self) -> str:
        return str(self.root)

    def __repr__(self) -> str:
        return f"RemoteStore({str(self)!r})"

    def _make_fs(self) -> tuple[AsyncFileSystem, str]:
        import fsspec

        storage_options = self.root._kwargs.copy()
        storage_options.pop("_url", None)
        fs, root = fsspec.core.url_to_fs(str(self.root), asynchronous=True, **self.root._kwargs)
        assert fs.__class__.async_impl, "FileSystem needs to support async operations."
        return fs, root

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        assert isinstance(key, str)
        fs, root = self._make_fs()
        path = _dereference_path(root, key)

        try:
            value: Buffer | None = await (
                fs._cat_file(path, start=byte_range[0], end=byte_range[1])
                if byte_range
                else fs._cat_file(path)
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

        return value

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        self._check_writable()
        assert isinstance(key, str)
        fs, root = self._make_fs()
        path = _dereference_path(root, key)

        # write data
        if byte_range:
            with fs._open(path, "r+b") as f:
                f.seek(byte_range[0])
                f.write(value)
        else:
            await fs._pipe_file(path, value)

    async def delete(self, key: str) -> None:
        self._check_writable()
        fs, root = self._make_fs()
        path = _dereference_path(root, key)
        if await fs._exists(path):
            await fs._rm(path)

    async def exists(self, key: str) -> bool:
        fs, root = self._make_fs()
        path = _dereference_path(root, key)
        exists: bool = await fs._exists(path)
        return exists
