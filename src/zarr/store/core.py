from __future__ import annotations

from pathlib import Path
from typing import Any

from zarr.abc.store import Store
from zarr.buffer import Buffer
from zarr.store.local import LocalStore


def _dereference_path(root: str, path: str) -> str:
    assert isinstance(root, str)
    assert isinstance(path, str)
    root = root.rstrip("/")
    path = f"{root}/{path}" if root != "" else path
    path = path.rstrip("/")
    return path


class StorePath:
    store: Store
    path: str

    def __init__(self, store: Store, path: str | None = None):
        self.store = store
        self.path = path or ""

    async def get(self, byte_range: tuple[int, int | None] | None = None) -> Buffer | None:
        return await self.store.get(self.path, byte_range)

    async def set(self, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        if byte_range is not None:
            raise NotImplementedError("Store.set does not have partial writes yet")
        await self.store.set(self.path, value)

    async def delete(self) -> None:
        await self.store.delete(self.path)

    async def exists(self) -> bool:
        return await self.store.exists(self.path)

    def __truediv__(self, other: str) -> StorePath:
        return self.__class__(self.store, _dereference_path(self.path, other))

    def __str__(self) -> str:
        return _dereference_path(str(self.store), self.path)

    def __repr__(self) -> str:
        return f"StorePath({self.store.__class__.__name__}, {str(self)!r})"

    def __eq__(self, other: Any) -> bool:
        try:
            if self.store == other.store and self.path == other.path:
                return True
        except Exception:
            pass
        return False


StoreLike = Store | StorePath | Path | str


def make_store_path(store_like: StoreLike) -> StorePath:
    if isinstance(store_like, StorePath):
        return store_like
    elif isinstance(store_like, Store):
        return StorePath(store_like)
    elif isinstance(store_like, str):
        return StorePath(LocalStore(Path(store_like)))
    raise TypeError


def _normalize_interval_index(
    data: Buffer, interval: None | tuple[int | None, int | None]
) -> tuple[int, int]:
    """
    Convert an implicit interval into an explicit start and length
    """
    if interval is None:
        start = 0
        length = len(data)
    else:
        maybe_start, maybe_len = interval
        if maybe_start is None:
            start = 0
        else:
            start = maybe_start

        if maybe_len is None:
            length = len(data) - start
        else:
            length = maybe_len

    return (start, length)
