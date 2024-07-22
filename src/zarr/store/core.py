from __future__ import annotations

from pathlib import Path
from typing import Any

from zarr.abc.store import Store
from zarr.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.common import OpenMode
from zarr.store.local import LocalStore
from zarr.store.memory import MemoryStore


def _dereference_path(root: str, path: str) -> str:
    assert isinstance(root, str)
    assert isinstance(path, str)
    root = root.rstrip("/")
    path = f"{root}/{path}" if root else path
    path = path.rstrip("/")
    return path


class StorePath:
    store: Store
    path: str

    def __init__(self, store: Store, path: str | None = None):
        self.store = store
        self.path = path or ""

    async def get(
        self,
        prototype: BufferPrototype = default_buffer_prototype,
        byte_range: tuple[int, int | None] | None = None,
    ) -> Buffer | None:
        return await self.store.get(self.path, prototype=prototype, byte_range=byte_range)

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


def make_store_path(store_like: StoreLike | None, *, mode: OpenMode | None = None) -> StorePath:
    if isinstance(store_like, StorePath):
        if mode is not None:
            assert mode == store_like.store.mode
        return store_like
    elif isinstance(store_like, Store):
        if mode is not None:
            assert mode == store_like.mode
        return StorePath(store_like)
    elif store_like is None:
        if mode is None:
            mode = "w"  # exception to the default mode = 'r'
        return StorePath(MemoryStore(mode=mode))
    elif isinstance(store_like, Path):
        return StorePath(LocalStore(store_like, mode=mode or "r"))
    elif isinstance(store_like, str):
        return StorePath(LocalStore(Path(store_like), mode=mode or "r"))
    raise TypeError
