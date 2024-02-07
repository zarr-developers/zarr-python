from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from zarr.v3.abc.store import Store
from zarr.v3.common import BytesLike


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

    def __init__(self, store: Store, path: Optional[str] = None):
        self.store = store
        self.path = path or ""

    @classmethod
    def from_path(cls, pth: Path) -> StorePath:
        return cls(Store.from_path(pth))

    async def get(
        self, byte_range: Optional[tuple[int, Optional[int]]] = None
    ) -> Optional[BytesLike]:
        return await self.store.get(self.path, byte_range)

    async def set(self, value: BytesLike, byte_range: Optional[tuple[int, int]] = None) -> None:
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

    def __eq__(self, other: object) -> bool:
        try:
            if self.store == other.store and self.path == other.path:
                return True
        except Exception:
            pass
        return False


StoreLike = Union[Store, StorePath, Path, str]


def make_store_path(store_like: StoreLike) -> StorePath:
    if isinstance(store_like, StorePath):
        return store_like
    elif isinstance(store_like, Store):
        return StorePath(store_like)
    # elif isinstance(store_like, Path):
    #     return StorePath(Store.from_path(store_like))
    elif isinstance(store_like, str):
        try:
            from upath import UPath

            return StorePath(Store.from_path(UPath(store_like)))
        except ImportError as e:
            raise e
            # return StorePath(LocalStore(Path(store_like)))
    raise TypeError
