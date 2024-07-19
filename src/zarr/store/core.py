from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zarr.abc.store import Store
from zarr.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.common import ZARR_JSON, ZARRAY_JSON, ZGROUP_JSON, OpenMode, ZarrFormat
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


async def contains_array(store_path: StorePath, zarr_format: ZarrFormat) -> bool:
    """
    Check if an array exists at a given StorePath.

    Parameters
    ----------

    store_path: StorePath
        The StorePath to check for an existing group.
    zarr_format:
        The zarr format to check for.

    Returns
    -------

    bool
        True if the StorePath contains a group, False otherwise

    """
    if zarr_format == 3:
        extant_meta_bytes = await (store_path / ZARR_JSON).get()
        if extant_meta_bytes is None:
            return False
        else:
            try:
                extant_meta_json = json.loads(extant_meta_bytes.to_bytes())
                # we avoid constructing a full metadata document here in the name of speed.
                if extant_meta_json["node_type"] == "array":
                    return True
            except (ValueError, KeyError):
                return False
    elif zarr_format == 2:
        return await (store_path / ZARRAY_JSON).exists()
    msg = f"Invalid zarr_format provided. Got {zarr_format}, expected 2 or 3"
    raise ValueError(msg)


async def contains_group(store_path: StorePath, zarr_format: ZarrFormat) -> bool:
    """
    Check if a group exists at a given StorePath.

    Parameters
    ----------

    store_path: StorePath
        The StorePath to check for an existing group.
    zarr_format:
        The zarr format to check for.

    Returns
    -------

    bool
        True if the StorePath contains a group, False otherwise

    """
    if zarr_format == 3:
        return await (store_path / ZARR_JSON).exists()
    elif zarr_format == 2:
        return await (store_path / ZGROUP_JSON).exists()
    msg = f"Invalid zarr_format provided. Got {zarr_format}, expected 2 or 3"  # type: ignore[unreachable]
    raise ValueError(msg)
