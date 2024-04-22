from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Union, Optional, List, Tuple

from zarr.v3.abc.store import Store
from zarr.v3.common import BytesLike, concurrent_map, to_thread


def _get(path: Path, byte_range: Optional[Tuple[int, Optional[int]]] = None) -> bytes:
    if byte_range is not None:
        start = byte_range[0]
        end = (start + byte_range[1]) if byte_range[1] is not None else None
    else:
        return path.read_bytes()
    with path.open("rb") as f:
        size = f.seek(0, io.SEEK_END)
        if start is not None:
            if start >= 0:
                f.seek(start)
            else:
                f.seek(max(0, size + start))
        if end is not None:
            if end < 0:
                end = size + end
            return f.read(end - f.tell())
        return f.read()


def _put(
    path: Path,
    value: BytesLike,
    start: Optional[int] = None,
    auto_mkdir: bool = True,
):
    if auto_mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    if start is not None:
        with path.open("r+b") as f:
            f.seek(start)
            f.write(value)
    else:
        return path.write_bytes(value)


class LocalStore(Store):
    supports_writes: bool = True
    supports_partial_writes: bool = True
    supports_listing: bool = True

    root: Path
    auto_mkdir: bool

    def __init__(self, root: Union[Path, str], auto_mkdir: bool = True):
        if isinstance(root, str):
            root = Path(root)
        assert isinstance(root, Path)

        self.root = root
        self.auto_mkdir = auto_mkdir

    def __str__(self) -> str:
        return f"file://{self.root}"

    def __repr__(self) -> str:
        return f"LocalStore({repr(str(self))})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.root == other.root

    async def get(
        self, key: str, byte_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> Optional[bytes]:
        assert isinstance(key, str)
        path = self.root / key

        try:
            return await to_thread(_get, path, byte_range)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    async def get_partial_values(
        self, key_ranges: List[Tuple[str, Tuple[int, int]]]
    ) -> List[bytes]:
        args = []
        for key, byte_range in key_ranges:
            assert isinstance(key, str)
            path = self.root / key
            if byte_range is not None:
                args.append((_get, path, byte_range[0], byte_range[1]))
            else:
                args.append((_get, path))
        return await concurrent_map(args, to_thread, limit=None)  # TODO: fix limit

    async def set(self, key: str, value: BytesLike) -> None:
        assert isinstance(key, str)
        path = self.root / key
        await to_thread(_put, path, value)

    async def set_partial_values(self, key_start_values: List[Tuple[str, int, bytes]]) -> None:
        args = []
        for key, start, value in key_start_values:
            assert isinstance(key, str)
            path = self.root / key
            if start is not None:
                args.append((_put, path, value, start))
            else:
                args.append((_put, path, value))
        await concurrent_map(args, to_thread, limit=None)  # TODO: fix limit

    async def delete(self, key: str) -> None:
        path = self.root / key
        if path.is_dir():  # TODO: support deleting directories? shutil.rmtree?
            shutil.rmtree(path)
        else:
            await to_thread(path.unlink, True)  # Q: we may want to raise if path is missing

    async def exists(self, key: str) -> bool:
        path = self.root / key
        return await to_thread(path.is_file)

    async def list(self) -> List[str]:
        """Retrieve all keys in the store.

        Returns
        -------
        list[str]
        """

        # Q: do we want to return strings or Paths?
        def _list(root: Path) -> List[str]:
            files = [str(p) for p in root.rglob("") if p.is_file()]
            return files

        return await to_thread(_list, self.root)

    async def list_prefix(self, prefix: str) -> List[str]:
        """Retrieve all keys in the store with a given prefix.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        list[str]
        """

        def _list_prefix(root: Path, prefix: str) -> List[str]:
            files = [str(p) for p in (root / prefix).rglob("*") if p.is_file()]
            return files

        return await to_thread(_list_prefix, self.root, prefix)

    async def list_dir(self, prefix: str) -> List[str]:
        """
        Retrieve all keys and prefixes with a given prefix and which do not contain the character
        “/” after the given prefix.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        list[str]
        """

        def _list_dir(root: Path, prefix: str) -> List[str]:
            base = root / prefix
            to_strip = str(base) + "/"
            try:
                return [str(key).replace(to_strip, "") for key in base.iterdir()]
            except (FileNotFoundError, NotADirectoryError):
                return []

        return await to_thread(_list_dir, self.root, prefix)
