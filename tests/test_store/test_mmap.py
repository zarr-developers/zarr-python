from __future__ import annotations

from typing import TYPE_CHECKING

import mmap
import pytest

import zarr
from zarr.core.buffer import Buffer, cpu
from zarr.storage import LocalStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    import pathlib


class MemoryMappedDirectoryStore(LocalStore):
    def _fromfile(self, fn):
        with open(fn, "rb") as fh:
            return memoryview(mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ))


class TestMemoryMappedDirectoryStore(StoreTests[MemoryMappedDirectoryStore, cpu.Buffer]):
    store_cls = MemoryMappedDirectoryStore
    buffer_cls = cpu.Buffer

    async def get(self, store: MemoryMappedDirectoryStore, key: str) -> Buffer:
        return self.buffer_cls.from_bytes((store.root / key).read_bytes())

    async def set(self, store: MemoryMappedDirectoryStore, key: str, value: Buffer) -> None:
        parent = (store.root / key).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        (store.root / key).write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, tmpdir) -> dict[str, str]:
        return {"root": str(tmpdir)}

    def test_store_repr(self, store: MemoryMappedDirectoryStore) -> None:
        assert str(store) == f"file://{store.root.as_posix()}"

    def test_store_supports_writes(self, store: MemoryMappedDirectoryStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: MemoryMappedDirectoryStore) -> None:
        assert store.supports_partial_writes

    def test_store_supports_listing(self, store: MemoryMappedDirectoryStore) -> None:
        assert store.supports_listing

    async def test_empty_with_empty_subdir(self, store: MemoryMappedDirectoryStore) -> None:
        assert await store.is_empty("")
        (store.root / "foo/bar").mkdir(parents=True)
        assert await store.is_empty("")

    def test_creates_new_directory(self, tmp_path: pathlib.Path):
        target = tmp_path.joinpath("a", "b", "c")
        assert not target.exists()

        store = self.store_cls(root=target)
        zarr.group(store=store)

    async def test_mmap_slice_reads(self, store: MemoryMappedDirectoryStore) -> None:
        """Test reading slices with memory mapping"""
        # Create array with large chunks
        z = zarr.create_array(store=store, shape=(2000, 2000), chunks=(1000, 1000), 
                            dtype='float64')
        # Write test data
        data = zarr.full(shape=(2000, 2000), chunks=(1000, 1000), fill_value=42.0, 
                        dtype='float64')
        z[:] = data[:]
        
        # Test reading various slices
        slices = [
            # Within single chunk
            (slice(100, 200), slice(100, 200)),
            # Across chunk boundaries
            (slice(900, 1100), slice(900, 1100)),
            # Full chunk
            (slice(0, 1000), slice(0, 1000))
        ]
        
        for test_slice in slices:
            assert (z[test_slice] == data[test_slice]).all()
