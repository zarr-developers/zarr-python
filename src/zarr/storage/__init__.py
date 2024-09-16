from zarr.storage.common import StoreLike, StorePath, make_store_path
from zarr.storage.local import LocalStore
from zarr.storage.memory import MemoryStore
from zarr.storage.remote import RemoteStore
from zarr.storage.zip import ZipStore

# alias for backwards compatibility
FSStore = RemoteStore
DirectoryStore = LocalStore

__all__ = [
    "DirectoryStore",
    "FSStore",
    "StorePath",
    "StoreLike",
    "make_store_path",
    "RemoteStore",
    "LocalStore",
    "MemoryStore",
    "ZipStore",
]
