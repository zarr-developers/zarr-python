# flake8: noqa
from zarr.v2.codecs import *
from zarr.v2.convenience import (
    consolidate_metadata,
    copy,
    copy_all,
    copy_store,
    load,
    open,
    open_consolidated,
    save,
    save_array,
    save_group,
    tree,
)
from zarr.v2.core import Array
from zarr.v2.creation import (
    array,
    create,
    empty,
    empty_like,
    full,
    full_like,
    ones,
    ones_like,
    open_array,
    open_like,
    zeros,
    zeros_like,
)
from zarr.v2.errors import CopyError, MetadataError
from zarr.v2.hierarchy import Group, group, open_group
from zarr.v2.n5 import N5Store, N5FSStore
from zarr.v2.storage import (
    ABSStore,
    DBMStore,
    DictStore,
    DirectoryStore,
    KVStore,
    LMDBStore,
    LRUStoreCache,
    MemoryStore,
    MongoDBStore,
    NestedDirectoryStore,
    RedisStore,
    SQLiteStore,
    TempStore,
    ZipStore,
)
from zarr.v2.sync import ProcessSynchronizer, ThreadSynchronizer
from zarr._version import version as __version__

# in case setuptools scm screw up and find version to be 0.0.0
assert not __version__.startswith("0.0.0")
