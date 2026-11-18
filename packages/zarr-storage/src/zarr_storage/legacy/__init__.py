import sys
import warnings
from types import ModuleType
from typing import Any

from zarr.errors import ZarrDeprecationWarning

from zarr_storage.legacy._abc import (
    ByteGetter as ByteGetter,
)
from zarr_storage.legacy._abc import (
    ByteRequest as ByteRequest,
)
from zarr_storage.legacy._abc import (
    ByteSetter as ByteSetter,
)
from zarr_storage.legacy._abc import (
    OffsetByteRequest as OffsetByteRequest,
)
from zarr_storage.legacy._abc import (
    RangeByteRequest as RangeByteRequest,
)
from zarr_storage.legacy._abc import (
    Store as Store,
)
from zarr_storage.legacy._abc import (
    SuffixByteRequest as SuffixByteRequest,
)
from zarr_storage.legacy._abc import (
    SupportsDeleteSync as SupportsDeleteSync,
)
from zarr_storage.legacy._abc import (
    SupportsGetSync as SupportsGetSync,
)
from zarr_storage.legacy._abc import (
    SupportsSetSync as SupportsSetSync,
)
from zarr_storage.legacy._abc import (
    SupportsSyncStore as SupportsSyncStore,
)
from zarr_storage.legacy._abc import (
    SyncByteGetter as SyncByteGetter,
)
from zarr_storage.legacy._abc import (
    SyncByteSetter as SyncByteSetter,
)
from zarr_storage.legacy._abc import (
    _store_supports_sync_io as _store_supports_sync_io,
)
from zarr_storage.legacy._abc import (
    set_or_delete as set_or_delete,
)
from zarr_storage.legacy._common import StoreLike, StorePath
from zarr_storage.legacy._fsspec import FsspecStore
from zarr_storage.legacy._latency import LatencyStore
from zarr_storage.legacy._local import LocalStore
from zarr_storage.legacy._logging import LoggingStore
from zarr_storage.legacy._memory import GpuMemoryStore, ManagedMemoryStore, MemoryStore
from zarr_storage.legacy._obstore import ObjectStore
from zarr_storage.legacy._wrapper import WrapperStore
from zarr_storage.legacy._zip import ZipStore

__all__ = [
    "ByteGetter",
    "ByteRequest",
    "ByteSetter",
    "FsspecStore",
    "GpuMemoryStore",
    "LatencyStore",
    "LocalStore",
    "LoggingStore",
    "ManagedMemoryStore",
    "MemoryStore",
    "ObjectStore",
    "OffsetByteRequest",
    "RangeByteRequest",
    "Store",
    "StoreLike",
    "StorePath",
    "SuffixByteRequest",
    "SupportsDeleteSync",
    "SupportsGetSync",
    "SupportsSetSync",
    "SupportsSyncStore",
    "SyncByteGetter",
    "SyncByteSetter",
    "WrapperStore",
    "ZipStore",
    "set_or_delete",
]


class VerboseModule(ModuleType):
    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "default_compressor":
            warnings.warn(
                "setting zarr_storage.legacy.default_compressor is deprecated, use "
                "zarr.config to configure array.v2_default_compressor "
                "e.g. config.set({'codecs.zstd':'numcodecs.Zstd', 'array.v2_default_compressor.numeric': 'zstd'})",
                ZarrDeprecationWarning,
                stacklevel=1,
            )
        else:
            super().__setattr__(attr, value)


sys.modules[__name__].__class__ = VerboseModule
