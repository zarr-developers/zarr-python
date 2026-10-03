import importlib
import sys
import warnings
from types import ModuleType
from typing import Any

from zarr.errors import ZarrDeprecationWarning
from zarr.storage._common import StoreLike, StorePath
from zarr.storage._fsspec import FsspecStore
from zarr.storage._local import LocalStore
from zarr.storage._logging import LoggingStore
from zarr.storage._memory import GpuMemoryStore, ManagedMemoryStore, MemoryStore
from zarr.storage._obstore import ObjectStore
from zarr.storage._wrapper import WrapperStore
from zarr.storage._zip import ZipStore

__all__ = [
    "FsspecStore",
    "GpuMemoryStore",
    "LocalStore",
    "LoggingStore",
    "ManagedMemoryStore",
    "MemoryStore",
    "ObjectStore",
    "StoreLike",
    "StorePath",
    "WrapperStore",
    "ZipStore",
]


class VerboseModule(ModuleType):
    def __reduce__(self) -> tuple[Any, ...]:
        # Pickle by reference, as for a plain module, instead of trying to serialize
        # this module's instance state. Without this, pickling fails with
        # "cannot pickle 'VerboseModule' object".
        return importlib.import_module, (self.__name__,)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "default_compressor":
            warnings.warn(
                "setting zarr.storage.default_compressor is deprecated, use "
                "zarr.config to configure array.v2_default_compressor "
                "e.g. config.set({'codecs.zstd':'numcodecs.Zstd', 'array.v2_default_compressor.numeric': 'zstd'})",
                ZarrDeprecationWarning,
                stacklevel=1,
            )
        else:
            super().__setattr__(attr, value)


sys.modules[__name__].__class__ = VerboseModule
