import sys
import warnings
from types import ModuleType
from typing import Any

from zarr.storage.common import StoreLike, StorePath, make_store_path
from zarr.storage.fsspec import FsspecStore
from zarr.storage.local import LocalStore
from zarr.storage.logging import LoggingStore
from zarr.storage.memory import MemoryStore
from zarr.storage.wrapper import WrapperStore
from zarr.storage.zip import ZipStore

__all__ = [
    "FsspecStore",
    "LocalStore",
    "LoggingStore",
    "MemoryStore",
    "StoreLike",
    "StorePath",
    "WrapperStore",
    "ZipStore",
    "make_store_path",
]


class VerboseModule(ModuleType):
    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "default_compressor":
            warnings.warn(
                "setting zarr.storage.default_compressor is deprecated, use "
                "zarr.config to configure array.v2_default_compressor "
                "e.g. config.set({'codecs.zstd':'numcodecs.Zstd', 'array.v2_default_compressor.numeric': 'zstd'})",
                DeprecationWarning,
                stacklevel=1,
            )
        else:
            super().__setattr__(attr, value)


sys.modules[__name__].__class__ = VerboseModule
