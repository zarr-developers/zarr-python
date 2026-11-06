import importlib
import sys
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

from zarr.errors import ZarrDeprecationWarning
from zarr.storage._common import StoreLike, StorePath
from zarr.storage._fsspec import FsspecStore
from zarr.storage._local import LocalStore
from zarr.storage._logging import LoggingStore
from zarr.storage._memory import GpuMemoryStore, ManagedMemoryStore, MemoryStore
from zarr.storage._obstore import ObjectStore
from zarr.storage._url_pipeline import PipelineContext, PipelineSegment, URLPipeline
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
    "PipelineContext",
    "PipelineSegment",
    "StoreLike",
    "StorePath",
    "URLPipeline",
    "WrapperStore",
    "ZipStore",
]


class VerboseModule(ModuleType):
    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "default_compressor":
            warnings.warn(
                "setting zarr.storage.default_compressor is deprecated, use "
                "the `compressors` argument to zarr.create_array() instead "
                "e.g. zarr.create_array(..., zarr_format=2, compressors=[numcodecs.Zstd()])",
                ZarrDeprecationWarning,
                stacklevel=1,
            )
        else:
            super().__setattr__(attr, value)

    def __reduce__(self) -> tuple[Callable[[str], ModuleType], tuple[str]]:
        # Serializers such as cloudpickle dispatch on the exact type, so they have no
        # reduce for this subclass. Pickle the module by name instead.
        return importlib.import_module, (self.__name__,)


sys.modules[__name__].__class__ = VerboseModule
