"""Array engine resolution and the built-in engine.

The protocols themselves live in `zarr.abc.engine`.
"""

from zarr.core.engine._default import (
    DefaultArrayEngine,
    DefaultAsyncArrayEngine,
    DefaultAsyncHierarchyEngine,
    DefaultHierarchyEngine,
    adopt_codec_pipeline,
)
from zarr.core.engine._resolve import (
    EngineName,
    classify_engine_arg,
    list_engines,
    resolve_async_engine,
    resolve_sync_engine,
    route_sync_engine_arg,
)

__all__ = [
    "DefaultArrayEngine",
    "DefaultAsyncArrayEngine",
    "DefaultAsyncHierarchyEngine",
    "DefaultHierarchyEngine",
    "EngineName",
    "adopt_codec_pipeline",
    "classify_engine_arg",
    "list_engines",
    "resolve_async_engine",
    "resolve_sync_engine",
    "route_sync_engine_arg",
]
