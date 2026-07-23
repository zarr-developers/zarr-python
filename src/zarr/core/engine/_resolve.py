"""Resolve an `engine=` argument to a bound array engine."""

from __future__ import annotations

import contextlib
import inspect
import weakref
from typing import TYPE_CHECKING, Literal, get_args

from zarr.core.engine._default import (
    DefaultAsyncHierarchyEngine,
    DefaultHierarchyEngine,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr.abc.engine import (
        ArrayEngine,
        AsyncArrayEngine,
        AsyncHierarchyEngine,
        HierarchyEngine,
    )
    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfig
    from zarr.core.metadata import ArrayMetadata

__all__ = [
    "EngineName",
    "classify_engine_arg",
    "list_engines",
    "resolve_async_engine",
    "resolve_sync_engine",
    "route_sync_engine_arg",
]

EngineName = Literal["default", "zarrista"]


def list_engines() -> list[str]:
    """Return the sorted names of the built-in array engines.

    Any of these names can be passed as the `engine=` argument to
    `zarr.open_array`, `zarr.create_array`, and the other array entry points to
    select the data-path engine backing the array.

    `"zarrista"` additionally requires the optional `zarrista` package to be
    installed; without it, resolving that engine raises an `ImportError`.
    """
    # `EngineName` is the single source of truth for known engine names --
    # `_hierarchy_factory` dispatches on exactly these literals.
    return sorted(get_args(EngineName))


def classify_engine_arg(engine: object) -> Literal["name", "sync", "async"]:
    """Classify an `engine=` argument as a name, a sync instance, or an async instance.

    `None` and `str` values classify as `"name"` -- valid wherever an `engine=`
    argument is accepted. Any other value must implement `read_selection`: a
    coroutine function classifies as `"async"` (an `AsyncArrayEngine`), anything
    else as `"sync"` (an `ArrayEngine`). Objects with no `read_selection` at all
    raise `TypeError` naming the two protocols.
    """
    if engine is None or isinstance(engine, str):
        return "name"
    read_selection = getattr(engine, "read_selection", None)
    if read_selection is None:
        raise TypeError(
            f"{engine!r} does not implement the ArrayEngine or AsyncArrayEngine protocol "
            "(missing a `read_selection` method)"
        )
    return "async" if inspect.iscoroutinefunction(read_selection) else "sync"


def route_sync_engine_arg(
    engine: ArrayEngine | AsyncArrayEngine | EngineName | None,
) -> tuple[AsyncArrayEngine | EngineName | None, ArrayEngine | EngineName | None]:
    """Route a sync-entry-point `engine=` argument to its two consumers.

    The public sync entry points (`zarr.create_array`, `zarr.open_array`, ...)
    accept the same broad `engine` type as their async counterparts so their
    signatures and docstrings match; this function is where the sync-specific
    rules actually get enforced.

    Returns `(engine_for_async_array, engine_for_array)`. A name (or `None`) is
    valid for both layers and is returned unchanged in both slots, so sync and
    async access to the same object use the same engine family. A sync
    `ArrayEngine` instance is returned only in the second slot -- the wrapped
    `AsyncArray` keeps its default engine. An `AsyncArrayEngine` instance cannot
    serve a sync entry point and raises `TypeError`.
    """
    kind = classify_engine_arg(engine)
    if kind == "async":
        # Fail fast at the API boundary rather than lazily when `Array`
        # resolves its engine; message kept identical to
        # `resolve_sync_engine`'s so the error looks the same regardless of
        # where the wrong-kind instance was actually caught.
        raise TypeError(
            "Array requires a synchronous engine (ArrayEngine); got an "
            f"async engine of type `{type(engine).__name__}`"
        )
    if kind == "name":
        return engine, engine  # type: ignore[return-value]
    # kind == "sync": the instance only serves the sync Array; the inner
    # AsyncArray keeps its default engine.
    return None, engine  # type: ignore[return-value]


# (name, kind, id(store)) -> hierarchy engine; entries evicted automatically once
# nothing keeps the hierarchy engine itself alive (see `_keepalive` below).
#
# Note: a hierarchy engine holds its `store` strongly (it must, to do I/O), so a
# plain dict keyed by a `weakref.ref`/`weakref.finalize` on the *store* cannot
# work here -- as long as the hierarchy sits in such a cache it keeps the store
# alive, so the store's refcount never reaches zero and the finalizer never
# fires. Using a `WeakValueDictionary` for the hierarchy itself sidesteps that:
# the cache entry disappears as soon as nothing external holds the hierarchy.
# `_keepalive` ties the hierarchy's lifetime to the array engines minted from
# it, so engines resolved for the same store while at least one is still alive
# share a hierarchy; once all of them (and the store) are unreferenced, both
# the hierarchy and the cache entry are collected.
_hierarchy_cache: weakref.WeakValueDictionary[tuple[str, str, int], object] = (
    weakref.WeakValueDictionary()
)


def _cached_hierarchy(
    name: str, kind: str, store: Store, factory: Callable[[Store], object]
) -> object:
    key = (name, kind, id(store))
    hierarchy = _hierarchy_cache.get(key)
    if hierarchy is None:
        hierarchy = factory(store)
        _hierarchy_cache[key] = hierarchy
    return hierarchy


def _keepalive(engine: object, hierarchy: object) -> object:
    """Attach `hierarchy` to `engine` so the hierarchy (and thus the cache entry
    tracking it) stays alive for as long as `engine` does. Best-effort: engines
    that forbid arbitrary attributes (e.g. via `__slots__`) simply won't share
    a cached hierarchy across calls.
    """
    with contextlib.suppress(AttributeError):
        engine._resolve_hierarchy_keepalive = hierarchy  # type: ignore[attr-defined]
    return engine


def _hierarchy_factory(name: str, *, sync: bool) -> Callable[[Store], object]:
    if name == "default":
        return DefaultHierarchyEngine if sync else DefaultAsyncHierarchyEngine
    if name == "zarrista":
        try:
            from zarr.zarrista import (
                ZarristaAsyncHierarchyEngine,
                ZarristaHierarchyEngine,
            )
        except ImportError as e:
            raise ImportError(
                "engine='zarrista' requires the `zarrista` package; "
                "install zarr with the `zarrista` extra"
            ) from e
        return ZarristaHierarchyEngine if sync else ZarristaAsyncHierarchyEngine
    raise ValueError(f"unknown engine name {name!r}; expected 'default' or 'zarrista'")


def resolve_async_engine(
    engine: AsyncArrayEngine | EngineName | None,
    *,
    store: Store,
    path: str,
    metadata: ArrayMetadata,
    config: ArrayConfig | None = None,
) -> AsyncArrayEngine:
    """Resolve an `engine=` argument to a bound `AsyncArrayEngine`.

    `None` and `"default"` produce the built-in codec-pipeline engine;
    `"zarrista"` lazily imports the `zarrista` package (raising a clear
    `ImportError` if it is not installed); an existing engine instance is
    returned unchanged. `config`, when given, is threaded to the engine so it
    honours the owning array's runtime configuration (e.g. `order`,
    `read_missing_chunks`); the hierarchy cache is keyed only by store, so
    engines for arrays with differing configs still share resources.
    """
    if engine is None:
        engine = "default"
    if isinstance(engine, str):
        factory = _hierarchy_factory(engine, sync=False)
        hierarchy: AsyncHierarchyEngine = _cached_hierarchy(  # type: ignore[assignment]
            engine, "async", store, factory
        )
        return _keepalive(hierarchy.array_engine(path, metadata, config), hierarchy)  # type: ignore[return-value]
    if classify_engine_arg(engine) == "sync":
        raise TypeError(
            "AsyncArray requires an async engine (AsyncArrayEngine); got a "
            f"synchronous engine of type `{type(engine).__name__}`"
        )
    return engine


def resolve_sync_engine(
    engine: ArrayEngine | EngineName | None,
    *,
    store: Store,
    path: str,
    metadata: ArrayMetadata,
    config: ArrayConfig | None = None,
) -> ArrayEngine:
    """Resolve an `engine=` argument to a bound `ArrayEngine`. See `resolve_async_engine`."""
    if engine is None:
        engine = "default"
    if isinstance(engine, str):
        factory = _hierarchy_factory(engine, sync=True)
        hierarchy: HierarchyEngine = _cached_hierarchy(  # type: ignore[assignment]
            engine, "sync", store, factory
        )
        return _keepalive(hierarchy.array_engine(path, metadata, config), hierarchy)  # type: ignore[return-value]
    if classify_engine_arg(engine) == "async":
        raise TypeError(
            "Array requires a synchronous engine (ArrayEngine); got an "
            f"async engine of type `{type(engine).__name__}`"
        )
    return engine
