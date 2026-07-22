"""Resolve an `engine=` argument to a bound array engine."""

from __future__ import annotations

import contextlib
import weakref
from typing import TYPE_CHECKING, Literal

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
    from zarr.core.metadata import ArrayMetadata

__all__ = ["EngineName", "resolve_async_engine", "resolve_sync_engine"]

EngineName = Literal["default", "zarrista"]

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
        return (  # type: ignore[no-any-return]
            ZarristaHierarchyEngine if sync else ZarristaAsyncHierarchyEngine
        )
    raise ValueError(f"unknown engine name {name!r}; expected 'default' or 'zarrista'")


def resolve_async_engine(
    engine: AsyncArrayEngine | EngineName | None,
    *,
    store: Store,
    path: str,
    metadata: ArrayMetadata,
) -> AsyncArrayEngine:
    """Resolve an `engine=` argument to a bound `AsyncArrayEngine`.

    `None` and `"default"` produce the built-in codec-pipeline engine;
    `"zarrista"` lazily imports the `zarrista` package (raising a clear
    `ImportError` if it is not installed); an existing engine instance is
    returned unchanged.
    """
    if engine is None:
        engine = "default"
    if isinstance(engine, str):
        factory = _hierarchy_factory(engine, sync=False)
        hierarchy: AsyncHierarchyEngine = _cached_hierarchy(  # type: ignore[assignment]
            engine, "async", store, factory
        )
        return _keepalive(hierarchy.array_engine(path, metadata), hierarchy)  # type: ignore[return-value]
    return engine


def resolve_sync_engine(
    engine: ArrayEngine | EngineName | None,
    *,
    store: Store,
    path: str,
    metadata: ArrayMetadata,
) -> ArrayEngine:
    """Resolve an `engine=` argument to a bound `ArrayEngine`. See `resolve_async_engine`."""
    if engine is None:
        engine = "default"
    if isinstance(engine, str):
        factory = _hierarchy_factory(engine, sync=True)
        hierarchy: HierarchyEngine = _cached_hierarchy(  # type: ignore[assignment]
            engine, "sync", store, factory
        )
        return _keepalive(hierarchy.array_engine(path, metadata), hierarchy)  # type: ignore[return-value]
    return engine
