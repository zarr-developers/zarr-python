from __future__ import annotations

from typing import TYPE_CHECKING

from zarr.core.config import config

if TYPE_CHECKING:
    from zarr.crud._backend import CrudBackend

# Backends are registered at import time (reference by zarr.crud, zarrs by
# zarr.zarrs). CPython's import lock plus the GIL make this dict safe without
# additional locking.
_BACKENDS: dict[str, CrudBackend] = {}


def register_backend(name: str, backend: CrudBackend) -> None:
    """Register a CRUD backend instance under `name`."""
    _BACKENDS[name] = backend


def get_backend(name: str | None = None) -> CrudBackend:
    """Resolve a backend by name, or the configured default when `name` is None.

    Selecting `"zarrs"` imports `zarr.zarrs` if needed so it can self-register.
    """
    if name is None:
        name = config.get("crud.backend")
    if name not in _BACKENDS and name == "zarrs":
        # "reference" is pre-registered by zarr.crud at import; "zarrs" lives in a
        # separate package that may not be imported yet, so load it on demand.
        try:
            import zarr.zarrs  # noqa: F401  (import registers the zarrs backend)
        except ImportError as e:
            raise ImportError(
                "the 'zarrs' CRUD backend requires the zarrs-bindings extension; "
                "install it with: uv sync --group zarrs"
            ) from e
    if name not in _BACKENDS:
        raise KeyError(f"no CRUD backend registered as {name!r}; registered: {sorted(_BACKENDS)}")
    return _BACKENDS[name]
