from __future__ import annotations

from typing import TYPE_CHECKING

from zarr.core.config import config

if TYPE_CHECKING:
    from zarr.crud._backend import CrudBackend

# Backends are registered at import time (reference by zarr.crud, zarrs by
# zarr.zarrs). CPython's import lock plus the GIL make this dict safe without
# additional locking.
_BACKENDS: dict[str, CrudBackend] = {}

# Backends that live in a separate, optionally-installed package: the module to
# import (which self-registers) and the install hint shown when it is missing.
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    "zarrs": ("zarr.zarrs", "the zarrs-bindings extension; install it with: uv sync --group zarrs"),
    "zarrista": (
        "zarr.zarrista",
        "the `zarrista` package; install it with: uv sync --group zarrista",
    ),
}


def register_backend(name: str, backend: CrudBackend) -> None:
    """Register a CRUD backend instance under `name`."""
    _BACKENDS[name] = backend


def list_backends() -> list[str]:
    """The names of every available CRUD backend: those already registered plus
    the lazily-loaded ones (`zarrs`, `zarrista`) that import on first use. These
    are exactly the non-native values accepted by the top-level `engine`."""
    return sorted(_BACKENDS.keys() | _LAZY_BACKENDS.keys())


def get_backend(name: str | None = None) -> CrudBackend:
    """Resolve a backend by name, or the configured default when `name` is None.

    Selecting a lazily-loaded backend (`"zarrs"`, `"zarrista"`) imports its
    package if needed so it can self-register.
    """
    if name is None:
        # The top-level `array.engine` config drives the default backend. The
        # `"zarr"` engine means the native path, which never routes here; map it
        # to the pure-Python reference backend for direct crud callers.
        name = config.get("array.engine")
        if name == "zarr":
            name = "reference"
    if name not in _BACKENDS and name in _LAZY_BACKENDS:
        # "reference" is pre-registered by zarr.crud at import; these backends
        # live in separate packages that may not be imported yet, so load on
        # demand.
        import importlib

        module, hint = _LAZY_BACKENDS[name]
        try:
            importlib.import_module(module)  # import registers the backend
        except ImportError as e:
            raise ImportError(f"the {name!r} CRUD backend requires {hint}") from e
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown backend {name!r}; available backends: {list_backends()}. "
            f"At the top level these are the valid engines, plus 'zarr' for the "
            f"native path."
        )
    return _BACKENDS[name]
