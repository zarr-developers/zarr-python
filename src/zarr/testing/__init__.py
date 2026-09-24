from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

    from zarr.testing.store import StoreTests
    from zarr.testing.utils import assert_bytes_equal

# Imported on first access, so that `import zarr.testing` works without pytest installed.
_LAZY_ATTRS = {
    "StoreTests": "zarr.testing.store",
    "assert_bytes_equal": "zarr.testing.utils",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import importlib

        return getattr(importlib.import_module(_LAZY_ATTRS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def pytest_configure(config: pytest.Config) -> None:
    # The tests in zarr.testing are intended to be run by downstream projects.
    # To allow those downstream projects to run with `--strict-markers`, we need
    # to register an entry point with pytest11 and register our "plugin" with it,
    # which just registers the markers used in zarr.testing
    config.addinivalue_line("markers", "gpu: mark a test as requiring CuPy and GPU")


# TODO: import public buffer tests?

__all__ = ["StoreTests", "assert_bytes_equal"]
