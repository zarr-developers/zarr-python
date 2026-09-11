import importlib.util
import warnings
from typing import TYPE_CHECKING

from zarr.errors import ZarrUserWarning

if importlib.util.find_spec("pytest") is not None:
    from zarr.testing.store import StoreTests
    from zarr.testing.utils import assert_bytes_equal
else:
    warnings.warn(
        "pytest not installed, skipping test suite", category=ZarrUserWarning, stacklevel=2
    )

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: "pytest.Config") -> None:
    # The tests in zarr.testing are intended to be run by downstream projects.
    # To allow those downstream projects to run with `--strict-markers`, we need
    # to register an entry point with pytest11 and register our "plugin" with it,
    # which just registers the markers used in zarr.testing
    config.addinivalue_line("markers", "gpu: mark a test as requiring CuPy and GPU")


# TODO: import public buffer tests?

__all__ = ["StoreTests", "assert_bytes_equal"]
