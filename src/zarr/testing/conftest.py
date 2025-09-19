import pytest


def pytest_configure(config: pytest.Config) -> None:
    # The tests in zarr.testing are intended to be run by downstream projects.
    # To allow those downstream projects to run with `--strict-markers`, we need
    # to register an entry point with pytest11 and register our "plugin" with it,
    # which just registers the markers used in zarr.testing
    config.addinivalue_line("markers", "gpu: mark a test as requiring CuPy and GPU")
