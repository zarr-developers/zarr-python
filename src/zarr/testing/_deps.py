from __future__ import annotations

__all__ = ["missing_dependency"]


def missing_dependency(name: str, module: str) -> ImportError:
    """Build the error raised when a `zarr.testing` module is missing a test dependency.

    `zarr.testing` depends on pytest and hypothesis, which are not dependencies of zarr.
    They are declared in the `testing` extra.
    """
    return ImportError(
        f"{module} requires {name}, which is not installed. "
        "Install the zarr testing extra with `pip install 'zarr[testing]'`."
    )
