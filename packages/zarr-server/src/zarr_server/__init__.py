"""Zarr-server: HTTP server for Zarr stores, arrays, and groups."""

from importlib.metadata import version

__version__ = version("zarr-server")

__all__ = ["__version__"]
