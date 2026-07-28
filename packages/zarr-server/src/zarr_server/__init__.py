"""Zarr-server: HTTP server for Zarr stores, arrays, and groups."""

from importlib.metadata import version

from zarr_server._serve import (
    BackgroundServer,
    CorsOptions,
    HTTPMethod,
    node_app,
    serve_node,
    serve_store,
    store_app,
)

__version__ = version("zarr-server")

__all__ = [
    "BackgroundServer",
    "CorsOptions",
    "HTTPMethod",
    "__version__",
    "node_app",
    "serve_node",
    "serve_store",
    "store_app",
]
