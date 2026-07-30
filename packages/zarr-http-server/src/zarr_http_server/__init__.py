"""Zarr-http-server: HTTP server for Zarr stores, arrays, and groups."""

from importlib.metadata import version

from zarr_http_server._serve import (
    DEFAULT_MAX_BODY_SIZE,
    BackgroundServer,
    CorsOptions,
    HTTPMethod,
    node_app,
    serve_node,
    serve_store,
    store_app,
)

__version__ = version("zarr-http-server")

__all__ = [
    "DEFAULT_MAX_BODY_SIZE",
    "BackgroundServer",
    "CorsOptions",
    "HTTPMethod",
    "__version__",
    "node_app",
    "serve_node",
    "serve_store",
    "store_app",
]
