"""
The zarrista CRUD backend for `zarr.crud`, backed by the Rust
[`zarrs`](https://zarrs.dev) crate through the
[`zarrista`](https://developmentseed.org/zarrista) package.

Importing this module registers the `"zarrista"` backend. Requires the
`zarrista` package (`uv sync --group zarrista`). Select it with
`zarr.config.set({"array.engine": "zarrista"})` or per call via
`engine="zarrista"`.

Unlike the in-tree zarrs bindings (`zarr.zarrs`), zarrista has no generic
Python-store callback bridge, so this backend only operates on stores it can map
to a zarrista store. It currently supports `LocalStore`; other stores raise
`UnsupportedStoreError`.
"""

try:
    import zarrista  # noqa: F401
except ImportError as e:
    raise ImportError(
        "zarr.zarrista requires the `zarrista` package, which is not installed. "
        "Install it with `uv sync --group zarrista` (or `pip install zarrista`)."
    ) from e

from zarr.crud import register_backend
from zarr.zarrista._backend import UnsupportedStoreError, ZarristaBackend

register_backend("zarrista", ZarristaBackend())

__all__ = ["UnsupportedStoreError", "ZarristaBackend"]
