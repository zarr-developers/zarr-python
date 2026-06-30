"""
The zarrs CRUD backend for `zarr.crud`, backed by the Rust
[`zarrs`](https://zarrs.dev) crate.

Importing this module registers the `"zarrs"` backend. Requires the
`zarrs-bindings` extension (in-repo Rust crate; `uv sync --group zarrs`). Select
it with `zarr.config.set({"array.engine": "zarrs"})` or per call via
`engine="zarrs"`.
"""

try:
    import _zarrs_bindings
except ImportError as e:
    raise ImportError(
        "zarr.zarrs requires the `zarrs-bindings` package, which is not installed. "
        "It is built from the zarr-python repository: run `uv sync --group zarrs`."
    ) from e

from zarr.crud import register_backend
from zarr.zarrs._backend import ZarrsBackend

__version__: str = _zarrs_bindings.version()

register_backend("zarrs", ZarrsBackend())

__all__ = ["ZarrsBackend", "__version__"]
