Added `zarr.zarrista`, a `zarr.crud` backend backed by the Rust
[zarrs](https://zarrs.dev) crate through the
[zarrista](https://developmentseed.org/zarrista) package. It accelerates
chunk-level I/O and, unlike the in-repo `zarr.zarrs` backend, has no generic
Python-store callback bridge: it operates only on stores it can map to a
zarrista store (`LocalStore` and obstore-backed `ObjectStore`) and raises
`UnsupportedStoreError` for any other store. Select it with the `array.engine`
config key or a per-call
`engine="zarrista"` argument; install it with `uv sync --group zarrista`.
