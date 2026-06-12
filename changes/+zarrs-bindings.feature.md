Added `zarr.zarrs`, an experimental low-level functional API for zarr hierarchy
CRUD backed by the Rust [zarrs](https://zarrs.dev) crate via the new in-repo
`zarrs-bindings` PyO3 crate. Array routines take an explicit metadata document,
enabling read-only views such as decoding chunks with externally supplied
metadata or reading raw encoded chunk bytes. Build for development with
`uv sync --group zarrs`.
