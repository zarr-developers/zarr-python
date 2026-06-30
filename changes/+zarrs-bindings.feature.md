Added `zarr.crud`, an experimental backend-agnostic low-level functional API for
zarr hierarchy CRUD (`create_*`, `read_chunk`, `read_region`, `read_encoded_chunk`,
`write_chunk`, `delete_chunk`, `read_metadata`, `delete_node`, `list_children`).
Array routines take an explicit metadata document, enabling read-only views.
Operations delegate to a pluggable `CrudBackend`: a pure-Python reference backend
(the default) or the zarrs-accelerated backend in `zarr.zarrs`, backed by the Rust
[zarrs](https://zarrs.dev) crate via the in-repo `zarrs-bindings` PyO3 crate.
Select a backend with the `array.engine` config key or a per-call `engine=`
argument. Build the zarrs backend for development with `uv sync --group zarrs`.
