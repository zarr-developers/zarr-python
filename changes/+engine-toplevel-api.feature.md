Added a selectable execution **engine** that routes zarr-python's top-level API
through `zarr.crud`. The engine is a per-array `ArrayConfig` field, defaulted from
the new `array.engine` config key (default `"zarr"`, the native path) and
overridable per call via an `engine=` argument on `create_array` / `open_array`
(and the `Group` variants). Selecting a `zarr.crud` backend (`"reference"`,
`"zarrs"`, `"zarrista"`) routes data access (`Array.__getitem__` /
`__setitem__`), array creation, and open through that backend. The policy is
strict: operations a non-native engine cannot express — advanced
(orthogonal/coordinate/mask/block) indexing, stores it cannot ingest, the
`out`/`fields` arguments — raise rather than silently falling back. Also added
`zarr.crud.write_region`, the write counterpart to `read_region`. The
unreleased `crud.backend` config key is replaced by `array.engine`.
