Replaced the `donfig`-based configuration with a statically-typed
configuration object. `zarr.config` now provides precise static types for
attribute access (`zarr.config.array.order`) and for the dotted-string API
(`zarr.config.get("array.order")`). The string API, environment-variable
ingestion (`ZARR_FOO__BAR`), YAML config files, `config.set` (permanent and
as a context manager), `config.reset`, `config.enable_gpu`, and the
`deprecations` mechanism are all preserved. The `donfig` dependency has been
removed.
