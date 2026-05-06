Replace the `zarr.core.array.DefaultFillValue` sentinel class with a PEP 661
sentinel from `typing_extensions.Sentinel`. The public singleton
`zarr.core.array.DEFAULT_FILL_VALUE` is preserved and continues to work as
before. The `DefaultFillValue` class is no longer exported; downstream code
that did `isinstance(x, DefaultFillValue)` should switch to
`x is DEFAULT_FILL_VALUE`. The minimum supported `typing_extensions` version
is now `4.15`.
