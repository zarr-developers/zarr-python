Make `LazyArray` indexing lazy by default: use `view[...]`, `view.oindex[...]`,
and `view.vindex[...]` directly, without a `.lazy` accessor. Iteration yields
lazy views; `result()` and NumPy conversion materialize values. Add synchronous
`write(values)` and assignment through composed views, and an explicit
`EagerArrayAdapter` for consumers such as Dask that require eager indexing.
