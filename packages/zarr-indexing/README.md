# zarr-indexing

Composable, lazy coordinate transforms for Zarr array indexing.

Documentation: <https://zarr-indexing.readthedocs.io/>

This package implements TensorStore-inspired index transforms. The core idea:
every indexing operation (slicing, fancy indexing, etc.) produces a coordinate
mapping from user space to storage space. These mappings compose lazily — no
I/O until you explicitly read or write.

Key types:

- `LazyArray` — wraps a system-memory/basic-indexing source and adds a `.lazy`
  accessor: `LazyArray.from_numpy(numpy_array).lazy[10:50, ::2].lazy.oindex[[3, 1, 1], :]`
  composes a transform and returns a new view without reading data, and
  `result()` materializes it into owned system memory. `LazyArray(source)` uses
  the conservative basic reader; `from_numpy` explicitly selects NumPy's
  optimized reader. Device arrays require an explicit custom reader responsible
  for transferring values into the supplied system-memory output buffer.
- `Reader` — the explicit backend execution boundary: transforms say which
  values belong in the result, while readers say how a backend obtains them
- `IndexDomain` — a rectangular region of integer coordinates
- `IndexTransform` — maps input coordinates to storage coordinates
- `ChunkPlan` and `ChunkProjection` — lazily partition a selection over a
  caller-selected grid and pair each chunk-local transform with its placement in
  the request, without binding a storage backend or scheduler
- `ConstantMap`, `DimensionMap`, `ArrayMap` — the three ways a single output
  dimension can depend on the input
- `compose` — chain two transforms into one

The package depends only on NumPy and the standard library; it does not import
`zarr`. It is developed in the [zarr-python](https://github.com/zarr-developers/zarr-python)
repository and consumed by `zarr` to resolve array indexing operations.

## Installation

```bash
pip install zarr-indexing
```

## Examples

- [Lazy indexing a NumPy array](examples/lazy_indexing_numpy/README.md)
- [Lazy indexing with Dask](examples/lazy_indexing_dask/README.md)

## Contributing

Development commands, the test suite and the docs build are described in
[CONTRIBUTING.md](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-indexing/CONTRIBUTING.md)
in the repository. Issues and pull requests go to
[zarr-developers/zarr-python](https://github.com/zarr-developers/zarr-python).
