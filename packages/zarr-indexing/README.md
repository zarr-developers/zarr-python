# zarr-indexing

Composable, lazy coordinate transforms for Zarr array indexing.

Documentation: <https://zarr-indexing.readthedocs.io/>

This package implements TensorStore-inspired index transforms. The core idea:
every indexing operation (slicing, fancy indexing, etc.) produces a coordinate
mapping from user space to storage space. These mappings compose lazily — no
I/O until you explicitly read or write.

Key types:

- `LazyArray` — wraps any array-API-like array (NumPy, zarr, CuPy, ...) and adds
  a `.lazy` accessor: `LazyArray(x).lazy[10:50, ::2].lazy.oindex[[3, 1, 1], :]`
  composes a transform and returns a new view without reading data, and
  `result()` materializes it
- `IndexDomain` — a rectangular region of integer coordinates
- `IndexTransform` — maps input coordinates to storage coordinates
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

## Developing

Package-scoped development commands live in the [`justfile`](./justfile)
(requires [just](https://github.com/casey/just)):

```
just test        # run the test suite (extra args go to pytest)
just lint        # ruff, same invocation as CI
just typecheck   # pyright, same invocation as CI
just docs-check  # strict build of the docs site
just check       # all of the above
just docs-serve  # serve the docs site locally
```

Run them from this directory, or from anywhere in the repository as
`just packages/zarr-indexing/<recipe>`.

The test recipe runs against the workspace-root environment, because the
chunk-resolution tests exercise this package against `zarr`'s chunk grids and
`zarr` is deliberately not a dependency of this package.

## License

MIT
