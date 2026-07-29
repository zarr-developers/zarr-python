# zarr-indexing

Composable, lazy coordinate transforms for Zarr array indexing.

This package implements TensorStore-inspired index transforms. The core idea:
every indexing operation (slicing, fancy indexing, etc.) produces a coordinate
mapping from user space to storage space. These mappings compose lazily — no
I/O until you explicitly read or write.

Key types:

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

## License

MIT
