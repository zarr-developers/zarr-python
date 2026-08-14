# Lazy Indexing a NumPy Array

This example demonstrates how to wrap an array in `zarr_indexing.LazyArray` and
index it without reading data.

The example shows how to:

- Wrap a NumPy array and read the forwarded `shape`, `dtype`, and `ndim`
- Compose selections through `.lazy[...]`, `.lazy.oindex[...]`, and
  `.lazy.vindex[...]`, and materialize the composed view once with `result()`
- Tell a box selection (slices and integers, described by an interval and a step
  per dimension) from a query selection (points gathered through an index array)
  using `is_box`, `bounding_box()`, and `strides()`
- Declare a partitioning with `with_parts()`, iterate it with `parts()`, and
  assemble a result from the partitions

`LazyArray` wraps any object exposing `shape`, `dtype`, and `__getitem__`, so the
same API applies to a Zarr array, and the partitioning is then discovered from
the array's chunks. The Dask example covers that case.

## Running the Example

The script declares its dependencies inline
([PEP 723](https://peps.python.org/pep-0723/)), so the easiest way to run it is
with [uv](https://docs.astral.sh/uv/), which installs them automatically:

```bash
cd packages/zarr-indexing
uv run --with-editable . examples/lazy_indexing_numpy/lazy_indexing_numpy.py
```

Alternatively, run it with plain Python, in which case you must first install
`zarr-indexing`, `numpy`, and `pytest` yourself:

```bash
cd packages/zarr-indexing
python examples/lazy_indexing_numpy/lazy_indexing_numpy.py
```
