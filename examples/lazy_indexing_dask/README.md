# Lazy Indexing with Dask

This example demonstrates how to use `zarr_indexing.LazyArray` with Dask, both as
an array Dask can wrap and as a source of independent tasks.

The example shows how to:

- Pass a `LazyArray` — over a Zarr array or over a view of one — to
  `dask.array.from_array`
- Build one Dask task per partition from `parts()`, compute them in parallel, and
  place each result with the partition's `out_selection`
- Read `is_complete` to tell which partitions cover a stored chunk completely
- Rely on `__dask_tokenize__`, so that equal selections produce equal tokens and
  Dask can cache and deduplicate the work

A `LazyArray` exposes no `chunks` attribute, so `dask.array.from_array` chooses
its own block size unless one is given. The partitioning that `parts()` reports
is discovered from the wrapped array and is independent of Dask's blocks.

## Running the Example

The script declares its dependencies inline
([PEP 723](https://peps.python.org/pep-0723/)), so the easiest way to run it is
with [uv](https://docs.astral.sh/uv/), which installs them automatically:

```bash
uv run examples/lazy_indexing_dask/lazy_indexing_dask.py
```

Alternatively, run it with plain Python, in which case you must first install
`zarr`, `zarr-indexing`, `dask[array]`, `numpy`, and `pytest` yourself:

```bash
python examples/lazy_indexing_dask/lazy_indexing_dask.py
```
