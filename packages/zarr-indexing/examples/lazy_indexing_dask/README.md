# Lazy Indexing with Dask

This example demonstrates how to use `zarr_indexing.LazyArray` with Dask, both as
an array Dask can wrap and as a source of independent tasks, and compares the two
ways of deferring an indexing operation.

The example shows how to:

- Pass a `LazyArray` — over a Zarr array or over a view of one — to
  `dask.array.from_array`
- Build one Dask task per partition from `parts()`, compute them in parallel, and
  place each result with the partition's `out_selection`
- Read `is_complete` to inspect coverage of a partition cell
- Inspect `__dask_tokenize__` for the example's equal source/selection pairs;
  token equality can support task deduplication but does not promise persistent caching
- Measure what a task graph costs for indexing-only work, against composing the
  same selections into one transform

A `LazyArray` exposes no `chunks` attribute, so `dask.array.from_array` chooses
its own block size unless one is given. The partitioning that `parts()` reports
is discovered from the wrapped array and is independent of Dask's blocks.

## Choosing Between Them

If Dask is doing arithmetic across chunks, reductions, rechunking, or distributed
execution, it is the right tool, and its task graph is what makes that work.

For indexing-only workloads, graph construction and scheduling can be an
additional cost. The example measures repeated leading slices and reports graph
layers and timings for the selected Dask version. It does not establish general
complexity bounds or a guaranteed speedup: Dask can optimize graphs, and costs
depend on the selection, chunk layout, and scheduler.

`LazyArray` stores one composed transform rather than retaining a wrapper for
each prior selection. Applying a chain still costs work for every operation;
index-array composition may process arrays whose size depends on earlier
selections. Reading also incurs partition planning and source I/O.

## Running the Example

The script declares its dependencies inline
([PEP 723](https://peps.python.org/pep-0723/)), so the easiest way to run it is
with [uv](https://docs.astral.sh/uv/), which installs them automatically:

```bash
cd packages/zarr-indexing
uv run --with-editable . examples/lazy_indexing_dask/lazy_indexing_dask.py
```

Alternatively, run it with plain Python, in which case you must first install
`zarr`, `zarr-indexing`, `dask[array]`, `numpy`, and `pytest` yourself:

```bash
cd packages/zarr-indexing
python examples/lazy_indexing_dask/lazy_indexing_dask.py
```
