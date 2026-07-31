# Lazy Indexing with Dask

This example demonstrates how to use `zarr_indexing.LazyArray` with Dask, both as
an array Dask can wrap and as a source of independent tasks, and compares the two
ways of deferring an indexing operation.

The example shows how to:

- Pass a `LazyArray` — over a Zarr array or over a view of one — to
  `dask.array.from_array`
- Build one Dask task per partition from `parts()`, compute them in parallel, and
  place each result with the partition's `out_selection`
- Read `is_complete` to tell which partitions cover a stored chunk completely
- Rely on `__dask_tokenize__`, so that equal selections produce equal tokens and
  Dask can cache and deduplicate the work
- Measure what a task graph costs for indexing-only work, against composing the
  same selections into one transform

A `LazyArray` exposes no `chunks` attribute, so `dask.array.from_array` chooses
its own block size unless one is given. The partitioning that `parts()` reports
is discovered from the wrapped array and is independent of Dask's blocks.

## Choosing Between Them

If Dask is doing arithmetic across chunks, reductions, rechunking, or distributed
execution, it is the right tool, and its task graph is what makes that work.

If Dask is used *only* to defer indexing — take a view now, read it later, with
no computation in between — then the graph is overhead. Dask slices the chunk
grid on every indexing operation and records another layer, so composing
selections costs time proportional to both the depth of the chain and the number
of chunks in the array, and reading walks what was accumulated. `LazyArray`
composes each selection into the single transform it already holds, so composing
is independent of the depth of the chain, and reading enumerates only the
partitions the selection touches. The last test in this example prints both, and
the gap widens with the number of chunks and the number of selections.

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
