# zarr-indexing

Composable, lazy coordinate transforms for Zarr array indexing.

Many arrays only support plain slicing: a chunked store, an FFI binding, an
HTTP endpoint. `zarr-indexing` grafts the full NumPy indexing dialect onto
such a source — negative steps, `oindex`, `vindex`, boolean masks, and
chains of all of them. Selections compose without reading anything; data
moves only when you ask for values; and a request can be turned into an
exact plan of the chunks it touches, for a reader, writer, or scheduler you
control.

## Install

`zarr-indexing` is developed in the
[zarr-python repository](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-indexing)
and released independently of `zarr` itself:

```
pip install zarr-indexing
```

## Quickstart

Wrap an array, compose a lazy view through `.lazy`, and call `result()` when
you want its values:

```python
--8<-- "snippets/canonical_slice.py:landing-quickstart"
```

Nothing is read until the `result()` call, however many selections are
composed. [Lazy views compose](guide/index.md#lazy-views-compose) shows how
the chain stays one description, and where the materialization boundary is.

## Learn more

- [Visual guide](guide/index.md) — one selection followed from coordinates to
  chunk plan. Using lazy indexing, start at
  [An index selects coordinates](guide/index.md#an-index-selects-coordinates);
  integrating a chunked backend, start at
  [A request becomes a chunk plan](guide/index.md#a-request-becomes-a-chunk-plan).
- [Indexing pattern reference](guide/patterns.md) — every selection form with
  its NumPy-verified result.
- [Integration boundaries](guide/integrations.md) — what a reader, writer, or
  scheduler owns, and what the plan owns.
- [Lazy indexing a NumPy array](examples/lazy_indexing_numpy.md) and
  [with Dask](examples/lazy_indexing_dask.md) — runnable examples.
- [The ndsel wire format](ndsel.md) — the JSON form of a selection.
- [Design notes](design-notes.md) — TensorStore lineage, box vs query, and
  deliberate limits.
- [API reference](api/index.md)
- [Changelog](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-indexing/CHANGELOG.md)
  · [License (MIT)](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-indexing/LICENSE.txt)
