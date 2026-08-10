# zarr-indexing

Composable, lazy coordinate transforms for Zarr array indexing.

Many arrays only support plain slicing: a chunked store, an FFI binding, an
HTTP endpoint. `zarr-indexing` grafts the full NumPy indexing dialect onto
such a source — negative steps, `oindex`, `vindex`, boolean masks, and
chains of all of them. Selections compose without reading anything; data
moves only when you ask for values; and a request can be turned into an
exact plan of the chunks it touches, for a reader, writer, or scheduler you
control.

<div class="grid cards" markdown>

- **Use lazy indexing**

  ---

  Build NumPy-shaped views without reading until you materialize a result.
  Start with [An index selects coordinates](guide/index.md#an-index-selects-coordinates).

- **Integrate a chunked source**

  ---

  Turn a request into source-independent chunk projections for your own reader,
  writer, or scheduler. Start with
  [A request becomes a chunk plan](guide/index.md#a-request-becomes-a-chunk-plan).
  If literal
  coordinates are unfamiliar, read
  [Coordinates are addresses](guide/index.md#coordinates-are-addresses) first.

</div>

## Install

`zarr-indexing` is developed in the
[zarr-python repository](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-indexing)
and released independently of `zarr` itself:

```
pip install zarr-indexing
```

## Minimal quickstart

Wrap an array, compose a lazy view through `.lazy`, and call `result()` when
you want its values:

```python
--8<-- "snippets/canonical_slice.py:landing-quickstart"
```

Continue through [Coordinates are addresses](guide/index.md#coordinates-are-addresses)
and [Lazy views compose](guide/index.md#lazy-views-compose) to understand how the
view stays lazy and how repeated indexing combines.

## Reference

- [Visual guide](guide/index.md)
- [Indexing pattern reference](guide/patterns.md)
- [Integration boundaries](guide/integrations.md)
- [Lazy indexing a NumPy array](examples/lazy_indexing_numpy.md)
- [Lazy indexing with Dask](examples/lazy_indexing_dask.md)
- [The ndsel wire format](ndsel.md)
- [Design notes](design-notes.md)
- [API reference](api/index.md)
- [Changelog](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-indexing/CHANGELOG.md)
- [License (MIT)](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-indexing/LICENSE.txt)
