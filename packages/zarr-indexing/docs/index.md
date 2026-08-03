# zarr-indexing

Composable, lazy coordinate transforms for Zarr array indexing.

<div class="grid cards" markdown>

- **Use lazy indexing**

  ---

  Build NumPy-shaped views without reading until you materialize a result.
  Start with [chapter 1: An index selects coordinates](guide/01-indexing.md).

- **Integrate a chunked source**

  ---

  Turn a request into source-independent chunk projections for your own reader,
  writer, or scheduler. Start with
  [chapter 4: A request becomes a chunk plan](guide/04-chunks.md). If literal
  coordinates are unfamiliar, read
  [chapter 2: Coordinates are addresses](guide/02-transforms.md) first.

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
--8<-- "examples/canonical_slice.py:landing-quickstart"
```

Continue through [chapter 2: Coordinates are addresses](guide/02-transforms.md)
and [chapter 3: Lazy views compose](guide/03-composition.md) to understand how
the view stays lazy and how repeated indexing combines.

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
