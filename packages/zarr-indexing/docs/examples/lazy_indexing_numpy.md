--8<-- "lazy_indexing_numpy/README.md"

`LazyArray(source)` uses the conservative built-in reader: `source` must expose
`shape`, `dtype`, and basic integer/slice indexing, and every selected slab must
be convertible to NumPy system memory. Coordinate arrays passed through
`oindex` or `vindex` are ordered and duplicate-preserving; they are not sets.
When a view is partitioned, each `Partition.view.transform` addresses the raw
source globally while `Partition.projection.chunk_transform` stays
zero-origin and chunk-local.

## Source Code

```python
--8<-- "lazy_indexing_numpy/lazy_indexing_numpy.py"
```
