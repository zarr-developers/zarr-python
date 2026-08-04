--8<-- "napari_chunk_cache/README.md"

`LazyArray` owns the indexing-derived result shape and assembly, while the
example's `SystemMemoryChunkReader` owns synchronous system-memory cache state
and chunk reads for each materialized part.

## Source Code

```python
--8<-- "napari_chunk_cache/napari_chunk_cache.py"
```
