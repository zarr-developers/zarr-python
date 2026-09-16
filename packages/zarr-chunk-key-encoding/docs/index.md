# zarr-chunk-key-encoding

Chunk key encodings map chunk-grid coordinates to storage keys. This package
implements the two encodings in the
[Zarr v3 core specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding):

- `default`: `(1, 23)` becomes `c/1/23` by default.
- `v2`: `(1, 23)` becomes `1.23` by default, matching Zarr v2 layouts.

## Quickstart

```python
>>> from zarr_chunk_key_encoding import chunk_key_encoding_from_json
>>> encoding = chunk_key_encoding_from_json(
...     {"name": "default", "configuration": {"separator": "/"}}
... )
>>> encoding.encode((1, 23))
'c/1/23'
>>> encoding.decode("c/1/23")
(1, 23)
```

[`decode`][zarr_chunk_key_encoding.ChunkKeyEncoding.decode] rejects keys that
[`encode`][zarr_chunk_key_encoding.ChunkKeyEncoding.encode] would not produce,
including non-canonical integer spellings. Encoded coordinates must be
non-negative integers.

## Prepared grid bounds

Use [`to_bounded`][zarr_chunk_key_encoding.ChunkKeyEncoding.to_bounded] when
repeated operations share one chunk or shard grid:

```python
>>> bounded = encoding.to_bounded((2, 3))
>>> bounded.encode((1, 2))
'c/1/2'
>>> bounded.decode("c/1/2")
(1, 2)
```

The resulting
[`BoundedChunkKeyEncoding`][zarr_chunk_key_encoding.BoundedChunkKeyEncoding]
validates and stores the grid shape once, then applies rank and bounds checks
to each operation. For a sharded array, pass its shard-grid shape.

## Scope

The package covers the closed core set and provides no registration or plugin
discovery API. It uses [zarr-metadata](https://zarr-metadata.readthedocs.io/)
for JSON types and does not depend on `zarr`.

Install it with `pip install zarr-chunk-key-encoding`. See the
[API reference](api/index.md) for the complete public surface.
