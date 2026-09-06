# zarr-chunk-key-encoding

Chunk key encodings for Zarr version 3 arrays.

This package implements the `default` and `v2` encodings defined by the
[Zarr v3 core specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding).
It uses [zarr-metadata](https://zarr-metadata.readthedocs.io/) for JSON types
and does not depend on `zarr`.

Documentation: <https://zarr-chunk-key-encoding.readthedocs.io/>

## Installation

```bash
pip install zarr-chunk-key-encoding
```

## Usage

```python
>>> from zarr_chunk_key_encoding import chunk_key_encoding_from_json
>>> encoding = chunk_key_encoding_from_json(
...     {"name": "default", "configuration": {"separator": "/"}}
... )
>>> encoding.encode((1, 23))
'c/1/23'
>>> encoding.decode("c/1/23")
(1, 23)
>>> encoding.to_json()
{'name': 'default', 'configuration': {'separator': '/'}}
```

Decoding is strict: malformed and non-canonical keys such as `c/01` and
`c/-1` raise `ChunkKeyDecodeError`. Encoding likewise rejects coordinates
that are not non-negative integers.

When many operations share a chunk or shard grid, prepare a bounded view:

```python
>>> bounded = encoding.to_bounded((2, 3))
>>> bounded.encode((1, 2))
'c/1/2'
>>> bounded.decode("c/1/2")
(1, 2)
```

The grid shape is normalized and validated once when the view is created.
Each operation then checks only its coordinates or decoded key against that
prepared shape. For sharded arrays, pass the shard-grid shape because shards
are the unit of storage.

Chunk key encoding is an extension point, but this package intentionally
covers only the two core encodings. Registration and plugin discovery are
shared concerns for all Zarr extension points and belong in a shared layer.

## Developing

Package-scoped commands live in the [`justfile`](./justfile):

```text
just test
just lint
just typecheck
just docs-check
just check
just docs-serve
```

Run them from this directory, or from the repository root as
`just packages/zarr-chunk-key-encoding/<recipe>`.

## License

MIT
