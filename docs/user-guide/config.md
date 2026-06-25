# Runtime configuration

[`zarr.config`][] is a `ZarrConfigManager` instance that manages all runtime
settings for zarr.  It provides both typed attribute access and a dotted-string
key API.

Configuration values can be set using code like the following:

```python exec="true" session="config" source="above" result="ansi"

import zarr

print(zarr.config.get('array.order'))
```

```python exec="true" session="config" source="above" result="ansi"
zarr.config.set({'array.order': 'F'})

print(zarr.config.get('array.order'))
```

Alternatively, configuration values can be set using environment variables.
The variable name uses a `ZARR_` prefix, with `__` to denote nesting, e.g.
`ZARR_ARRAY__ORDER=F`.

The configuration can also be read from YAML files.  Place a `zarr.yaml` (or
any `.yaml`/`.yml` file) in `~/.config/zarr/`, or point the `ZARR_CONFIG`
environment variable at a specific file path.

Configuration options include the following:

- Default Zarr format `default_zarr_format`
- Default array order in memory `array.order`
- Whether empty chunks are written to storage `array.write_empty_chunks`
- Enable experimental rectilinear chunks `array.rectilinear_chunks`
- Whether missing chunks are filled with the array's fill value on read `array.read_missing_chunks` (default `True`). Set to `False` to raise a [`ChunkNotFoundError`][zarr.errors.ChunkNotFoundError] instead.
- Async and threading options, e.g. `async.concurrency` and `threading.max_workers`
- Selections of implementations of codecs, codec pipelines and buffers
- Enabling GPU support with `zarr.config.enable_gpu()`. See GPU support for more.
- Control request merging when reading multiple chunks from the same shard with `array.sharding_coalesce_max_gap_bytes` and `array.sharding_coalesce_max_bytes`. Reads of nearby chunks are coalesced into a single request to the store when separated by at most `sharding_coalesce_max_gap_bytes` and the resulting merged read is no larger than `sharding_coalesce_max_bytes`.

For selecting custom implementations of codecs, pipelines, buffers and ndbuffers,
first register the implementations in the registry and then select them in the config.
For example, an implementation of the bytes codec in a class `'custompackage.NewBytesCodec'`,
requires the value of `codecs.bytes.name` to be `'custompackage.NewBytesCodec'`.

This is the current default configuration:

```python exec="true" session="config" source="above" result="ansi"
from pprint import pprint
pprint(zarr.config.to_dict())
```
