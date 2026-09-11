# Runtime configuration

[`zarr.config`][] is responsible for managing the configuration of zarr and
is based on the [donfig](https://github.com/pytroll/donfig) Python library.

Configuration values can be set using code like the following:

```python exec="true" session="config" source="above" result="ansi"
import zarr

zarr.config.set({'array.order': 'F'})

print(zarr.config.get('array.order'))
```

`zarr.config.set` can also be used as a context manager, which restores the
previous configuration on exit, and `zarr.config.reset` restores the default
configuration:

```python exec="true" session="config" source="above" result="ansi"
zarr.config.reset()

with zarr.config.set({'array.order': 'F'}):
    print(zarr.config.get('array.order'))

print(zarr.config.get('array.order'))
```

Alternatively, configuration values can be set using environment variables, e.g.
`ZARR_ARRAY__ORDER=F`.

The configuration can also be read from a YAML file in standard locations.
For more information, see the
[donfig documentation](https://donfig.readthedocs.io/en/latest/).

Configuration options include the following:

- Default Zarr format `default_zarr_format`
- Default array order in memory `array.order`
- Whether empty chunks are written to storage `array.write_empty_chunks`
- Enable experimental rectilinear chunks `array.rectilinear_chunks`
- Whether missing chunks are filled with the array's fill value on read `array.read_missing_chunks` (default `True`). Set to `False` to raise a [`ChunkNotFoundError`][zarr.errors.ChunkNotFoundError] instead.
- Async and threading options, e.g. `async.concurrency` and `threading.max_workers`
- Selections of implementations of codecs, codec pipelines and buffers
- Enabling GPU support with `zarr.config.enable_gpu()`. See [GPU support](gpu.md) for more.
- Control request merging when reading multiple chunks from the same shard with `array.sharding_coalesce_max_gap_bytes` and `array.sharding_coalesce_max_bytes`. Reads of nearby chunks are coalesced into a single request to the store when separated by at most `sharding_coalesce_max_gap_bytes` and the resulting merged read is no larger than `sharding_coalesce_max_bytes`.
- Set the temporary write location for `LocalStore` writes with `store.local.tmp_dir`.

For selecting custom implementations of codecs, pipelines, buffers and ndbuffers,
first register the implementations in the registry and then select them in the config.
For example, an implementation of the bytes codec in a class `'custompackage.NewBytesCodec'`,
requires the value of `codecs.bytes` to be `'custompackage.NewBytesCodec'`.

This is the current default configuration:

```python exec="true" session="config" source="above" result="ansi"
from pprint import pprint
import io
output = io.StringIO()
zarr.config.pprint(stream=output, width=60)
print(output.getvalue())
```
