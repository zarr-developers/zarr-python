# Runtime configuration

[`zarr.config`][] is responsible for managing the configuration of zarr and
is based on the [donfig](https://github.com/pytroll/donfig) Python library.

Configuration values can be set using code like the following:

```python exec="true" session="config" source="above" result="ansi"

import zarr

print(zarr.config.get('array.order'))
```

```python exec="true" session="config" source="above" result="ansi"
zarr.config.set({'array.order': 'F'})

print(zarr.config.get('array.order'))
```

Alternatively, configuration values can be set using environment variables, e.g.
`ZARR_ARRAY__ORDER=F`.

The configuration can also be read from a YAML file in standard locations.
For more information, see the
[donfig documentation](https://donfig.readthedocs.io/en/latest/).

Configuration options include the following:

- Default Zarr format `default_zarr_version`
- Default array order in memory `array.order`
- Whether empty chunks are written to storage `array.write_empty_chunks`
- Async and threading options, e.g. `async.concurrency` and `threading.max_workers`
- Selections of implementations of codecs, codec pipelines and buffers
- Enabling GPU support with `zarr.config.enable_gpu()`. See GPU support for more.

For selecting custom implementations of codecs, pipelines, buffers and ndbuffers,
first register the implementations in the registry and then select them in the config.
For example, an implementation of the bytes codec in a class `'custompackage.NewBytesCodec'`,
requires the value of `codecs.bytes.name` to be `'custompackage.NewBytesCodec'`.

This is the current default configuration:

```python exec="true" session="config" source="above" result="ansi"
from pprint import pprint
import io
output = io.StringIO()
zarr.config.pprint(stream=output, width=60)
print(output.getvalue())
```
