# Runtime configuration

[`zarr.config`][] is a `ZarrConfigManager` instance that manages all runtime
settings for zarr.  It provides both typed attribute access and a dotted-string
key API.

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

Alternatively, configuration values can be set using environment variables.
The variable name uses a `ZARR_` prefix, with `__` to denote nesting, e.g.
`ZARR_ARRAY__ORDER=F`.

External configuration is processed in two stages. Collection recognizes names
from the configuration schema and a separate set of environment controls:
`ZARR_CONFIG` and `ZARR_ROOT_CONFIG` control file discovery, and
`ZARR_BENCHMARK_CLEAR_CACHE` controls benchmark cache clearing. These controls
are left to their consumers and never become configuration fields. Unrecognized
`ZARR_*` names produce a warning and are ignored. The `codecs` namespace is open,
so environment variables can also select implementations for custom codec names.

Config creation then validates the collected values against the schema and
applies them to the defaults. Invalid values raise an error identifying the
field, for example `ZARR_ARRAY__ORDER=Q`. Programmatic `config.set()` validates
keys but continues to defer value validation to the setting's use site.

The configuration can also be read from YAML files. Environment variables and
YAML files are read by [`donfig`](https://donfig.readthedocs.io/), so zarr uses
donfig's [standard search
locations](https://donfig.readthedocs.io/en/latest/configuration.html#yaml-files),
in increasing order of precedence:

- `/etc/zarr/` (override this directory with the `ZARR_ROOT_CONFIG`
  environment variable),
- `<sys.prefix>/etc/zarr/` and each entry in Python's `site.PREFIXES` (e.g.
  inside a virtual environment),
- `~/.config/zarr/`,
- the path in the `ZARR_CONFIG` environment variable, which may point at a
  single file or a directory and takes precedence over all of the above.

Place a `zarr.yaml` in any of these directories, or point `ZARR_CONFIG` at a
specific file. YAML files contain configuration fields only: environment
controls such as `benchmark_clear_cache` are not valid YAML config keys.
Unrecognized keys are ignored with a warning during collection; recognized
values are checked when the typed configuration is created.

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

For selecting custom implementations of codecs, pipelines, buffers and ndbuffers,
first register the implementations in the registry and then select them in the config.
For example, an implementation of the bytes codec in a class `'custompackage.NewBytesCodec'`,
requires the value of `codecs.bytes` to be `'custompackage.NewBytesCodec'`.

This is the current default configuration:

```python exec="true" session="config" source="above" result="ansi"
from pprint import pprint
pprint(zarr.config.to_dict())
```
