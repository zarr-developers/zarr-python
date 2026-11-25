# zarr-codec

An initial extraction of the existing Zarr-Python codec interfaces:
codec base classes, partial-IO mixins, sync capability protocols, and CodecPipeline.

```python
from zarr_codec.legacy import BytesBytesCodec
```

## Status and scope

This is an experimental extraction draft, not a replacement Zarr backend.
The `legacy` namespace reproduces `src/zarr/abc/codec.py` from Zarr-Python commit
`1b16efee6`. Signatures, docstrings, inherited implementations, and async
behavior are preserved. New APIs can be developed under a different namespace;
none are introduced here. Built-in concrete stores/codecs remain in `zarr`.

**There is still a runtime dependency on `zarr`.** The legacy interfaces use
its shared foundations. This package must not become a dependency of `zarr`
until that dependency is removed. The tested compatibility baseline is the
source checkout at the commit above; the dependency range is not a claim that
every release in that range has been tested.

These are independent class definitions, not aliases to Zarr's classes.
Existing Zarr entry points do not yet recognize them as their own nominal
base classes. Do not switch an existing extension to this namespace and expect
it to plug into Zarr before runtime integration lands. At that point, the old
Zarr import paths should re-export one canonical definition, preserving class
identity. There are no deprecations or changes to Zarr's runtime in this draft.

## Remaining extraction dependencies

- Metadata and its recursive serialization behavior.
- Buffer and NDBuffer, including runtime generic type bounds.
- NamedConfig, configuration, and concurrent_map for batching.
- Annotation dependencies: ArraySpec, dtype classes, metadata, indexing, and store interfaces.

The global config remains shared with Zarr, preserving concurrency behavior.
Replacing these dependencies with smaller protocols would be an API design
change and is deliberately deferred.

## Development

From the repository root, expose this draft package and use a Hatch test environment:

```sh
PYTHONPATH=packages/zarr-codec/src hatch run test.py3.12-minimal:pytest packages/zarr-codec/tests --import-mode=importlib
```

The contract tests compare the draft against the source checkout. Behavioral
tests exercise third-party-style subclasses, not only copied signatures.

Build this distribution from its directory with `hatch build`. API documentation
is in `docs/api/index.md`; inherited docstrings remain in the extracted module.
