# zarr-storage

An initial extraction of the existing Zarr-Python storage interfaces:
Store, byte requests, byte getters/setters, and sync capability protocols.

```python
from zarr_storage.legacy import Store
```

## Status and scope

This is an experimental extraction draft, not a replacement Zarr backend.
The `legacy` namespace reproduces `src/zarr/abc/store.py` from Zarr-Python commit
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

- Buffer and BufferPrototype, including default-buffer registry selection.
- Configuration and concurrent_map for inherited operations.

The range-coalescing helper is local because it must recognize this namespace's byte-request classes.

## Development

From the repository root, expose this draft package and use a Hatch test environment:

```sh
PYTHONPATH=packages/zarr-storage/src hatch run test.py3.12-minimal:pytest packages/zarr-storage/tests --import-mode=importlib
```

The contract tests compare the draft against the source checkout. Behavioral
tests exercise third-party-style subclasses, not only copied signatures.

Build this distribution from its directory with `hatch build`. API documentation
is in `docs/api/index.md`; inherited docstrings remain in the extracted module.
