# zarr-storage

An extraction of Zarr-Python's existing storage layer into the `legacy`
namespace, with concrete stores, wrappers, and reusable conformance tests.

```python
from zarr.core.buffer import default_buffer_prototype
from zarr_storage.legacy import LatencyStore, MemoryStore, WrapperStore

store = LatencyStore(MemoryStore(), get_latency=0.01)
```

## Included APIs

- The `Store` ABC, byte requests, byte getter/setter protocols, and sync capabilities.
- `MemoryStore`, `ManagedMemoryStore`, `GpuMemoryStore`, `LocalStore`, `ZipStore`,
  `FsspecStore`, and `ObjectStore`.
- `WrapperStore`, `LoggingStore`, and `LatencyStore`.
- `StorePath`, store construction/path utilities, and byte-range coalescing.
- Experimental `CacheStore` at `zarr_storage.legacy.experimental.cache_store`.
- `zarr_storage.testing.StoreTests`, state machines, strategies, buffer fixtures,
  and assertion helpers for third-party implementations.

`LatencyStore` is available without pytest. The conformance utilities are optional:
install `zarr-storage[testing]` to use them. Install `zarr-storage[remote]` for
fsspec and obstore backends. GPU execution additionally requires a suitable CuPy
installation and hardware, as it does in Zarr-Python.

## Status and compatibility

This is an experimental extraction draft. The source comes from Zarr-Python's
`src/zarr/abc/store.py`, `src/zarr/storage`, and storage-related testing utilities
at commit `9c29a0da9`. Imports are redirected to the extracted implementations;
legacy signatures, inherited behavior, and async execution are preserved.
Future APIs can coexist under a different namespace. No new storage contract or
deprecation is introduced here.

**There is still a runtime dependency on `zarr`.** Shared buffers, configuration,
concurrency, metadata-aware IO helpers, and sync utilities remain there. This
package must not become a dependency of `zarr` until that dependency is removed.
The tested compatibility baseline is this source checkout, not every published
release in the declared dependency range.

The extracted classes have distinct identities. Importing this package does not
change Zarr's imports or make its array entry points accept these stores. At
runtime adoption, the old Zarr import paths should re-export one canonical
implementation. Until then, use these stores through their storage APIs.

## Tests

The package includes the full `tests/test_store` suite and the experimental
cache-store suite, redirected to the extracted implementations. The original
Zarr tests remain in place. Tests cover sync/async IO, ranges, listing, lifecycle,
read-only behavior, pickling, wrappers, caching, and array/group integration.

The integration tests have an explicit, test-only fixture that rebinds Zarr's
storage references to the extracted implementations, simulating future
re-exports. They use real stores and real arrays; no IO methods are mocked by
that fixture. Bindings are restored after each test. Import-isolation tests
separately verify that normal package imports leave Zarr unchanged.

From the repository root:

```sh
# Core stores; optional-backend and GPU tests skip when dependencies are absent.
PYTHONPATH=packages/zarr-storage/src hatch run test.py3.12-minimal:pytest packages/zarr-storage/tests

# Includes fsspec/obstore tests and a local moto S3 server.
PYTHONPATH=packages/zarr-storage/src hatch run test.py3.12-optional:pytest packages/zarr-storage/tests

# Stateful tests retain the upstream opt-in flag.
PYTHONPATH=packages/zarr-storage/src hatch run test.py3.12-minimal:pytest packages/zarr-storage/tests/test_store/test_stateful.py --run-slow-hypothesis
```

Unsupported backend operations retain their upstream skips/xfails. The standalone
package suite uses a 50-example Hypothesis profile with no deadline; select a
registered profile via `HYPOTHESIS_PROFILE` to override it.

Build from this directory with `hatch build`. CI runs source and wheel tests on
Python 3.12/3.14, with an additional optional-backend job on Python 3.12.
