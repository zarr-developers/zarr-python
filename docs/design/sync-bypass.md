# Design: Fully Synchronous Read/Write Bypass

## Problem

Zarr-python's read/write path is inherently async: every `Array.__getitem__`
or `Array.__setitem__` call passes through several layers of async machinery
before any actual work happens. For workloads where both the codec chain and
the store are fundamentally synchronous (e.g. gzip + MemoryStore, or
zstd + LocalStore), this async overhead dominates latency.

The call chain looks like this:

```
Array.__getitem__
  └─ sync()                          # (1) thread hop: submits coroutine to background event loop
       └─ AsyncArray._get_selection  #     runs on the event loop thread
            └─ CodecPipeline.read    #     async pipeline
                 ├─ concurrent_map   # (2) launches tasks on event loop
                 │    └─ ByteGetter.get(prototype)   # (3) async store IO
                 │         └─ MemoryStore.get()       #     just a dict lookup!
                 └─ codec.decode()
                      └─ asyncio.to_thread(...)       # (4) thread hop for CPU work
                           └─ gzip.decompress(...)    #     actual compute
```

There are four sources of overhead, marked (1)-(4):

1. **`sync()` bridge**: Every synchronous `Array` method calls `sync()`, which
   uses `asyncio.run_coroutine_threadsafe()` to submit work to a background
   event loop thread. Even when the coroutine does zero awaiting, this costs
   ~30-50us for the round-trip through the event loop.

2. **`concurrent_map` batching**: The pipeline groups chunks into batches and
   dispatches them via `concurrent_map`, which creates asyncio tasks. For
   single-chunk reads (the common case), this is pure overhead.

3. **Async store IO**: `StorePath.get()` / `StorePath.set()` are `async def`.
   For `MemoryStore` (a dict lookup) and `LocalStore` (a file read), the
   underlying operation is synchronous — wrapping it in `async def` forces an
   unnecessary context switch through the event loop.

4. **`asyncio.to_thread` for codec compute**: `BatchedCodecPipeline` runs each
   codec's encode/decode in `asyncio.to_thread()`, adding another thread hop.
   `SyncCodecPipeline` (the foundation this work builds on) already eliminates
   this by calling `_decode_sync` / `_encode_sync` inline.

The net effect: a MemoryStore read of a single small chunk spends more time
in async machinery than in actual decompression.


## Solution

When the codec pipeline and store both support synchronous operation, bypass
the event loop entirely: run IO, codec compute, and buffer scatter all on the
calling thread, with zero async overhead.

The solution has three layers:

### Layer 1: Sync Store IO

Add `supports_sync`, `get_sync()`, `set_sync()`, and `delete_sync()` to the
store abstraction. These are opt-in: the `Store` ABC provides default
implementations that raise `NotImplementedError`, and only stores with native
sync capabilities override them.

```
Store ABC (defaults: supports_sync=False, methods raise NotImplementedError)
  ├── MemoryStore  (supports_sync=True, direct dict access)
  ├── LocalStore   (supports_sync=True, direct file IO via _get/_put)
  └── FsspecStore  (unchanged, remains async-only)

StorePath delegates to its underlying Store:
  get_sync()  →  self.store.get_sync(self.path, ...)
  set_sync()  →  self.store.set_sync(self.path, ...)
```

**Key decision**: `StorePath` is what gets passed to the codec pipeline as a
`ByteGetter` / `ByteSetter`. By adding sync methods to `StorePath`, the
pipeline can call them directly without knowing the concrete store type.

**Protocol gap**: The `ByteGetter` / `ByteSetter` protocols only define async
methods (`get`, `set`, `delete`). Rather than modifying these widely-used
protocols, the sync pipeline methods use `Any` type annotations for the
byte_getter/byte_setter parameters and call `.get_sync()` etc. at runtime.
This is a pragmatic tradeoff: the sync path is an optimization that only
activates when `supports_sync` is True, so the runtime type is always a
`StorePath` that has these methods.

### Layer 2: Sync Codec Pipeline IO

Add `supports_sync_io`, `read_sync()`, and `write_sync()` to the
`CodecPipeline` ABC (non-abstract, default raises `NotImplementedError`).

`SyncCodecPipeline` implements these with a simple sequential loop:

```python
# read_sync: for each chunk
for byte_getter, chunk_spec, chunk_sel, out_sel, _ in batch_info:
    chunk_bytes = byte_getter.get_sync(prototype=chunk_spec.prototype)  # sync IO
    chunk_array = self._decode_one(chunk_bytes, ...)                    # sync compute
    out[out_selection] = chunk_array[chunk_selection]                   # scatter
```

No batching, no `concurrent_map`, no event loop — just a Python for-loop.

**Sharding fallback**: When `supports_partial_decode` is True (i.e. the codec
pipeline uses sharding), `supports_sync_io` returns False and the Array falls
back to the standard `sync()` path. This is because `ShardingCodec`'s
`decode_partial` is async (it reads sub-ranges from the store) and does not
have a sync equivalent.

### Layer 3: Array Bypass

Each of the 10 sync `Array` selection methods (5 getters, 5 setters) gains a
fast path:

```python
def get_basic_selection(self, selection, *, out=None, prototype=None, fields=None):
    indexer = BasicIndexer(selection, self.shape, self.metadata.chunk_grid)
    if self._can_use_sync_path():
        return _get_selection_sync(
            self.async_array.store_path, self.async_array.metadata,
            self.async_array.codec_pipeline, self.async_array.config,
            indexer, out=out, fields=fields, prototype=prototype,
        )
    return sync(self.async_array._get_selection(indexer, ...))
```

`_can_use_sync_path()` checks three conditions:
1. The codec pipeline supports sync IO (`supports_sync_io`)
2. No partial decode is active (rules out sharding)
3. The store supports sync (`supports_sync`)

When all three hold, `_get_selection_sync` / `_set_selection_sync` run the
entire operation on the calling thread. These functions mirror the async
`_get_selection` / `_set_selection` exactly, but call `codec_pipeline.read_sync()`
/ `write_sync()` instead of `await codec_pipeline.read()` / `write()`.


## Resulting Call Chain

With the sync bypass active, the call chain becomes:

```
Array.__getitem__
  └─ _get_selection_sync             # runs on calling thread
       └─ SyncCodecPipeline.read_sync
            ├─ StorePath.get_sync    # direct dict/file access, no event loop
            ├─ _decode_one           # inline codec chain, no to_thread
            └─ out[sel] = array      # scatter into output
```

No `sync()`, no event loop, no `asyncio.to_thread`, no `concurrent_map`.


## Additional Optimization: Codec Instance Caching

`GzipCodec` was creating a new `GZip(level)` instance on every encode/decode
call. `ZstdCodec` and `BloscCodec` already cache their codec instances via
`@cached_property`. We apply the same pattern to `GzipCodec`:

```python
@cached_property
def _gzip_codec(self) -> GZip:
    return GZip(self.level)
```

This is safe because `GzipCodec` is a frozen dataclass — `level` never
changes after construction, so the cached instance is always valid.


## What Stays Unchanged

- **`BatchedCodecPipeline`**: Unmodified. It inherits the default
  `supports_sync_io=False` from the ABC.
- **Remote stores** (`FsspecStore`): `supports_sync` stays `False`. All
  remote IO remains async.
- **Sharded arrays**: Fall back to the `sync()` path because
  `supports_partial_decode` is True.
- **All async APIs**: `AsyncArray`, `async def read/write`, etc. are
  completely untouched. The sync bypass is an optimization of the
  synchronous `Array` class only.


## Files Modified

| File | Layer | Change |
|------|-------|--------|
| `src/zarr/abc/store.py` | 1 | `supports_sync`, `get_sync`, `set_sync`, `delete_sync` on `Store` ABC |
| `src/zarr/storage/_memory.py` | 1 | Sync store methods (direct dict access) |
| `src/zarr/storage/_local.py` | 1 | Sync store methods (direct `_get`/`_put` calls) |
| `src/zarr/storage/_common.py` | 1 | Sync methods on `StorePath` (delegates to store) |
| `src/zarr/abc/codec.py` | 2 | `supports_sync_io`, `read_sync`, `write_sync` on `CodecPipeline` ABC |
| `src/zarr/experimental/sync_codecs.py` | 2 | `read_sync`, `write_sync` implementation |
| `src/zarr/core/array.py` | 3 | `_can_use_sync_path`, `_get_selection_sync`, `_set_selection_sync`, 10 method modifications |
| `src/zarr/codecs/gzip.py` | — | `@cached_property` for GZip instance |


## Design Tradeoffs

**Duplication of `_get_selection` / `_set_selection`**: The sync versions
(`_get_selection_sync`, `_set_selection_sync`) duplicate the setup logic
(dtype resolution, buffer creation, value coercion) from the async originals.
This is intentional: extracting shared helpers would add complexity and
indirection to the hot path for no functional benefit. The two versions
should be kept in sync manually.

**Sequential chunk processing**: `read_sync` and `write_sync` process chunks
sequentially in a for-loop, with no parallelism. For the target use case
(MemoryStore, LocalStore), this is optimal: MemoryStore is a dict lookup
(~1us), LocalStore is a file read that benefits from OS page cache, and
Python's GIL prevents true parallelism for CPU-bound codec work anyway. The
async path with `concurrent_map` is better for remote stores where IO latency
can be overlapped.

**`Any` type annotations**: The `read_sync` and `write_sync` methods on
`SyncCodecPipeline` use `Any` for the byte_getter/byte_setter type in the
`batch_info` tuples. This avoids modifying the `ByteGetter`/`ByteSetter`
protocols, which are public API. The runtime type is always `StorePath`, which
has the sync methods; the type system just can't express this constraint
through the existing protocol hierarchy.

**No sync partial decode/encode**: Sharding's `decode_partial` /
`encode_partial` methods are inherently async (they issue byte-range reads to
the store). Rather than adding sync variants to the sharding codec (which
would require significant refactoring), we simply fall back to the `sync()`
path for sharded arrays. This is the right tradeoff because sharded arrays
typically involve remote stores where async IO is beneficial anyway.
