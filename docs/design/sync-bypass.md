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
# read_sync: for each chunk (non-sharded path)
for byte_getter, chunk_spec, chunk_sel, out_sel, _ in batch_info:
    chunk_bytes = byte_getter.get_sync(prototype=chunk_spec.prototype)  # sync IO
    chunk_array = self._decode_one(chunk_bytes, ...)                    # sync compute
    out[out_selection] = chunk_array[chunk_selection]                   # scatter
```

No batching, no `concurrent_map`, no event loop — just a Python for-loop.

**Sharding support**: When the pipeline uses `ShardingCodec` (i.e.
`supports_partial_decode` is True), `read_sync` delegates to
`ShardingCodec._decode_partial_sync()` instead. This method fetches
the shard index and requested chunk bytes via sync byte-range reads
(`byte_getter.get_sync()` with `RangeByteRequest`/`SuffixByteRequest`),
then decodes through the inner pipeline's `read_sync` — all on the
calling thread. See [Sync Sharding](#sync-sharding) below for details.

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

`_can_use_sync_path()` checks two conditions:
1. The codec pipeline supports sync IO (`supports_sync_io`)
2. The store supports sync (`supports_sync`)

When both hold, `_get_selection_sync` / `_set_selection_sync` run the
entire operation on the calling thread. These functions mirror the async
`_get_selection` / `_set_selection` exactly, but call `codec_pipeline.read_sync()`
/ `write_sync()` instead of `await codec_pipeline.read()` / `write()`.


## Resulting Call Chain

With the sync bypass active, the call chain for non-sharded arrays becomes:

```
Array.__getitem__
  └─ _get_selection_sync             # runs on calling thread
       └─ SyncCodecPipeline.read_sync
            ├─ StorePath.get_sync    # direct dict/file access, no event loop
            ├─ _decode_one           # inline codec chain, no to_thread
            └─ out[sel] = array      # scatter into output
```

For sharded arrays:

```
Array.__getitem__
  └─ _get_selection_sync                          # runs on calling thread
       └─ SyncCodecPipeline.read_sync
            └─ ShardingCodec._decode_partial_sync
                 ├─ StorePath.get_sync(byte_range) # sync byte-range read for shard index
                 ├─ _decode_shard_index_sync        # inline index codec chain
                 ├─ StorePath.get_sync(byte_range) # sync byte-range read per chunk
                 └─ inner_pipeline.read_sync        # inner codec chain (sync)
                      ├─ _ShardingByteGetter.get_sync  # dict lookup
                      ├─ _decode_one                    # inline codec chain
                      └─ out[sel] = array               # scatter
```

No `sync()`, no event loop, no `asyncio.to_thread`, no `concurrent_map`.


## Sync Sharding

`ShardingCodec` participates in the fully-synchronous path through sync
variants of all its methods:

**Shard index codec chain**: The index codecs (typically `BytesCodec` +
`Crc32cCodec`) are run inline via `_decode_shard_index_sync` /
`_encode_shard_index_sync`. These classify the index codecs using
`codecs_from_list`, resolve metadata forward through the chain, then
run the decode/encode in the correct order — all without constructing a
pipeline object.

**Full shard decode/encode** (`_decode_sync` / `_encode_sync`): Receives
complete shard bytes, decodes the index, then delegates to the inner
codec pipeline's `read_sync` / `write_sync` with `_ShardingByteGetter` /
`_ShardingByteSetter` (dict-backed, so "IO" is a dict lookup).

**Partial shard decode/encode** (`_decode_partial_sync` /
`_encode_partial_sync`): The partial path is where most of the IO happens —
it issues sync byte-range reads to fetch the shard index and individual
chunk data from the store. Once bytes are in memory, the inner pipeline
decodes them synchronously.

**Inner pipeline**: `ShardingCodec.codec_pipeline` is obtained via
`get_pipeline_class()`. When `SyncCodecPipeline` is configured globally,
the inner pipeline is also a `SyncCodecPipeline`, enabling recursive sync
dispatch for nested sharding.


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


## Bugfix: _decode_async Metadata Resolution

The async fallback path in `SyncCodecPipeline._decode_async()` (used when
a codec in the chain doesn't support sync) had a metadata resolution bug:
it passed the same unresolved `chunk_specs` to every codec during decode.

Size-changing codecs like `FixedScaleOffset` and `PackBits` alter the data
shape/dtype, so each codec needs specs resolved through the forward chain.
The fix resolves metadata forward (aa -> ab -> bb), records specs at each
step, then uses the correct resolved specs during reverse decode traversal.
This matches `BatchedCodecPipeline._codecs_with_resolved_metadata_batched`.


## What Stays Unchanged

- **`BatchedCodecPipeline`**: Unmodified. It inherits the default
  `supports_sync_io=False` from the ABC.
- **Remote stores** (`FsspecStore`): `supports_sync` stays `False`. All
  remote IO remains async.
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
| `src/zarr/abc/codec.py` | 2 | `_decode_sync`, `_encode_sync`, `supports_sync` on `BaseCodec`; `supports_sync_io`, `read_sync`, `write_sync` on `CodecPipeline` |
| `src/zarr/experimental/sync_codecs.py` | 2 | `read_sync`, `write_sync`, `_decode_async` metadata fix |
| `src/zarr/codecs/sharding.py` | 2 | `_decode_sync`, `_encode_sync`, `_decode_partial_sync`, `_encode_partial_sync`, shard index sync codec chain |
| `src/zarr/core/array.py` | 3 | `_can_use_sync_path`, `_get_selection_sync`, `_set_selection_sync`, 10 method modifications |
| `src/zarr/codecs/gzip.py` | — | `@cached_property` for GZip instance |
| `src/zarr/codecs/blosc.py` | — | `_decode_sync`/`_encode_sync`; `_decode_single`/`_encode_single` delegate to sync |
| `src/zarr/codecs/zstd.py` | — | `_decode_sync`/`_encode_sync`; `_decode_single`/`_encode_single` delegate to sync |
| `src/zarr/codecs/bytes.py` | — | `_decode_sync`/`_encode_sync` (was `_decode_single`/`_encode_single`) |
| `src/zarr/codecs/crc32c_.py` | — | `_decode_sync`/`_encode_sync` (was `_decode_single`/`_encode_single`) |
| `src/zarr/codecs/transpose.py` | — | `_decode_sync`/`_encode_sync`; `_decode_single`/`_encode_single` delegate to sync |
| `src/zarr/codecs/vlen_utf8.py` | — | `_decode_sync`/`_encode_sync` for `VLenUTF8Codec` and `VLenBytesCodec` |


## Performance

Benchmarks on MemoryStore with `SyncCodecPipeline` vs `BatchedCodecPipeline`:

**Non-sharded arrays** (zstd compression, 100x100 float64, 32x32 chunks):
- Single-chunk read: ~2-4x faster
- Full-array read: ~2-11x faster (varies with chunk count)
- Single-chunk write: ~2-3x faster

**Sharded arrays** (4x4 shard of 8x8 inner chunks, zstd, MemoryStore):
- Single-chunk read: ~1.5-2.5x faster
- Full-array read: ~1.5-2x faster
- Single-chunk write: ~1.3-1.6x faster
- Full-array write: ~1.3-1.5x faster

The sharded speedup is smaller because the shard index decode and
per-chunk byte-range reads add overhead that wasn't present in the
non-sharded path. Still, eliminating the event loop round-trip and
`asyncio.to_thread` for each inner chunk decode provides a meaningful
improvement.


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
protocols, which are public API. The runtime type is always `StorePath` (or
`_ShardingByteGetter`/`_ShardingByteSetter` for inner-shard access), which
has the sync methods; the type system just can't express this constraint
through the existing protocol hierarchy.

**Sync sharding — sequential chunk reads**: The sync partial decode path
fetches each chunk's bytes sequentially via `byte_getter.get_sync()` with
byte-range requests. The async path can overlap these reads via
`concurrent_map`. For MemoryStore this doesn't matter (dict lookup is ~1us).
For LocalStore, OS page cache means sequential reads are fast for warm data.
For remote stores where overlapping IO would help, `supports_sync` is False
and the async path is used automatically.

**Inline shard index codec chain**: `_decode_shard_index_sync` and
`_encode_shard_index_sync` run the index codecs (BytesCodec + Crc32cCodec)
directly rather than constructing a temporary `CodecPipeline`. This avoids
the overhead of pipeline construction for a simple two-codec chain and keeps
the sync path self-contained.
