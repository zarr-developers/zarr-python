# Experimental features

This section contains documentation for experimental Zarr Python features. The features described here are exciting and potentially useful, but also volatile -- we might change them at any time. Take this into account if you consider depending on these features.

## `CacheStore`

Zarr Python 3.1.4 adds [`zarr.experimental.cache_store.CacheStore`][] provides a dual-store caching implementation
that can be wrapped around any Zarr store to improve performance for repeated data access.
This is particularly useful when working with remote stores (e.g., S3, HTTP) where network
latency can significantly impact data access speed.

The CacheStore implements a cache that uses a separate Store instance as the cache backend,
providing persistent caching capabilities with time-based expiration, size-based eviction,
and flexible cache storage options. It automatically evicts the least recently used items
when the cache reaches its maximum size.

Because the `CacheStore` uses an ordinary Zarr `Store` object as the caching layer, you can reuse the data stored in the cache later.

> **Note:** The CacheStore is a wrapper store that maintains compatibility with the full
> `zarr.abc.store.Store` API while adding transparent caching functionality.

## Basic Usage

Creating a CacheStore requires both a source store and a cache store. The cache store
can be any Store implementation, providing flexibility in cache persistence:

```python exec="true" session="experimental" source="above"
import zarr
from zarr.storage import LocalStore
import numpy as np
from tempfile import mkdtemp
from zarr.experimental.cache_store import CacheStore

# Create a local store and a separate cache store
local_store_path = mkdtemp(suffix='.zarr')
source_store = LocalStore(local_store_path)
cache_store = zarr.storage.MemoryStore()  # In-memory cache
cached_store = CacheStore(
    store=source_store,
    cache_store=cache_store,
    max_size=256*1024*1024  # 256MB cache
)

# Create an array using the cached store
zarr_array = zarr.zeros((100, 100), chunks=(10, 10), dtype='f8', store=cached_store, mode='w')

# Write some data to force chunk creation
zarr_array[:] = np.random.random((100, 100))
```

The dual-store architecture allows you to use different store types for source and cache,
such as a remote store for source data and a local store for persistent caching.

## Performance Benefits

The CacheStore provides significant performance improvements for repeated data access:

```python exec="true" session="experimental" source="above" result="ansi"
import time

# Benchmark reading with cache
start = time.time()
for _ in range(100):
    _ = zarr_array[:]
elapsed_cache = time.time() - start

# Compare with direct store access (without cache)
zarr_array_nocache = zarr.open(local_store_path, mode='r')
start = time.time()
for _ in range(100):
    _ = zarr_array_nocache[:]
elapsed_nocache = time.time() - start

# Cache provides speedup for repeated access
speedup = elapsed_nocache / elapsed_cache
print(f"Speedup is {speedup}")
```

Cache effectiveness is particularly pronounced with repeated access to the same data chunks.


## Cache Configuration

The CacheStore can be configured with several parameters:

**max_size**: Controls the maximum size of cached data in bytes

```python exec="true" session="experimental" source="above"
# 256MB cache with size limit
cache = CacheStore(
    store=source_store,
    cache_store=cache_store,
    max_size=256*1024*1024
)

# Unlimited cache size (use with caution)
cache = CacheStore(
    store=source_store,
    cache_store=cache_store,
    max_size=None
)
```

**max_age_seconds**: Controls time-based cache expiration

```python exec="true" session="experimental" source="above"
# Cache expires after 1 hour
cache = CacheStore(
    store=source_store,
    cache_store=cache_store,
    max_age_seconds=3600
)

# Cache never expires
cache = CacheStore(
    store=source_store,
    cache_store=cache_store,
    max_age_seconds="infinity"
)
```

**cache_set_data**: Controls whether written data is cached

```python exec="true" session="experimental" source="above" result="ansi"
# Cache data when writing (default)
cache = CacheStore(
    store=source_store,
    cache_store=cache_store,
    cache_set_data=True
)

# Don't cache written data (read-only cache)
cache = CacheStore(
    store=source_store,
    cache_store=cache_store,
    cache_set_data=False
)
```

## Cache Statistics

The CacheStore provides statistics to monitor cache performance and state:

```python exec="true" session="experimental" source="above" result="ansi"
# Access some data to generate cache activity
data = zarr_array[0:50, 0:50]  # First access - cache miss
data = zarr_array[0:50, 0:50]  # Second access - cache hit

# Get comprehensive cache information
info = cached_store.cache_info()
print(info['cache_store_type'])  # e.g., 'MemoryStore'
print(info['max_age_seconds'])
print(info['max_size'])
print(info['current_size'])
print(info['tracked_keys'])
print(info['cached_keys'])
print(info['cache_set_data'])
```

The `cache_info()` method returns a dictionary with detailed information about the cache state.

## Cache Management

The CacheStore provides methods for manual cache management:

```python exec="true" session="experimental" source="above"
# Clear all cached data and tracking information
import asyncio
asyncio.run(cached_store.clear_cache())

# Check cache info after clearing
info = cached_store.cache_info()
assert info['tracked_keys'] == 0
assert info['current_size'] == 0
```

The `clear_cache()` method is an async method that clears both the cache store
(if it supports the `clear` method) and all internal tracking data.

## Best Practices

1. **Choose appropriate cache store**: Use MemoryStore for fast temporary caching or LocalStore for persistent caching
2. **Size the cache appropriately**: Set `max_size` based on available storage and expected data access patterns
3. **Use with remote stores**: The cache provides the most benefit when wrapping slow remote stores
4. **Monitor cache statistics**: Use `cache_info()` to tune cache size and access patterns
5. **Consider data locality**: Group related data accesses together to improve cache efficiency
6. **Set appropriate expiration**: Use `max_age_seconds` for time-sensitive data or "infinity" for static data

## Working with Different Store Types

The CacheStore can wrap any store that implements the `zarr.abc.store.Store` interface
and use any store type for the cache backend:

### Local Store with Memory Cache

```python exec="true" session="experimental-memory-cache" source="above"
from zarr.storage import LocalStore, MemoryStore
from zarr.experimental.cache_store import CacheStore
from tempfile import mkdtemp

local_store_path = mkdtemp(suffix='.zarr')
source_store = LocalStore(local_store_path)
cache_store = MemoryStore()
cached_store = CacheStore(
    store=source_store,
    cache_store=cache_store,
    max_size=128*1024*1024
)
```

### Memory Store with Persistent Cache

```python exec="true" session="experimental-local-cache" source="above"
from tempfile import mkdtemp
from zarr.storage import MemoryStore, LocalStore
from zarr.experimental.cache_store import CacheStore

memory_store = MemoryStore()
local_store_path = mkdtemp(suffix='.zarr')
persistent_cache = LocalStore(local_store_path)
cached_store = CacheStore(
    store=memory_store,
    cache_store=persistent_cache,
    max_size=256*1024*1024
)
```

The dual-store architecture provides flexibility in choosing the best combination
of source and cache stores for your specific use case.

## Examples from Real Usage

Here's a complete example demonstrating cache effectiveness:

```python exec="true" session="experimental-final" source="above" result="ansi"
import numpy as np
import time
from tempfile import mkdtemp
import zarr
import zarr.storage
from zarr.experimental.cache_store import CacheStore

# Create test data with dual-store cache
local_store_path = mkdtemp(suffix='.zarr')
source_store = zarr.storage.LocalStore(local_store_path)
cache_store = zarr.storage.MemoryStore()
cached_store = CacheStore(
    store=source_store,
    cache_store=cache_store,
    max_size=256*1024*1024
)
zarr_array = zarr.zeros((100, 100), chunks=(10, 10), dtype='f8', store=cached_store, mode='w')
zarr_array[:] = np.random.random((100, 100))

# Demonstrate cache effectiveness with repeated access
start = time.time()
data = zarr_array[20:30, 20:30]  # First access (cache miss)
first_access = time.time() - start
print(f"First access took {first_access}")

start = time.time()
data = zarr_array[20:30, 20:30]  # Second access (cache hit)
second_access = time.time() - start
print(f"Second access took {second_access}")

# Check cache statistics
info = cached_store.cache_info()
assert info['cached_keys'] > 0  # Should have cached keys
assert info['current_size'] > 0  # Should have cached data
print(f"Cache contains {info['cached_keys']} keys with {info['current_size']} bytes")
```

This example shows how the CacheStore can significantly reduce access times for repeated
data reads, particularly important when working with remote data sources. The dual-store
architecture allows for flexible cache persistence and management.
