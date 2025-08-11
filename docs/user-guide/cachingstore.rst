.. only:: doctest

   >>> import shutil
   >>> shutil.rmtree('test.zarr', ignore_errors=True)

.. _user-guide-cachestore:

CacheStore guide
================

The :class:`zarr.storage.CacheStore` provides a dual-store caching implementation
that can be wrapped around any Zarr store to improve performance for repeated data access.
This is particularly useful when working with remote stores (e.g., S3, HTTP) where network
latency can significantly impact data access speed.

The CacheStore implements a cache that uses a separate Store instance as the cache backend,
providing persistent caching capabilities with time-based expiration, size-based eviction,
and flexible cache storage options. It automatically evicts the least recently used items
when the cache reaches its maximum size.

.. note::
   The CacheStore is a wrapper store that maintains compatibility with the full
   :class:`zarr.abc.store.Store` API while adding transparent caching functionality.

Basic Usage
-----------

Creating a CacheStore requires both a source store and a cache store. The cache store
can be any Store implementation, providing flexibility in cache persistence:

   >>> import zarr
   >>> import zarr.storage
   >>> import numpy as np
   >>>
   >>> # Create a local store and a separate cache store
   >>> source_store = zarr.storage.LocalStore('test.zarr')
   >>> cache_store = zarr.storage.MemoryStore()  # In-memory cache
   >>> cached_store = zarr.storage.CacheStore(
   ...     store=source_store,
   ...     cache_store=cache_store,
   ...     max_size=256*1024*1024  # 256MB cache
   ... )
   >>>
   >>> # Create an array using the cached store
   >>> zarr_array = zarr.zeros((100, 100), chunks=(10, 10), dtype='f8', store=cached_store, mode='w')
   >>>
   >>> # Write some data to force chunk creation
   >>> zarr_array[:] = np.random.random((100, 100))

The dual-store architecture allows you to use different store types for source and cache,
such as a remote store for source data and a local store for persistent caching.

Performance Benefits
--------------------

The CacheStore provides significant performance improvements for repeated data access:

   >>> import time
   >>>
   >>> # Benchmark reading with cache
   >>> start = time.time()
   >>> for _ in range(100):
   ...     _ = zarr_array[:]
   >>> elapsed_cache = time.time() - start
   >>>
   >>> # Compare with direct store access (without cache)
   >>> zarr_array_nocache = zarr.open('test.zarr', mode='r')
   >>> start = time.time()
   >>> for _ in range(100):
   ...     _ = zarr_array_nocache[:]
   >>> elapsed_nocache = time.time() - start
   >>>
   >>> # Cache provides speedup for repeated access
   >>> speedup = elapsed_nocache / elapsed_cache  # doctest: +SKIP

Cache effectiveness is particularly pronounced with repeated access to the same data chunks.

Remote Store Caching
--------------------

The CacheStore is most beneficial when used with remote stores where network latency
is a significant factor. You can use different store types for source and cache:

   >>> from zarr.storage import FsspecStore, LocalStore
   >>>
   >>> # Create a remote store (S3 example) - for demonstration only
   >>> remote_store = FsspecStore.from_url('s3://bucket/data.zarr', storage_options={'anon': True})  # doctest: +SKIP
   >>>
   >>> # Use a local store for persistent caching  
   >>> local_cache_store = LocalStore('cache_data')  # doctest: +SKIP
   >>>
   >>> # Create cached store with persistent local cache
   >>> cached_store = zarr.storage.CacheStore(  # doctest: +SKIP
   ...     store=remote_store,
   ...     cache_store=local_cache_store,
   ...     max_size=512*1024*1024  # 512MB cache
   ... )
   >>>
   >>> # Open array through cached store  
   >>> z = zarr.open(cached_store)  # doctest: +SKIP

The first access to any chunk will be slow (network retrieval), but subsequent accesses
to the same chunk will be served from the local cache, providing dramatic speedup.
The cache persists between sessions when using a LocalStore for the cache backend.

Cache Configuration
-------------------

The CacheStore can be configured with several parameters:

**max_size**: Controls the maximum size of cached data in bytes

   >>> # 256MB cache with size limit
   >>> cache = zarr.storage.CacheStore(
   ...     store=source_store,
   ...     cache_store=cache_store,
   ...     max_size=256*1024*1024
   ... )
   >>>
   >>> # Unlimited cache size (use with caution)
   >>> cache = zarr.storage.CacheStore(
   ...     store=source_store,
   ...     cache_store=cache_store,
   ...     max_size=None
   ... )

**max_age_seconds**: Controls time-based cache expiration

   >>> # Cache expires after 1 hour
   >>> cache = zarr.storage.CacheStore(
   ...     store=source_store,
   ...     cache_store=cache_store,
   ...     max_age_seconds=3600
   ... )
   >>>
   >>> # Cache never expires
   >>> cache = zarr.storage.CacheStore(
   ...     store=source_store,
   ...     cache_store=cache_store,
   ...     max_age_seconds="infinity"
   ... )

**cache_set_data**: Controls whether written data is cached

   >>> # Cache data when writing (default)
   >>> cache = zarr.storage.CacheStore(
   ...     store=source_store,
   ...     cache_store=cache_store,
   ...     cache_set_data=True
   ... )
   >>>
   >>> # Don't cache written data (read-only cache)
   >>> cache = zarr.storage.CacheStore(
   ...     store=source_store,
   ...     cache_store=cache_store,
   ...     cache_set_data=False
   ... )

Cache Statistics
----------------

The CacheStore provides statistics to monitor cache performance and state:

   >>> # Access some data to generate cache activity
   >>> data = zarr_array[0:50, 0:50]  # First access - cache miss
   >>> data = zarr_array[0:50, 0:50]  # Second access - cache hit
   >>>
   >>> # Get comprehensive cache information
   >>> info = cached_store.cache_info()
   >>> info['cache_store_type']  # doctest: +SKIP
   'MemoryStore'
   >>> isinstance(info['max_age_seconds'], (int, str))
   True
   >>> isinstance(info['max_size'], (int, type(None)))
   True
   >>> info['current_size'] >= 0
   True
   >>> info['tracked_keys'] >= 0
   True
   >>> info['cached_keys'] >= 0
   True
   >>> isinstance(info['cache_set_data'], bool)
   True

The `cache_info()` method returns a dictionary with detailed information about the cache state.

Cache Management
----------------

The CacheStore provides methods for manual cache management:

   >>> # Clear all cached data and tracking information
   >>> import asyncio
   >>> asyncio.run(cached_store.clear_cache())  # doctest: +SKIP
   >>>
   >>> # Check cache info after clearing
   >>> info = cached_store.cache_info()  # doctest: +SKIP
   >>> info['tracked_keys'] == 0  # doctest: +SKIP
   True
   >>> info['current_size'] == 0  # doctest: +SKIP
   True

The `clear_cache()` method is an async method that clears both the cache store
(if it supports the `clear` method) and all internal tracking data.

Best Practices
--------------

1. **Choose appropriate cache store**: Use MemoryStore for fast temporary caching or LocalStore for persistent caching
2. **Size the cache appropriately**: Set ``max_size`` based on available storage and expected data access patterns
3. **Use with remote stores**: The cache provides the most benefit when wrapping slow remote stores
4. **Monitor cache statistics**: Use `cache_info()` to tune cache size and access patterns
5. **Consider data locality**: Group related data accesses together to improve cache efficiency
6. **Set appropriate expiration**: Use `max_age_seconds` for time-sensitive data or "infinity" for static data

Working with Different Store Types
----------------------------------

The CacheStore can wrap any store that implements the :class:`zarr.abc.store.Store` interface
and use any store type for the cache backend:

Local Store with Memory Cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   >>> from zarr.storage import LocalStore, MemoryStore
   >>> source_store = LocalStore('data.zarr')
   >>> cache_store = MemoryStore()
   >>> cached_store = zarr.storage.CacheStore(
   ...     store=source_store,
   ...     cache_store=cache_store,
   ...     max_size=128*1024*1024
   ... )

Remote Store with Local Cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   >>> from zarr.storage import FsspecStore, LocalStore
   >>> remote_store = FsspecStore.from_url('s3://bucket/data.zarr', storage_options={'anon': True})  # doctest: +SKIP
   >>> local_cache = LocalStore('local_cache')  # doctest: +SKIP
   >>> cached_store = zarr.storage.CacheStore(  # doctest: +SKIP
   ...     store=remote_store,
   ...     cache_store=local_cache,
   ...     max_size=1024*1024*1024,
   ...     max_age_seconds=3600
   ... )

Memory Store with Persistent Cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   >>> from zarr.storage import MemoryStore, LocalStore
   >>> memory_store = MemoryStore()
   >>> persistent_cache = LocalStore('persistent_cache')
   >>> cached_store = zarr.storage.CacheStore(
   ...     store=memory_store,
   ...     cache_store=persistent_cache,
   ...     max_size=256*1024*1024
   ... )

The dual-store architecture provides flexibility in choosing the best combination
of source and cache stores for your specific use case.

Examples from Real Usage
------------------------

Here's a complete example demonstrating cache effectiveness:

   >>> import zarr
   >>> import zarr.storage
   >>> import time
   >>> import numpy as np
   >>>
   >>> # Create test data with dual-store cache
   >>> source_store = zarr.storage.LocalStore('benchmark.zarr')
   >>> cache_store = zarr.storage.MemoryStore()
   >>> cached_store = zarr.storage.CacheStore(
   ...     store=source_store,
   ...     cache_store=cache_store,
   ...     max_size=256*1024*1024
   ... )
   >>> zarr_array = zarr.zeros((100, 100), chunks=(10, 10), dtype='f8', store=cached_store, mode='w')
   >>> zarr_array[:] = np.random.random((100, 100))
   >>>
   >>> # Demonstrate cache effectiveness with repeated access
   >>> start = time.time()
   >>> data = zarr_array[20:30, 20:30]  # First access (cache miss)
   >>> first_access = time.time() - start
   >>>
   >>> start = time.time()
   >>> data = zarr_array[20:30, 20:30]  # Second access (cache hit)
   >>> second_access = time.time() - start
   >>>
   >>> # Check cache statistics
   >>> info = cached_store.cache_info()
   >>> info['cached_keys'] > 0  # Should have cached keys
   True
   >>> info['current_size'] > 0  # Should have cached data
   True

This example shows how the CacheStore can significantly reduce access times for repeated
data reads, particularly important when working with remote data sources. The dual-store
architecture allows for flexible cache persistence and management.

.. _Zip Store Specification: https://github.com/zarr-developers/zarr-specs/pull/311
.. _fsspec: https://filesystem-spec.readthedocs.io
