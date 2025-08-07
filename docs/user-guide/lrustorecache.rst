.. only:: doctest

   >>> import shutil
   >>> shutil.rmtree('test.zarr', ignore_errors=True)

.. _user-guide-lrustorecache:

LRUStoreCache guide
===================

The :class:`zarr.storage.LRUStoreCache` provides a least-recently-used (LRU) cache layer
that can be wrapped around any Zarr store to improve performance for repeated data access.
This is particularly useful when working with remote stores (e.g., S3, HTTP) where network
latency can significantly impact data access speed.

The LRUStoreCache implements a cache that stores frequently accessed data chunks in memory,
automatically evicting the least recently used items when the cache reaches its maximum size.

.. note::
   The LRUStoreCache is a wrapper store that maintains compatibility with the full
   :class:`zarr.abc.store.Store` API while adding transparent caching functionality.

Basic Usage
-----------

Creating an LRUStoreCache is straightforward - simply wrap any existing store with the cache:

   >>> import zarr
   >>> import zarr.storage
   >>> import numpy as np
   >>>
   >>> # Create a local store and wrap it with LRU cache
   >>> local_store = zarr.storage.LocalStore('test.zarr')
   >>> cache = zarr.storage.LRUStoreCache(local_store, max_size=2**28)  # 256MB cache
   >>>
   >>> # Create an array using the cached store
   >>> zarr_array = zarr.zeros((100, 100), chunks=(10, 10), dtype='f8', store=cache, mode='w')
   >>> 
   >>> # Write some data to force chunk creation
   >>> zarr_array[:] = np.random.random((100, 100))

The ``max_size`` parameter controls the maximum memory usage of the cache in bytes. Set it to
``None`` for unlimited cache size (use with caution).

Performance Benefits
-------------------

The LRUStoreCache provides significant performance improvements for repeated data access:

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
   >>> print(f"Speedup: {elapsed_nocache/elapsed_cache:.2f}x")

Cache effectiveness is particularly pronounced with repeated access to the same data chunks.

Remote Store Caching
--------------------

The LRUStoreCache is most beneficial when used with remote stores where network latency
is a significant factor:

   >>> import gcsfs
   >>>
   >>> # Create a remote store (Google Cloud Storage example)
   >>> gcs = gcsfs.GCSFileSystem(token='anon')
   >>> remote_store = gcsfs.GCSMap(
   ...     root='your-bucket/data.zarr',
   ...     gcs=gcs,
   ...     check=False
   ... )
   >>>
   >>> # Wrap with LRU cache for better performance
   >>> cached_store = zarr.storage.LRUStoreCache(remote_store, max_size=2**28)
   >>>
   >>> # Open array through cached store
   >>> z = zarr.open(cached_store)

The first access to any chunk will be slow (network retrieval), but subsequent accesses
to the same chunk will be served from the local cache, providing dramatic speedup.

Cache Configuration
------------------

The LRUStoreCache can be configured with several parameters:

**max_size**: Controls the maximum memory usage of the cache in bytes

   >>> # 256MB cache
   >>> cache = zarr.storage.LRUStoreCache(store, max_size=2**28)
   >>>
   >>> # Unlimited cache size (use with caution)
   >>> cache = zarr.storage.LRUStoreCache(store, max_size=None)

**read_only**: Create a read-only cache

   >>> cache = zarr.storage.LRUStoreCache(store, max_size=2**28, read_only=True)

Cache Statistics
---------------

The LRUStoreCache provides statistics to monitor cache performance:

   >>> # Access some data to generate cache activity
   >>> data = zarr_array[0:50, 0:50]  # First access - cache miss
   >>> data = zarr_array[0:50, 0:50]  # Second access - cache hit
   >>>
   >>> print(f"Cache hits: {cache.hits}")
   >>> print(f"Cache misses: {cache.misses}")
   >>> print(f"Cache hit ratio: {cache.hits / (cache.hits + cache.misses):.2%}")

Cache Management
---------------

The cache provides methods for manual cache management:

   >>> # Clear all cached values but keep keys cache
   >>> cache.invalidate_values()
   >>>
   >>> # Clear keys cache
   >>> cache.invalidate_keys()
   >>>
   >>> # Clear entire cache
   >>> cache.invalidate()

Best Practices
--------------

1. **Size the cache appropriately**: Set ``max_size`` based on available memory and expected data access patterns
2. **Use with remote stores**: The cache provides the most benefit when wrapping slow remote stores
3. **Monitor cache statistics**: Use hit/miss ratios to tune cache size and access patterns
4. **Consider data locality**: Group related data accesses together to improve cache efficiency

Working with Different Store Types
----------------------------------

The LRUStoreCache can wrap any store that implements the :class:`zarr.abc.store.Store` interface:

Local Store Caching
~~~~~~~~~~~~~~~~~~~

   >>> local_store = zarr.storage.LocalStore('data.zarr')
   >>> cached_local = zarr.storage.LRUStoreCache(local_store, max_size=2**27)

FsSpec Store Caching
~~~~~~~~~~~~~~~~~~~~

   >>> from zarr.storage import FsspecStore
   >>> remote_store = FsspecStore.from_url('s3://bucket/data.zarr', storage_options={'anon': True})
   >>> cached_remote = zarr.storage.LRUStoreCache(remote_store, max_size=2**28)

Memory Store Caching
~~~~~~~~~~~~~~~~~~~~

   >>> from zarr.storage import MemoryStore
   >>> memory_store = MemoryStore()
   >>> cached_memory = zarr.storage.LRUStoreCache(memory_store, max_size=2**26)

.. note::
   While caching a MemoryStore may seem redundant, it can be useful for limiting memory usage
   of large in-memory datasets.

Examples from Real Usage
-----------------------

Here's a complete example demonstrating cache effectiveness:

   >>> import zarr
   >>> import zarr.storage
   >>> import time
   >>> import numpy as np
   >>>
   >>> # Create test data
   >>> local_store = zarr.storage.LocalStore('benchmark.zarr')
   >>> cache = zarr.storage.LRUStoreCache(local_store, max_size=2**28)
   >>> zarr_array = zarr.zeros((100, 100), chunks=(10, 10), dtype='f8', store=cache, mode='w')
   >>> zarr_array[:] = np.random.random((100, 100))
   >>>
   >>> # Demonstrate cache effectiveness with repeated access
   >>> print("First access (cache miss):")
   >>> start = time.time()
   >>> data = zarr_array[20:30, 20:30]
   >>> first_access = time.time() - start
   >>>
   >>> print("Second access (cache hit):")
   >>> start = time.time()
   >>> data = zarr_array[20:30, 20:30]  # Same data should be cached
   >>> second_access = time.time() - start
   >>>
   >>> print(f"First access time: {first_access:.4f} s")
   >>> print(f"Second access time: {second_access:.4f} s")
   >>> print(f"Cache speedup: {first_access/second_access:.2f}x")

This example shows how the LRUStoreCache can significantly reduce access times for repeated
data reads, particularly important when working with remote data sources.

.. _Zip Store Specification: https://github.com/zarr-developers/zarr-specs/pull/311
.. _fsspec: https://filesystem-spec.readthedocs.io
