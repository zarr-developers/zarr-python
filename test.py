import zarr
import zarr.storage
import time
import numpy as np
import os
from zarr.storage._dual_cache import CacheStore
from zarr.storage import MemoryStore, FsspecStore

# Example 1: Local store benchmark
print("=== Local Store Benchmark ===")
local_store = zarr.storage.LocalStore('test.zarr')
# Use MemoryStore as cache backend with CacheStore
cache_backend = MemoryStore()
cache = CacheStore(local_store, cache_store=cache_backend)

# Create array with zeros (fill_value=0), then write non-zero data to force chunk creation
zarr_array = zarr.zeros((100, 100), chunks=(10, 10), dtype='f8', store=cache, mode='w')

# Force the data to be written by writing non-fill-value data to all chunks
print("Writing random data to all chunks...")
zarr_array[:] = np.random.random((100, 100))  # This forces all chunks to be materialized and written

print(f"Chunks created in test.zarr: {os.listdir('test.zarr')}")
if 'c' in os.listdir('test.zarr'):
    chunk_files = os.listdir('test.zarr/c')
    print(f"Number of chunk files: {len(chunk_files)}")
    print(f"Sample chunk files: {chunk_files[:5]}")  # Show first 5

# Read benchmark with cache
start = time.time()
for _ in range(100):
    _ = zarr_array[:]
elapsed_cache = time.time() - start

# Read benchmark without cache
zarr_array_nocache = zarr.open('test.zarr', mode='r')
start = time.time()
for _ in range(100):
    _ = zarr_array_nocache[:]
elapsed_nocache = time.time() - start

print(f"Read time with CacheStore: {elapsed_cache:.4f} s")
print(f"Read time without cache: {elapsed_nocache:.4f} s")
print(f"Speedup: {elapsed_nocache/elapsed_cache:.2f}x\n")

###############################################

# Example 2: Remote store (with error handling)
print("=== Remote Store Benchmark ===")
import gcsfs
import zarr

# Use Google Cloud Storage filesystem
gcs = gcsfs.GCSFileSystem(token='anon', asynchronous=True)  # anonymous access with async support
gcs_path = 'ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/A186/lung-right/4.26um_VOI-3_bm18.ome.zarr/6'

# Wrap with zarr's FsspecStore to make it v3 compatible
store = FsspecStore(gcs, path=gcs_path)

# Use MemoryStore as cache backend with CacheStore
remote_cache_backend = MemoryStore()
cache = CacheStore(store, cache_store=remote_cache_backend)

try:
    # Open the zarr array directly since this appears to be a zarr array path
    z = zarr.open(cache)
    print(f"Array info - Shape: {z.shape}, dtype: {z.dtype}")

    # Benchmark reading with cache
    print("Benchmarking reads with CacheStore...")
    start = time.time()
    for _ in range(10):  # Fewer iterations for remote access
        _ = z[0:10, 0:10, 0:10]  # Read a small chunk
    elapsed_cache = time.time() - start

    # Benchmark reading without cache (direct store access)
    print("Benchmarking reads without cache...")
    z_nocache = zarr.open(store)  # Direct store without cache
    start = time.time()
    for _ in range(10):  # Same number of iterations
        _ = z_nocache[0:10, 0:10, 0:10]  # Read the same small chunk
    elapsed_nocache = time.time() - start

    print(f"Read time with CacheStore: {elapsed_cache:.4f} s")
    print(f"Read time without cache: {elapsed_nocache:.4f} s")
    print(f"Speedup: {elapsed_nocache/elapsed_cache:.2f}x")
    
    # Test cache effectiveness with repeated access
    print("\nTesting cache effectiveness...")
    print("First access (from remote):")
    start = time.time()
    _ = z[20:30, 20:30, 20:30]
    first_access = time.time() - start
    
    print("Second access (from cache):")
    start = time.time()
    _ = z[20:30, 20:30, 20:30]  # Same chunk should be cached
    second_access = time.time() - start
    
    print(f"First access time: {first_access:.4f} s")
    print(f"Second access time: {second_access:.4f} s")
    print(f"Cache speedup: {first_access/second_access:.2f}x")
except Exception as e:
    print(f"Error accessing zarr array: {e}")
    print("This might be a group - trying to list contents...")
    try:
        # Try opening as group without specifying mode
        root = zarr.open_group(store=cache)
        print(f"Available arrays/groups: {list(root.keys())}")
    except Exception as e2:
        print(f"Error accessing as group: {e2}")
        # If still failing, try direct store access
        try:
            print("Trying direct store listing...")
            # List keys directly from the store
            keys = list(store.keys())
            print(f"Store keys: {keys[:10]}...")  # Show first 10 keys
        except Exception as e3:
            print(f"Direct store access failed: {e3}")
