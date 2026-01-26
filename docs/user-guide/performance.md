# Optimizing performance

## Chunk optimizations

### Chunk size and shape

In general, chunks of at least 1 megabyte (1M) uncompressed size seem to provide
better performance, at least when using the Blosc compression library.

The optimal chunk shape will depend on how you want to access the data. E.g.,
for a 2-dimensional array, if you only ever take slices along the first
dimension, then chunk across the second dimension. If you know you want to chunk
across an entire dimension you can use the full size of that dimension within the
`chunks` argument, e.g.:

```python exec="true" session="performance" source="above" result="ansi"
import zarr
z1 = zarr.create_array(store={}, shape=(10000, 10000), chunks=(100, 10000), dtype='int32')
print(z1.chunks)
```

Alternatively, if you only ever take slices along the second dimension, then
chunk across the first dimension, e.g.:

```python exec="true" session="performance" source="above" result="ansi"
z2 = zarr.create_array(store={}, shape=(10000, 10000), chunks=(10000, 100), dtype='int32')
print(z2.chunks)
```

If you require reasonable performance for both access patterns then you need to
find a compromise, e.g.:

```python exec="true" session="performance" source="above" result="ansi"
z3 = zarr.create_array(store={}, shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
print(z3.chunks)
```

If you are feeling lazy, you can let Zarr guess a chunk shape for your data by
providing `chunks='auto'`, although please note that the algorithm for guessing
a chunk shape is based on simple heuristics and may be far from optimal. E.g.:

```python exec="true" session="performance" source="above" result="ansi"
z4 = zarr.create_array(store={}, shape=(10000, 10000), chunks='auto', dtype='int32')
print(z4.chunks)
```

If you know you are always going to be loading the entire array into memory, you
can turn off chunks by providing `chunks` equal to `shape`, in which case there
will be one single chunk for the array:

```python exec="true" session="performance" source="above" result="ansi"
z5 = zarr.create_array(store={}, shape=(10000, 10000), chunks=(10000, 10000), dtype='int32')
print(z5.chunks)
```

### Sharding

If you have large arrays but need small chunks to efficiently access the data, you can
use sharding. Sharding provides a mechanism to store multiple chunks in a single
storage object or file. This can be useful because traditional file systems and object
storage systems may have performance issues storing and accessing many files.
Additionally, small files can be inefficient to store if they are smaller than the
block size of the file system.

Picking a good combination of chunk shape and shard shape is important for performance.
The chunk shape determines what unit of your data can be read independently, while the
shard shape determines what unit of your data can be written efficiently.

For an example, consider you have a 100 GB array and need to read small chunks of 1 MB.
Without sharding, each chunk would be one file resulting in 100,000 files. That can
already cause performance issues on some file systems.
With sharding, you could use a shard size of 1 GB. This would result in 1000 chunks per
file and 100 files in total, which seems manageable for most storage systems.
You would still be able to read each 1 MB chunk independently, but you would need to
write your data in 1 GB increments.

To use sharding, you need to specify the `shards` parameter when creating the array.

```python exec="true" session="performance" source="above" result="ansi"
z6 = zarr.create_array(store={}, shape=(10000, 10000, 1000), shards=(1000, 1000, 1000), chunks=(100, 100, 100), dtype='uint8')
print(z6.info)
```

`shards` can be `"auto"` as well, in which case the `array.target_shard_size_bytes` setting can be used to control the size of shards (i.e., the size of the chunks cumulatively and uncompressed within the shard will be as close to, without being bigger than, `array.target_shard_size_bytes`); otherwise, a default is used.

### Chunk memory layout

The order of bytes **within each chunk** of an array can be changed via the
`order` config option, to use either C or Fortran layout. For
multi-dimensional arrays, these two layouts may provide different compression
ratios, depending on the correlation structure within the data. E.g.:

```python exec="true" session="performance" source="above" result="ansi"
import numpy as np

a = np.arange(100000000, dtype='int32').reshape(10000, 10000).T
c = zarr.create_array(store={}, shape=a.shape, chunks=(1000, 1000), dtype=a.dtype, config={'order': 'C'})
c[:] = a
print(c.info_complete())
```

```python exec="true" session="performance" source="above" result="ansi"
with zarr.config.set({'array.order': 'F'}):
    f = zarr.create_array(store={}, shape=a.shape, chunks=(1000, 1000), dtype=a.dtype)
    f[:] = a
print(f.info_complete())

```

In the above example, Fortran order gives a better compression ratio. This is an
artificial example but illustrates the general point that changing the order of
bytes within chunks of an array may improve the compression ratio, depending on
the structure of the data, the compression algorithm used, and which compression
filters (e.g., byte-shuffle) have been applied.

### Empty chunks

It is possible to configure how Zarr handles the storage of chunks that are "empty"
(i.e., every element in the chunk is equal to the array's fill value). When creating
an array with `write_empty_chunks=False`, Zarr will check whether a chunk is empty before compression and storage. If a chunk is empty,
then Zarr does not store it, and instead deletes the chunk from storage
if the chunk had been previously stored.

This optimization prevents storing redundant objects and can speed up reads, but the cost is
added computation during array writes, since the contents of
each chunk must be compared to the fill value, and these advantages are contingent on the content of the array.
If you know that your data will form chunks that are almost always non-empty, then there is no advantage to the optimization described above.
In this case, creating an array with `write_empty_chunks=True`will instruct Zarr to write every chunk without checking for emptiness.

The following example illustrates the effect of the `write_empty_chunks` flag on
the time required to write an array with different values.:

```python exec="true" session="performance" source="above" result="ansi"
import zarr
import numpy as np
import time

def timed_write(write_empty_chunks):
    """
    Measure the time required and number of objects created when writing
    to a Zarr array with random ints or fill value.
    """
    chunks = (8192,)
    shape = (chunks[0] * 1024,)
    data = np.random.randint(0, 255, shape)
    dtype = 'uint8'
    arr = zarr.create_array(
        f'data/example-{write_empty_chunks}.zarr',
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=0,
        config={'write_empty_chunks': write_empty_chunks}
     )
    # initialize all chunks
    arr[:] = 100
    result = []
    for value in (data, arr.fill_value):
        start = time.time()
        arr[:] = value
        elapsed = time.time() - start
        result.append((elapsed, arr.nchunks_initialized))
    return result

# log results
for write_empty_chunks in (True, False):
    full, empty = timed_write(write_empty_chunks)
    print(f'\nwrite_empty_chunks={write_empty_chunks}:\n\tRandom Data: {full[0]:.4f}s, {full[1]} objects stored\n\t Empty Data: {empty[0]:.4f}s, {empty[1]} objects stored\n')
```

In this example, writing random data is slightly slower with `write_empty_chunks=True`,
but writing empty data is substantially faster and generates far fewer objects in storage.

### Changing chunk shapes (rechunking)

Coming soon.

## Parallel computing and synchronization

Zarr is designed to support parallel computing and enables concurrent reads and writes to arrays.
This section covers how to optimize Zarr's concurrency settings for different parallel computing
scenarios.

### Concurrent I/O operations

Zarr uses asynchronous I/O internally to enable concurrent reads and writes across multiple chunks.
The level of concurrency is controlled by the `async.concurrency` configuration setting, which
determines the maximum number of concurrent I/O operations.

The default value is 10, which is a conservative value. You may get improved performance by tuning
the concurrency limit. You can adjust this value based on your specific needs:

```python
import zarr

# Set concurrency for the current session
zarr.config.set({'async.concurrency': 128})

# Or use environment variable
# export ZARR_ASYNC_CONCURRENCY=128
```

Higher concurrency values can improve throughput when:
- Working with remote storage (e.g., S3, GCS) where network latency is high
- Reading/writing many small chunks in parallel
- The storage backend can handle many concurrent requests

Lower concurrency values may be beneficial when:
- Working with local storage with limited I/O bandwidth
- Memory is constrained (each concurrent operation requires buffer space)
- Using Zarr within a parallel computing framework (see below)

### Using Zarr with Dask

[Dask](https://www.dask.org/) is a popular parallel computing library that works well with Zarr for processing large arrays. When using Zarr with Dask, it's important to consider the interaction between Dask's thread pool and Zarr's concurrency settings.

**Important**: When using many Dask threads, you may need to reduce both Zarr's `async.concurrency` and `threading.max_workers` settings to avoid creating too many concurrent operations. The total number of concurrent I/O operations can be roughly estimated as:

```
total_concurrency ≈ dask_threads × zarr_async_concurrency
```

For example, if you're running Dask with 10 threads and Zarr's default concurrency of 64, you could potentially have up to 640 concurrent operations, which may overwhelm your storage system or cause memory issues.

**Recommendation**: When using Dask with many threads, configure Zarr's concurrency settings:

```python
import zarr
import dask.array as da

# If using Dask with many threads (e.g., 8-16), reduce Zarr's concurrency settings
zarr.config.set({
    'async.concurrency': 4,      # Limit concurrent async operations
    'threading.max_workers': 4,  # Limit Zarr's internal thread pool
})

# Open Zarr array
z = zarr.open_array('data/large_array.zarr', mode='r')

# Create Dask array from Zarr array
arr = da.from_array(z, chunks=z.chunks)

# Process with Dask
result = arr.mean(axis=0).compute()
```

**Configuration guidelines for Dask workloads**:

- `async.concurrency`: Controls the maximum number of concurrent async I/O operations. Start with a lower value (e.g., 4-8) when using many Dask threads.
- `threading.max_workers`: Controls Zarr's internal thread pool size for blocking operations (defaults to CPU count). Reduce this to avoid thread contention with Dask's scheduler.

You may need to experiment with different values to find the optimal balance for your workload. Monitor your system's resource usage and adjust these settings based on whether your storage system or CPU is the bottleneck.

### Thread safety and process safety

Zarr arrays are designed to be thread-safe for concurrent reads and writes from multiple threads within the same process. However, proper synchronization is required when writing to overlapping regions from multiple threads.

For multi-process parallelism, Zarr provides safe concurrent writes as long as:
- Different processes write to different chunks
- The storage backend supports atomic writes (most do)

When writing to the same chunks from multiple processes, you should use external synchronization mechanisms or ensure that writes are coordinated to avoid race conditions.

## Pickle support

Zarr arrays and groups can be pickled, as long as the underlying store object can be
pickled. With the exception of the `zarr.storage.MemoryStore`, any of the
storage classes provided in the `zarr.storage` module can be pickled.

If an array or group is backed by a persistent store such as the a `zarr.storage.LocalStore`,
`zarr.storage.ZipStore` or `zarr.storage.FsspecStore` then the store data
**are not** pickled. The only thing that is pickled is the necessary parameters to allow the store
to re-open any underlying files or databases upon being unpickled.

E.g., pickle/unpickle a local store array:

```python exec="true" session="performance" source="above" result="ansi"
import pickle
data = np.arange(100000)
z1 = zarr.create_array(store='data/perf-example-2.zarr', shape=data.shape, chunks=data.shape, dtype=data.dtype)
z1[:] = data
s = pickle.dumps(z1)
z2 = pickle.loads(s)
assert z1 == z2
print(np.all(z1[:] == z2[:]))
```

## Configuring Blosc

Coming soon.
