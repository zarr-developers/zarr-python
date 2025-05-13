.. only:: doctest

   >>> import shutil
   >>> shutil.rmtree('data', ignore_errors=True)

.. _user-guide-performance:

Optimizing performance
======================

.. _user-guide-chunks:

Chunk optimizations
-------------------

.. _user-guide-chunks-shape:

Chunk size and shape
~~~~~~~~~~~~~~~~~~~~

In general, chunks of at least 1 megabyte (1M) uncompressed size seem to provide
better performance, at least when using the Blosc compression library.

The optimal chunk shape will depend on how you want to access the data. E.g.,
for a 2-dimensional array, if you only ever take slices along the first
dimension, then chunk across the second dimension. If you know you want to chunk
across an entire dimension you can use the full size of that dimension within the
``chunks`` argument, e.g.::

   >>> import zarr
   >>> z1 = zarr.create_array(store={}, shape=(10000, 10000), chunks=(100, 10000), dtype='int32')
   >>> z1.chunks
   (100, 10000)

Alternatively, if you only ever take slices along the second dimension, then
chunk across the first dimension, e.g.::

   >>> z2 = zarr.create_array(store={}, shape=(10000, 10000), chunks=(10000, 100), dtype='int32')
   >>> z2.chunks
   (10000, 100)

If you require reasonable performance for both access patterns then you need to
find a compromise, e.g.::

   >>> z3 = zarr.create_array(store={}, shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
   >>> z3.chunks
   (1000, 1000)

If you are feeling lazy, you can let Zarr guess a chunk shape for your data by
providing ``chunks='auto'``, although please note that the algorithm for guessing
a chunk shape is based on simple heuristics and may be far from optimal. E.g.::

   >>> z4 = zarr.create_array(store={}, shape=(10000, 10000), chunks='auto', dtype='int32')
   >>> z4.chunks
   (625, 625)

If you know you are always going to be loading the entire array into memory, you
can turn off chunks by providing ``chunks`` equal to ``shape``, in which case there
will be one single chunk for the array::

   >>> z5 = zarr.create_array(store={}, shape=(10000, 10000), chunks=(10000, 10000), dtype='int32')
   >>> z5.chunks
   (10000, 10000)


Sharding
~~~~~~~~

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

To use sharding, you need to specify the ``shards`` parameter when creating the array.

   >>> z6 = zarr.create_array(store={}, shape=(10000, 10000, 1000), shards=(1000, 1000, 1000), chunks=(100, 100, 100), dtype='uint8')
   >>> z6.info
   Type               : Array
   Zarr format        : 3
   Data type          : UInt8()
   Shape              : (10000, 10000, 1000)
   Shard shape        : (1000, 1000, 1000)
   Chunk shape        : (100, 100, 100)
   Order              : C
   Read-only          : False
   Store type         : MemoryStore
   Filters            : ()
   Serializer         : BytesCodec(endian=None)
   Compressors        : (ZstdCodec(level=0, checksum=False),)
   No. bytes          : 100000000000 (93.1G)

.. _user-guide-chunks-order:

Chunk memory layout
~~~~~~~~~~~~~~~~~~~

The order of bytes **within each chunk** of an array can be changed via the
``order`` config option, to use either C or Fortran layout. For
multi-dimensional arrays, these two layouts may provide different compression
ratios, depending on the correlation structure within the data. E.g.::

   >>> import numpy as np
   >>>
   >>> a = np.arange(100000000, dtype='int32').reshape(10000, 10000).T
   >>> c = zarr.create_array(store={}, shape=a.shape, chunks=(1000, 1000), dtype=a.dtype, config={'order': 'C'})
   >>> c[:] = a
   >>> c.info_complete()
   Type               : Array
   Zarr format        : 3
   Data type          : Int32(endianness='little')
   Shape              : (10000, 10000)
   Chunk shape        : (1000, 1000)
   Order              : C
   Read-only          : False
   Store type         : MemoryStore
   Filters            : ()
   Serializer         : BytesCodec(endian=<Endian.little: 'little'>)
   Compressors        : (ZstdCodec(level=0, checksum=False),)
   No. bytes          : 400000000 (381.5M)
   No. bytes stored   : 342588911 (326.7M)
   Storage ratio      : 1.2
   Chunks Initialized : 100
   >>> with zarr.config.set({'array.order': 'F'}):
   ...     f = zarr.create_array(store={}, shape=a.shape, chunks=(1000, 1000), dtype=a.dtype)
   ...     f[:] = a
   >>> f.info_complete()
   Type               : Array
   Zarr format        : 3
   Data type          : Int32(endianness='little')
   Shape              : (10000, 10000)
   Chunk shape        : (1000, 1000)
   Order              : F
   Read-only          : False
   Store type         : MemoryStore
   Filters            : ()
   Serializer         : BytesCodec(endian=<Endian.little: 'little'>)
   Compressors        : (ZstdCodec(level=0, checksum=False),)
   No. bytes          : 400000000 (381.5M)
   No. bytes stored   : 342588911 (326.7M)
   Storage ratio      : 1.2
   Chunks Initialized : 100

In the above example, Fortran order gives a better compression ratio. This is an
artificial example but illustrates the general point that changing the order of
bytes within chunks of an array may improve the compression ratio, depending on
the structure of the data, the compression algorithm used, and which compression
filters (e.g., byte-shuffle) have been applied.

.. _user-guide-chunks-empty-chunks:

Empty chunks
~~~~~~~~~~~~

It is possible to configure how Zarr handles the storage of chunks that are "empty"
(i.e., every element in the chunk is equal to the array's fill value). When creating
an array with ``write_empty_chunks=False``, Zarr will check whether a chunk is empty before compression and storage. If a chunk is empty,
then Zarr does not store it, and instead deletes the chunk from storage
if the chunk had been previously stored.

This optimization prevents storing redundant objects and can speed up reads, but the cost is
added computation during array writes, since the contents of
each chunk must be compared to the fill value, and these advantages are contingent on the content of the array.
If you know that your data will form chunks that are almost always non-empty, then there is no advantage to the optimization described above.
In this case, creating an array with ``write_empty_chunks=True`` (the default) will instruct Zarr to write every chunk without checking for emptiness.

The following example illustrates the effect of the ``write_empty_chunks`` flag on
the time required to write an array with different values.::

   >>> import zarr
   >>> import numpy as np
   >>> import time
   >>>
   >>> def timed_write(write_empty_chunks):
   ...     """
   ...     Measure the time required and number of objects created when writing
   ...     to a Zarr array with random ints or fill value.
   ...     """
   ...     chunks = (8192,)
   ...     shape = (chunks[0] * 1024,)
   ...     data = np.random.randint(0, 255, shape)
   ...     dtype = 'uint8'
   ...     arr = zarr.create_array(
   ...         f'data/example-{write_empty_chunks}.zarr',
   ...         shape=shape,
   ...         chunks=chunks,
   ...         dtype=dtype,
   ...         fill_value=0,
   ...         config={'write_empty_chunks': write_empty_chunks}
   ...      )
   ...     # initialize all chunks
   ...     arr[:] = 100
   ...     result = []
   ...     for value in (data, arr.fill_value):
   ...         start = time.time()
   ...         arr[:] = value
   ...         elapsed = time.time() - start
   ...         result.append((elapsed, arr.nchunks_initialized))
   ...     return result
   ... # log results
   >>> for write_empty_chunks in (True, False):
   ...     full, empty = timed_write(write_empty_chunks)
   ...     print(f'\nwrite_empty_chunks={write_empty_chunks}:\n\tRandom Data: {full[0]:.4f}s, {full[1]} objects stored\n\t Empty Data: {empty[0]:.4f}s, {empty[1]} objects stored\n')
   write_empty_chunks=True:
   	Random Data: ..., 1024 objects stored
   	 Empty Data: ...s, 1024 objects stored
   <BLANKLINE>
   write_empty_chunks=False:
   	Random Data: ...s, 1024 objects stored
   	 Empty Data: ...s, 0 objects stored
   <BLANKLINE>

In this example, writing random data is slightly slower with ``write_empty_chunks=True``,
but writing empty data is substantially faster and generates far fewer objects in storage.

.. _user-guide-rechunking:

Changing chunk shapes (rechunking)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coming soon.

.. _user-guide-sync:

Parallel computing and synchronization
--------------------------------------

Coming soon.

.. _user-guide-pickle:

Pickle support
--------------

Zarr arrays and groups can be pickled, as long as the underlying store object can be
pickled. With the exception of the :class:`zarr.storage.MemoryStore`, any of the
storage classes provided in the :mod:`zarr.storage` module can be pickled.

If an array or group is backed by a persistent store such as the a :class:`zarr.storage.LocalStore`,
:class:`zarr.storage.ZipStore` or :class:`zarr.storage.FsspecStore` then the store data
**are not** pickled. The only thing that is pickled is the necessary parameters to allow the store
to re-open any underlying files or databases upon being unpickled.

E.g., pickle/unpickle an local store array::

   >>> import pickle
   >>> data = np.arange(100000)
   >>> z1 = zarr.create_array(store='data/example-2.zarr', shape=data.shape, chunks=data.shape, dtype=data.dtype)
   >>> z1[:] = data
   >>> s = pickle.dumps(z1)
   >>> z2 = pickle.loads(s)
   >>> z1 == z2
   True
   >>> np.all(z1[:] == z2[:])
   np.True_

.. _user-guide-tips-blosc:

Configuring Blosc
-----------------

Coming soon.
