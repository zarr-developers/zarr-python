user-guide-performance

Optimizing performance
======================

.. ipython:: python
   :suppress:

   rm -r data/

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
across an entire dimension you can use ``None`` or ``-1`` within the ``chunks``
argument, e.g.:

.. ipython:: python

   import zarr

   z1 = zarr.zeros((10000, 10000), chunks=(100, None), dtype='i4')
   z1.chunks

Alternatively, if you only ever take slices along the second dimension, then
chunk across the first dimension, e.g.:

.. ipython:: python

   z2 = zarr.zeros((10000, 10000), chunks=(None, 100), dtype='i4')
   z2.chunks

If you require reasonable performance for both access patterns then you need to
find a compromise, e.g.:

.. ipython:: python

   z3 = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4')
   z3.chunks

If you are feeling lazy, you can let Zarr guess a chunk shape for your data by
providing ``chunks=True``, although please note that the algorithm for guessing
a chunk shape is based on simple heuristics and may be far from optimal. E.g.:

.. ipython:: python

   z4 = zarr.zeros((10000, 10000), chunks=True, dtype='i4')
   z4.chunks

If you know you are always going to be loading the entire array into memory, you
can turn off chunks by providing ``chunks=False``, in which case there will be
one single chunk for the array:

.. ipython:: python

   z5 = zarr.zeros((10000, 10000), chunks=False, dtype='i4')
   z5.chunks

.. _user-guide-chunks-order:

Chunk memory layout
~~~~~~~~~~~~~~~~~~~

The order of bytes **within each chunk** of an array can be changed via the
``order`` config option, to use either C or Fortran layout. For
multi-dimensional arrays, these two layouts may provide different compression
ratios, depending on the correlation structure within the data. E.g.:

.. ipython:: python

   a = np.arange(100000000, dtype='i4').reshape(10000, 10000).T
   # TODO: replace with create_array after #2463
   c = zarr.array(a, chunks=(1000, 1000))
   c.info_complete()
   with zarr.config.set({'array.order': 'F'}):
       f = zarr.array(a, chunks=(1000, 1000))
   f.info_complete()

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
the time required to write an array with different values.:

.. ipython:: python

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
       with zarr.config.set({"array.write_empty_chunks": write_empty_chunks}):
           arr = zarr.open(
               f"data/example-{write_empty_chunks}.zarr",
               shape=shape,
               chunks=chunks,
               dtype=dtype,
               fill_value=0,
               mode='w'
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

E.g., pickle/unpickle an local store array:

.. ipython:: python

   import pickle

   # TODO: replace with create_array after #2463
   z1 = zarr.array(store="data/example-2", data=np.arange(100000))
   s = pickle.dumps(z1)
   z2 = pickle.loads(s)
   z1 == z2
   np.all(z1[:] == z2[:])

.. _user-guide-tips-blosc:

Configuring Blosc
-----------------

Coming soon.
