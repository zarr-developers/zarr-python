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

Sometimes you are not free to choose the initial chunking of your input data, or
you might have data saved with chunking which is not optimal for the analysis you
have planned. In such cases it can be advantageous to re-chunk the data. For small
datasets, or when the mismatch between input and output chunks is small
such that only a few chunks of the input dataset need to be read to create each
chunk in the output array, it is sufficient to simply copy the data to a new array
with the desired chunking, e.g.:

.. .. ipython:: python
..    :verbatim:

..    a = zarr.zeros((10000, 10000), chunks=(100,100), dtype='uint16', store='a.zarr')
..    b = zarr.array(a, chunks=(100, 200), store='b.zarr')

If the chunk shapes mismatch, however, a simple copy can lead to non-optimal data
access patterns and incur a substantial performance hit when using
file based stores. One of the most pathological examples is
switching from column-based chunking to row-based chunking e.g.:

.. .. ipython:: python
..    :verbatim:

..    a = zarr.zeros((10000,10000), chunks=(10000, 1), dtype='uint16', store='a.zarr')
..    b = zarr.array(a, chunks=(1,10000), store='b.zarr')

which will require every chunk in the input data set to be repeatedly read when creating
each output chunk. If the entire array will fit within memory, this is simply resolved
by forcing the entire input array into memory as a numpy array before converting
back to zarr with the desired chunking.

.. .. ipython:: python
..    :verbatim:

..    a = zarr.zeros((10000,10000), chunks=(10000, 1), dtype='uint16', store='a.zarr')
..    b = a[...]
..    c = zarr.array(b, chunks=(1,10000), store='c.zarr')

For data sets which have mismatched chunks and which do not fit in memory, a
more sophisticated approach to rechunking, such as offered by the
`rechunker <https://github.com/pangeo-data/rechunker>`_ package and discussed
`here <https://medium.com/pangeo/rechunker-the-missing-link-for-chunked-array-analytics-5b2359e9dc11>`_
may offer a substantial improvement in performance.

.. _user-guide-sync:

Parallel computing and synchronization
--------------------------------------

Zarr arrays have been designed for use as the source or sink for data in
parallel computations. By data source we mean that multiple concurrent read
operations may occur. By data sink we mean that multiple concurrent write
operations may occur, with each writer updating a different region of the
array. Zarr arrays have **not** been designed for situations where multiple
readers and writers are concurrently operating on the same array.

Both multi-threaded and multi-process parallelism are possible. The bottleneck
for most storage and retrieval operations is compression/decompression, and the
Python global interpreter lock (GIL) is released wherever possible during these
operations, so Zarr will generally not block other Python threads from running.

When using a Zarr array as a data sink, some synchronization (locking) may be
required to avoid data loss, depending on how data are being updated. If each
worker in a parallel computation is writing to a separate region of the array,
and if region boundaries are perfectly aligned with chunk boundaries, then no
synchronization is required. However, if region and chunk boundaries are not
perfectly aligned, then synchronization is required to avoid two workers
attempting to modify the same chunk at the same time, which could result in data
loss.

To give a simple example, consider a 1-dimensional array of length 60, ``z``,
divided into three chunks of 20 elements each. If three workers are running and
each attempts to write to a 20 element region (i.e., ``z[0:20]``, ``z[20:40]``
and ``z[40:60]``) then each worker will be writing to a separate chunk and no
synchronization is required. However, if two workers are running and each
attempts to write to a 30 element region (i.e., ``z[0:30]`` and ``z[30:60]``)
then it is possible both workers will attempt to modify the middle chunk at the
same time, and synchronization is required to prevent data loss.

Zarr provides support for chunk-level synchronization. E.g., create an array
with thread synchronization:

.. .. ipython:: python
..    :verbatim:

..    z = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4', synchronizer=zarr.ThreadSynchronizer())
..    z

This array is safe to read or write within a multi-threaded program.

Zarr also provides support for process synchronization via file locking,
provided that all processes have access to a shared file system, and provided
that the underlying file system supports file locking (which is not the case for
some networked file systems). E.g.:

.. .. ipython:: python
..    :verbatim:

..    synchronizer = zarr.ProcessSynchronizer('data/example.sync')

..    z = zarr.open_array('data/example', mode='w', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4', synchronizer=synchronizer)
..    z

This array is safe to read or write from multiple processes.

When using multiple processes to parallelize reads or writes on arrays using the Blosc
compression library, it may be necessary to set ``numcodecs.blosc.use_threads = False``,
as otherwise Blosc may share incorrect global state amongst processes causing programs
to hang. See also the section on :ref:`user-guide-tips-blosc` below.

Please note that support for parallel computing is an area of ongoing research
and development. If you are using Zarr for parallel computing, we welcome
feedback, experience, discussion, ideas and advice, particularly about issues
related to data integrity and performance.

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

   z1 = zarr.array(store="data/example-2", data=np.arange(100000))
   s = pickle.dumps(z1)
   z2 = pickle.loads(s)
   z1 == z2
   np.all(z1[:] == z2[:])

.. _user-guide-tips-blosc:

Configuring Blosc
-----------------

Coming soon.

.. The Blosc compressor is able to use multiple threads internally to accelerate
.. compression and decompression. By default, Blosc uses up to 8
.. internal threads. The number of Blosc threads can be changed to increase or
.. decrease this number, e.g.:

.. .. ipython:: python
..    :verbatim:

..    from numcodecs import blosc

..    blosc.set_nthreads(2)  # doctest: +SKIP

.. When a Zarr array is being used within a multi-threaded program, Zarr
.. automatically switches to using Blosc in a single-threaded
.. "contextual" mode. This is generally better as it allows multiple
.. program threads to use Blosc simultaneously and prevents CPU thrashing
.. from too many active threads. If you want to manually override this
.. behaviour, set the value of the ``blosc.use_threads`` variable to
.. ``True`` (Blosc always uses multiple internal threads) or ``False``
.. (Blosc always runs in single-threaded contextual mode). To re-enable
.. automatic switching, set ``blosc.use_threads`` to ``None``.

.. Please note that if Zarr is being used within a multi-process program, Blosc may not
.. be safe to use in multi-threaded mode and may cause the program to hang. If using Blosc
.. in a multi-process program then it is recommended to set ``blosc.use_threads = False``.
