.. _tutorial:

Tutorial
========

Zarr provides classes and functions for working with N-dimensional
arrays that behave like NumPy arrays but whose data is divided into
chunks and compressed. If you are already familiar with HDF5 datasets
then Zarr arrays provide similar functionality, but with some
additional flexibility.

.. _tutorial_create:

Creating an array
-----------------

Zarr has a number of convenience functions for creating arrays. For
example::

    >>> import zarr
    >>> z = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4')
    >>> z
    zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
    compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
    nbytes: 381.5M; nbytes_stored: 317; ratio: 1261829.7; initialized: 0/100
    store: builtins.dict

The code above creates a 2-dimensional array of 32-bit integers with
10000 rows and 10000 columns, divided into chunks where each chunk has
1000 rows and 1000 columns (and so there will be 100 chunks in total).

For a complete list of array creation routines see the
:mod:`zarr.creation` module documentation.

.. _tutorial_array:
     
Reading and writing data
------------------------

Zarr arrays support a similar interface to NumPy arrays for reading
and writing data. For example, the entire array can be filled with a
scalar value::

    >>> z[:] = 42
    >>> z
    zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 381.5M; nbytes_stored: 2.2M; ratio: 170.4; initialized: 100/100
      store: builtins.dict

Notice that the values of ``nbytes_stored``, ``ratio`` and
``initialized`` have changed. This is because when a Zarr array is
first created, none of the chunks are initialized. Writing data into
the array will cause the necessary chunks to be initialized.

Regions of the array can also be written to, e.g.::

    >>> import numpy as np
    >>> z[0, :] = np.arange(10000)
    >>> z[:, 0] = np.arange(10000)

The contents of the array can be retrieved by slicing, which will load
the requested region into a NumPy array, e.g.::

    >>> z[0, 0]
    0
    >>> z[-1, -1]
    42
    >>> z[0, :]
    array([   0,    1,    2, ..., 9997, 9998, 9999], dtype=int32)
    >>> z[:, 0]
    array([   0,    1,    2, ..., 9997, 9998, 9999], dtype=int32)
    >>> z[:]
    array([[   0,    1,    2, ..., 9997, 9998, 9999],
           [   1,   42,   42, ...,   42,   42,   42],
           [   2,   42,   42, ...,   42,   42,   42],
           ...,
           [9997,   42,   42, ...,   42,   42,   42],
           [9998,   42,   42, ...,   42,   42,   42],
           [9999,   42,   42, ...,   42,   42,   42]], dtype=int32)

.. _tutorial_persist:
	   
Persistent arrays
-----------------

In the examples above, compressed data for each chunk of the array was
stored in memory. Zarr arrays can also be stored on a file system,
enabling persistence of data between sessions. For example::

    >>> z1 = zarr.open('example.zarr', mode='w', shape=(10000, 10000),
    ...                chunks=(1000, 1000), dtype='i4', fill_value=0)
    >>> z1
    zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 381.5M; nbytes_stored: 317; ratio: 1261829.7; initialized: 0/100
      store: zarr.storage.DirectoryStore

The array above will store its configuration metadata and all
compressed chunk data in a directory called 'example.zarr' relative to
the current working directory. The :func:`zarr.creation.open` function
provides a convenient way to create a new persistent array or continue
working with an existing array. Note that there is no need to close an
array, and data are automatically flushed to disk whenever an array is
modified.

Persistent arrays support the same interface for reading and writing
data, e.g.::

    >>> z1[:] = 42
    >>> z1[0, :] = np.arange(10000)
    >>> z1[:, 0] = np.arange(10000)

Check that the data have been written and can be read again::

    >>> z2 = zarr.open('example.zarr', mode='r')
    >>> z2
    zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 381.5M; nbytes_stored: 2.3M; ratio: 163.8; initialized: 100/100
      store: zarr.storage.DirectoryStore
    >>> np.all(z1[:] == z2[:])
    True

.. _tutorial_resize:    

Resizing and appending
----------------------

A Zarr array can be resized, which means that any of its dimensions
can be increased or decreased in length. For example::

    >>> z = zarr.zeros(shape=(10000, 10000), chunks=(1000, 1000))
    >>> z[:] = 42
    >>> z.resize(20000, 10000)
    >>> z
    zarr.core.Array((20000, 10000), float64, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 1.5G; nbytes_stored: 5.9M; ratio: 259.9; initialized: 100/200
      store: builtins.dict

Note that when an array is resized, the underlying data are not
rearranged in any way. If one or more dimensions are shrunk, any
chunks falling outside the new array shape will be deleted from the
underlying store.

For convenience, Zarr arrays also provide an ``append()`` method,
which can be used to append data to any axis. E.g.::

    >>> a = np.arange(10000000, dtype='i4').reshape(10000, 1000)
    >>> z = zarr.array(a, chunks=(1000, 100))
    >>> z
    zarr.core.Array((10000, 1000), int32, chunks=(1000, 100), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 38.1M; nbytes_stored: 2.0M; ratio: 19.3; initialized: 100/100
      store: builtins.dict
    >>> z.append(a)
    >>> z
    zarr.core.Array((20000, 1000), int32, chunks=(1000, 100), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 76.3M; nbytes_stored: 4.0M; ratio: 19.3; initialized: 200/200
      store: builtins.dict
    >>> z.append(np.vstack([a, a]), axis=1)
    >>> z
    zarr.core.Array((20000, 2000), int32, chunks=(1000, 100), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 152.6M; nbytes_stored: 7.9M; ratio: 19.3; initialized: 400/400
      store: builtins.dict

.. _tutorial_compress:
      
Compression
-----------

By default, Zarr uses the `Blosc <http://www.blosc.org/>`_ compression
library to compress each chunk of an array. Blosc is extremely fast
and can be configured in a variety of ways to improve the compression
ratio for different types of data. Blosc is in fact a
"meta-compressor", which means that it can used a number of different
compression algorithms internally to compress the data. Blosc also
provides highly optimised implementations of byte and bit shuffle
filters, which can significantly improve compression ratios for some
data.

Options for the compressor can be controlled via the
``compression_opts`` keyword argument accepted by all array creation
functions. For example::

    >>> z = zarr.array(np.arange(100000000, dtype='i4').reshape(10000, 10000),
    ...                chunks=(1000, 1000), compression='blosc',
    ...                compression_opts=dict(cname='lz4', clevel=3, shuffle=2))
    >>> z
    zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 3, 'cname': 'lz4', 'shuffle': 2}
      nbytes: 381.5M; nbytes_stored: 17.6M; ratio: 21.7; initialized: 100/100
      store: builtins.dict

The array above will use Blosc as the primary compressor, using the
LZ4 algorithm (compression level 3) internally within Blosc, and with
the bitshuffle filter applied.

In addition to Blosc, other compression libraries can also be
used. Zarr comes with support for zlib, BZ2 and LZMA compression, via
the Python standard library. For example, here is an array using zlib
compression, level 1::

    >>> z = zarr.array(np.arange(100000000, dtype='i4').reshape(10000, 10000),
    ...                chunks=(1000, 1000), compression='zlib',
    ...                compression_opts=1)
    >>> z
    zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      compression: zlib; compression_opts: 1
      nbytes: 381.5M; nbytes_stored: 132.2M; ratio: 2.9; initialized: 100/100
      store: builtins.dict

Here is an example using LZMA with a custom filter pipeline including
the delta filter::

    >>> import lzma
    >>> filters = [dict(id=lzma.FILTER_DELTA, dist=4),
    ...            dict(id=lzma.FILTER_LZMA2, preset=1)]
    >>> z = zarr.array(np.arange(100000000, dtype='i4').reshape(10000, 10000),
    ...                chunks=(1000, 1000), compression='lzma',
    ...                compression_opts=dict(filters=filters))
    >>> z
    zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      compression: lzma; compression_opts: {'preset': None, 'filters': [{'dist': 4, 'id': 3}, {'preset': 1, 'id': 33}], 'check': 0, 'format': 1}
      nbytes: 381.5M; nbytes_stored: 248.1K; ratio: 1574.7; initialized: 100/100
      store: builtins.dict

.. _tutorial_sync:

Parallel computing and synchronization
--------------------------------------

Zarr arrays can be used as either the source or sink for data in
parallel computations. Both multi-threaded and multi-process
parallelism are supported. The Python global interpreter lock (GIL) is
released for both compression and decompression operations, so Zarr
will not block other Python threads from running.

A Zarr array can be read concurrently by multiple threads or processes.
No synchronization (i.e., locking) is required for concurrent reads.

A Zarr array can also be written to concurrently by multiple threads
or processes. Some synchronization may be required, depending on the
way the data is being written.

If each worker in a parallel computation is writing to a separate
region of the array, and if region boundaries are perfectly aligned
with chunk boundaries, then no synchronization is required. However,
if region and chunk boundaries are not perfectly aligned, then
synchronization is required to avoid two workers attempting to modify
the same chunk at the same time.

To give a simple example, consider a 1-dimensional array of length 60,
``z``, divided into three chunks of 20 elements each. If three workers
are running and each attempts to write to a 20 element region (i.e.,
``z[0:20]``, ``z[20:40]`` and ``z[40:60]``) then each worker will be
writing to a separate chunk and no synchronization is
required. However, if two workers are running and each attempts to
write to a 30 element region (i.e., ``z[0:30]`` and ``z[30:60]``) then
it is possible both workers will attempt to modify the middle chunk at
the same time, and synchronization is required to prevent data loss.

Zarr provides support for chunk-level synchronization. E.g., create an
array with thread synchronization::

    >>> z = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4',
    ...                 synchronizer=zarr.ThreadSynchronizer())
    >>> z
    zarr.sync.SynchronizedArray((10000, 10000), int32, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 381.5M; nbytes_stored: 317; ratio: 1261829.7; initialized: 0/100
      store: builtins.dict; synchronizer: zarr.sync.ThreadSynchronizer

This array is safe to read or write within a multi-threaded program.

Zarr also provides support for process synchronization via file locking,
provided that all processes have access to a shared file system. E.g.::

    >>> synchronizer = zarr.ProcessSynchronizer('example.zarr')
    >>> z = zarr.open('example.zarr', mode='w', shape=(10000, 10000),
    ...               chunks=(1000, 1000), dtype='i4',
    ...               synchronizer=synchronizer)
    >>> z
    zarr.sync.SynchronizedArray((10000, 10000), int32, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 381.5M; nbytes_stored: 317; ratio: 1261829.7; initialized: 0/100
      store: zarr.storage.DirectoryStore; synchronizer: zarr.sync.ProcessSynchronizer

This array is safe to read or write from multiple processes.

.. _tutorial_attrs:

User attributes
---------------

Zarr arrays also support custom key/value attributes, which can be useful
for associating an array with application-specific metadata. For example::

    >>> z = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4')
    >>> z.attrs['foo'] = 'bar'
    >>> z.attrs['baz'] = 42
    >>> sorted(z.attrs)
    ['baz', 'foo']
    >>> 'foo' in z.attrs
    True
    >>> z.attrs['foo']
    'bar'
    >>> z.attrs['baz']
    42

Internally Zarr uses JSON to store array attributes, so attribute values
must be JSON serializable.

.. _tutorial_tips:

Tips and tricks
---------------

.. _tutorial_tips_copy:

Copying large arrays
~~~~~~~~~~~~~~~~~~~~

Data can be copied between large arrays without needing much memory,
e.g.::

    >>> z1 = zarr.empty((10000, 10000), chunks=(1000, 1000), dtype='i4')
    >>> z1[:] = 42
    >>> z2 = zarr.empty_like(z1)
    >>> z2[:] = z1

Internally the example above works chunk-by-chunk, extracting only the
data from ``z1`` required to fill each chunk in ``z2``. The source of
the data (``z1``) could equally be an h5py Dataset.

.. _tutorial_tips_order:

Changing memory layout
~~~~~~~~~~~~~~~~~~~~~~

The order of bytes within each chunk of an array can be changed via
the ``order`` keyword argument, to use either C or Fortran layout. For
multi-dimensional arrays, these two layouts may provide different
compression ratios, depending on the correlation structure within the
data. E.g.::

    >>> a = np.arange(100000000, dtype='i4').reshape(10000, 10000).T
    >>> zarr.array(a, chunks=(1000, 1000))
    zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 381.5M; nbytes_stored: 26.1M; ratio: 14.6; initialized: 100/100
      store: builtins.dict
    >>> zarr.array(a, chunks=(1000, 1000), order='F')
    zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=F)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 381.5M; nbytes_stored: 10.0M; ratio: 38.0; initialized: 100/100
      store: builtins.dict

In the above example, Fortran order gives a better compression ratio. This
is an artifical example but illustrates the general point that changing the
order of bytes within chunks of an array may improve the compression ratio,
depending on the structure of the data, the compression algorithm used, and
which compression filters (e.g., byte shuffle) have been applied.

.. _tutorial_tips_storage:

Storage alternatives
~~~~~~~~~~~~~~~~~~~~

Zarr can use any object that implements the ``MutableMapping`` interface as
the store for an array.

Here is an example storing an array directly into a Zip file via the
`zict <https://github.com/mrocklin/zict>`_ package::

    >>> import zict
    >>> import os
    >>> store = zict.Zip('example.zip', mode='w')
    >>> z = zarr.zeros((1000, 1000), chunks=(100, 100), dtype='i4',
    ...                compression='zlib', compression_opts=1, store=store)
    >>> z
    zarr.core.Array((1000, 1000), int32, chunks=(100, 100), order=C)
      compression: zlib; compression_opts: 1
      nbytes: 3.8M; initialized: 0/100
      store: zict.zip.Zip
    >>> z[:] = 42
    >>> store.flush()  # only required for zict.Zip
    >>> os.path.getsize('example.zip')
    30828

Re-open and check that data have been written::

    >>> store = zict.Zip('example.zip', mode='r')
    >>> z = zarr.Array(store)
    >>> z
    zarr.core.Array((1000, 1000), int32, chunks=(100, 100), order=C)
      compression: zlib; compression_opts: 1
      nbytes: 3.8M; initialized: 100/100
      store: zict.zip.Zip
    >>> z[:]
    array([[42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42],
           ...,
           [42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42]], dtype=int32)

Note that there are some restrictions on how Zip files can be used,
because items within a Zip file cannot be updated in place. This means
that data in the array should only be written once and write
operations should be aligned with chunk boundaries.

.. _tutorial_tips_chunks:

Chunk size and shape
~~~~~~~~~~~~~~~~~~~~

In general, chunks of at least 1 megabyte (1M) seem to provide the best
performance, at least when using the Blosc compression library.

The optimal chunk shape will depend on how you want to access the data. E.g.,
for a 2-dimensional array, if you only ever take slices along the first
dimension, then chunk across the second dimenson. If you know you want to
chunk across an entire dimension you can use ``None`` within the ``chunks``
argument, e.g.::

    >>> z1 = zarr.zeros((10000, 10000), chunks=(100, None), dtype='i4')
    >>> z1.chunks
    (100, 10000)

Alternatively, if you only ever take slices along the second dimension, then
chunk across the first dimension, e.g.::

    >>> z2 = zarr.zeros((10000, 10000), chunks=(None, 100), dtype='i4')
    >>> z2.chunks
    (10000, 100)

If you require reasonable performance for both access patterns then you need
to find a compromise, e.g.::

    >>> z3 = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4')
    >>> z3.chunks
    (1000, 1000)

.. _tutorial_tips_blosc:
    
Configuring Blosc
~~~~~~~~~~~~~~~~~

The Blosc compressor is able to use multiple threads internally to
accelerate compression and decompression. By default, Zarr allows
Blosc to use up to 4 internal threads. The number of Blosc threads can
be changed, e.g.::

    >>> from zarr import blosc
    >>> blosc.set_nthreads(2)
    4

When a Zarr array is being used within a multi-threaded program, Zarr
automatically switches to using Blosc in a single-threaded
"contextual" mode. This is generally better as it allows multiple
program threads to use Blosc simultaneously and prevents CPU thrashing
from too many active threads. If you want to manually override this
behaviour, set the value of the ``blosc.use_threads`` variable to
``True`` (Blosc always uses multiple internal threads) or ``False``
(Blosc always runs in single-threaded contextual mode). To re-enable
automatic switching, set ``blosc.use_threads`` to ``None``.
