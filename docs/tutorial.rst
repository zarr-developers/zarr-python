.. _tutorial:

Tutorial
========

Zarr provides classes and functions for working with N-dimensional
arrays that behave like NumPy arrays but whose data is divided into
chunks and compressed. If you are already familiar with HDF5
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
    Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 323; ratio: 1238390.1; initialized: 0/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict

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
    Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 1.8M; ratio: 215.1; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict

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

    >>> z1 = zarr.open_array('example.zarr', mode='w', shape=(10000, 10000),
    ...                      chunks=(1000, 1000), dtype='i4', fill_value=0)
    >>> z1
    Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 323; ratio: 1238390.1; initialized: 0/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: DirectoryStore

The array above will store its configuration metadata and all
compressed chunk data in a directory called 'example.zarr' relative to
the current working directory. The :func:`zarr.creation.open_array` function
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

    >>> z2 = zarr.open_array('example.zarr', mode='r')
    >>> z2
    Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 1.9M; ratio: 204.5; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: DirectoryStore
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
    Array((20000, 10000), float64, chunks=(1000, 1000), order=C)
      nbytes: 1.5G; nbytes_stored: 3.6M; ratio: 422.3; initialized: 100/200
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict

Note that when an array is resized, the underlying data are not
rearranged in any way. If one or more dimensions are shrunk, any
chunks falling outside the new array shape will be deleted from the
underlying store.

For convenience, Zarr arrays also provide an ``append()`` method,
which can be used to append data to any axis. E.g.::

    >>> a = np.arange(10000000, dtype='i4').reshape(10000, 1000)
    >>> z = zarr.array(a, chunks=(1000, 100))
    >>> z
    Array((10000, 1000), int32, chunks=(1000, 100), order=C)
      nbytes: 38.1M; nbytes_stored: 1.9M; ratio: 20.3; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict
    >>> z.append(a)
    (20000, 1000)
    >>> z
    Array((20000, 1000), int32, chunks=(1000, 100), order=C)
      nbytes: 76.3M; nbytes_stored: 3.8M; ratio: 20.3; initialized: 200/200
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict
    >>> z.append(np.vstack([a, a]), axis=1)
    (20000, 2000)
    >>> z
    Array((20000, 2000), int32, chunks=(1000, 100), order=C)
      nbytes: 152.6M; nbytes_stored: 7.5M; ratio: 20.3; initialized: 400/400
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict

.. _tutorial_compress:
      
Compressors
-----------

By default, Zarr uses the `Blosc <http://www.blosc.org/>`_ compression
library to compress each chunk of an array. Blosc is extremely fast
and can be configured in a variety of ways to improve the compression
ratio for different types of data. Blosc is in fact a
"meta-compressor", which means that it can used a number of different
compression algorithms internally to compress the data. Blosc also
provides highly optimized implementations of byte and bit shuffle
filters, which can significantly improve compression ratios for some
data.

Different compressors can be provided via the ``compressor`` keyword argument
accepted by all array creation functions. For example::

    >>> z = zarr.array(np.arange(100000000, dtype='i4').reshape(10000, 10000),
    ...                chunks=(1000, 1000),
    ...                compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2))
    >>> z
    Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 4.4M; ratio: 87.6; initialized: 100/100
      compressor: Blosc(cname='zstd', clevel=3, shuffle=2)
      store: dict

The array above will use Blosc as the primary compressor, using the
Zstandard algorithm (compression level 3) internally within Blosc, and with
the bitshuffle filter applied.

A list of the internal compression libraries available within Blosc can be
obtained via::

    >>> from zarr import blosc
    >>> blosc.list_compressors()
    ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']

In addition to Blosc, other compression libraries can also be
used. Zarr comes with support for zlib, BZ2 and LZMA compression, via
the Python standard library. For example, here is an array using zlib
compression, level 1::

    >>> z = zarr.array(np.arange(100000000, dtype='i4').reshape(10000, 10000),
    ...                chunks=(1000, 1000),
    ...                compressor=zarr.Zlib(level=1))
    >>> z
    Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 132.2M; ratio: 2.9; initialized: 100/100
      compressor: Zlib(level=1)
      store: dict

Here is an example using LZMA with a custom filter pipeline including
LZMA's built-in delta filter::

    >>> import lzma
    >>> lzma_filters = [dict(id=lzma.FILTER_DELTA, dist=4),
    ...                 dict(id=lzma.FILTER_LZMA2, preset=1)]
    >>> compressor = zarr.LZMA(filters=lzma_filters)
    >>> z = zarr.array(np.arange(100000000, dtype='i4').reshape(10000, 10000),
    ...                chunks=(1000, 1000), compressor=compressor)
    >>> z
    Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 248.9K; ratio: 1569.7; initialized: 100/100
      compressor: LZMA(format=1, check=-1, preset=None, filters=[{'dist': 4, 'id': 3}, {'preset': 1, 'id': 33}])
      store: dict

The default compressor can be changed by setting the value of the
``zarr.storage.default_compressor`` variable, e.g.::

    >>> import zarr.storage
    >>> # switch to using Zstandard via Blosc by default
    ... zarr.storage.default_compressor = zarr.Blosc(cname='zstd', clevel=1, shuffle=1)
    >>> z = zarr.zeros(100000000, chunks=1000000)
    >>> z
    Array((100000000,), float64, chunks=(1000000,), order=C)
      nbytes: 762.9M; nbytes_stored: 302; ratio: 2649006.6; initialized: 0/100
      compressor: Blosc(cname='zstd', clevel=1, shuffle=1)
      store: dict
    >>> # switch back to Blosc defaults
    ... zarr.storage.default_compressor = zarr.Blosc()

To disable compression, set ``compressor=None`` when creating an array, e.g.::

    >>> z = zarr.zeros(100000000, chunks=1000000, compressor=None)
    >>> z
    Array((100000000,), float64, chunks=(1000000,), order=C)
      nbytes: 762.9M; nbytes_stored: 209; ratio: 3827751.2; initialized: 0/100
      store: dict

.. _tutorial_filters:

Filters
-------

In some cases, compression can be improved by transforming the data in some
way. For example, if nearby values tend to be correlated, then shuffling the
bytes within each numerical value or storing the difference between adjacent
values may increase compression ratio. Some compressors provide built-in
filters that apply transformations to the data prior to compression. For
example, the Blosc compressor has highly optimized built-in implementations of
byte- and bit-shuffle filters, and the LZMA compressor has a built-in
implementation of a delta filter. However, to provide additional
flexibility for implementing and using filters in combination with different
compressors, Zarr also provides a mechanism for configuring filters outside of
the primary compressor.

Here is an example using the Zarr delta filter with the Blosc compressor:

    >>> filters = [zarr.Delta(dtype='i4')]
    >>> compressor = zarr.Blosc(cname='zstd', clevel=1, shuffle=1)
    >>> z = zarr.array(np.arange(100000000, dtype='i4').reshape(10000, 10000),
    ...                chunks=(1000, 1000), filters=filters, compressor=compressor)
    >>> z
    Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 633.4K; ratio: 616.7; initialized: 100/100
      filters: Delta(dtype=int32)
      compressor: Blosc(cname='zstd', clevel=1, shuffle=1)
      store: dict

Zarr comes with implementations of delta, scale-offset, quantize, packbits and
categorize filters. It is also relatively straightforward to implement custom
filters. For more information see the :mod:`zarr.codecs` API docs.

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
    Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 323; ratio: 1238390.1; initialized: 0/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict; synchronizer: ThreadSynchronizer

This array is safe to read or write within a multi-threaded program.

Zarr also provides support for process synchronization via file locking,
provided that all processes have access to a shared file system. E.g.::

    >>> synchronizer = zarr.ProcessSynchronizer('example.sync')
    >>> z = zarr.open_array('example', mode='w', shape=(10000, 10000),
    ...                     chunks=(1000, 1000), dtype='i4',
    ...                     synchronizer=synchronizer)
    >>> z
    Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 323; ratio: 1238390.1; initialized: 0/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: DirectoryStore; synchronizer: ProcessSynchronizer

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

.. _tutorial_groups:

Groups
------

Zarr supports hierarchical organization of arrays via groups. As with arrays,
groups can be stored in memory, on disk, or via other storage systems that
support a similar interface.

To create a group, use the :func:`zarr.hierarchy.group` function::

    >>> root_group = zarr.group()
    >>> root_group
    Group(/, 0)
      store: DictStore

Groups have a similar API to the Group class from `h5py <http://www.h5py.org/>`_.
For example, groups can contain other groups::

    >>> foo_group = root_group.create_group('foo')
    >>> bar_group = foo_group.create_group('bar')

Groups can also contain arrays, e.g.::

    >>> z1 = bar_group.zeros('baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4',
    ...                      compressor=zarr.Blosc(cname='zstd', clevel=1, shuffle=1))
    >>> z1
    Array(/foo/bar/baz, (10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 324; ratio: 1234567.9; initialized: 0/100
      compressor: Blosc(cname='zstd', clevel=1, shuffle=1)
      store: DictStore

Arrays are known as "datasets" in HDF5 terminology. For compatibility with
h5py, Zarr groups also implement the :func:`zarr.hierarchy.Group.create_dataset`
and :func:`zarr.hierarchy.Group.require_dataset` methods, e.g.::

    >>> z = bar_group.create_dataset('quux', shape=(10000, 10000),
    ...                              chunks=(1000, 1000), dtype='i4',
    ...                              fill_value=0, compression='gzip',
    ...                              compression_opts=1)
    >>> z
    Array(/foo/bar/quux, (10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 275; ratio: 1454545.5; initialized: 0/100
      compressor: Zlib(level=1)
      store: DictStore

Members of a group can be accessed via the suffix notation, e.g.::

    >>> root_group['foo']
    Group(/foo, 1)
      groups: 1; bar
      store: DictStore

The '/' character can be used to access multiple levels of the hierarchy,
e.g.::

    >>> root_group['foo/bar']
    Group(/foo/bar, 2)
      arrays: 2; baz, quux
      store: DictStore
    >>> root_group['foo/bar/baz']
    Array(/foo/bar/baz, (10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 324; ratio: 1234567.9; initialized: 0/100
      compressor: Blosc(cname='zstd', clevel=1, shuffle=1)
      store: DictStore

The :func:`zarr.hierarchy.open_group` provides a convenient way to create or
re-open a group stored in a directory on the file-system, with sub-groups
stored in sub-directories, e.g.::

    >>> persistent_group = zarr.open_group('example', mode='w')
    >>> persistent_group
    Group(/, 0)
      store: DirectoryStore
    >>> z = persistent_group.create_dataset('foo/bar/baz', shape=(10000, 10000),
    ...                                     chunks=(1000, 1000), dtype='i4',
    ...                                     fill_value=0)
    >>> z
    Array(/foo/bar/baz, (10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 323; ratio: 1238390.1; initialized: 0/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: DirectoryStore

For more information on groups see the :mod:`zarr.hierarchy` API docs.

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
    Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      nbytes: 381.5M; nbytes_stored: 26.3M; ratio: 14.5; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict
    >>> zarr.array(a, chunks=(1000, 1000), order='F')
    Array((10000, 10000), int32, chunks=(1000, 1000), order=F)
      nbytes: 381.5M; nbytes_stored: 9.2M; ratio: 41.6; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict

In the above example, Fortran order gives a better compression ratio. This
is an artifical example but illustrates the general point that changing the
order of bytes within chunks of an array may improve the compression ratio,
depending on the structure of the data, the compression algorithm used, and
which compression filters (e.g., byte shuffle) have been applied.

.. _tutorial_tips_storage:

Storage alternatives
~~~~~~~~~~~~~~~~~~~~

Zarr can use any object that implements the ``MutableMapping`` interface as
the store for a group or an array.

Here is an example storing an array directly into a Zip file::

    >>> store = zarr.ZipStore('example.zip', mode='w')
    >>> z = zarr.zeros((1000, 1000), chunks=(100, 100), dtype='i4', store=store)
    >>> z
    Array((1000, 1000), int32, chunks=(100, 100), order=C)
      nbytes: 3.8M; nbytes_stored: 319; ratio: 12539.2; initialized: 0/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: ZipStore
    >>> z[:] = 42
    >>> z
    Array((1000, 1000), int32, chunks=(100, 100), order=C)
      nbytes: 3.8M; nbytes_stored: 21.8K; ratio: 179.2; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: ZipStore
    >>> store.close()
    >>> import os
    >>> os.path.getsize('example.zip')
    30721

Re-open and check that data have been written::

    >>> store = zarr.ZipStore('example.zip', mode='r')
    >>> z = zarr.Array(store)
    >>> z
    Array((1000, 1000), int32, chunks=(100, 100), order=C)
      nbytes: 3.8M; nbytes_stored: 21.8K; ratio: 179.2; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: ZipStore
    >>> z[:]
    array([[42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42],
           ...,
           [42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42]], dtype=int32)
    >>> store.close()

Note that there are some restrictions on how Zip files can be used,
because items within a Zip file cannot be updated in place. This means
that data in the array should only be written once and write
operations should be aligned with chunk boundaries.

Note also that the ``close()`` method must be called after writing any data to
the store, otherwise essential records will not be written to the underlying
zip file.

The Dask project has implementations of the ``MutableMapping``
interface for distributed storage systems, see the `S3Map
<http://s3fs.readthedocs.io/en/latest/api.html#s3fs.mapping.S3Map>`_
and `HDFSMap
<http://hdfs3.readthedocs.io/en/latest/api.html#hdfs3.mapping.HDFSMap>`_
classes.

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

If you are feeling lazy, you can let Zarr guess a chunk shape for your data,
although please note that the algorithm for guessing a chunk shape is based on
simple heuristics and may be far from optimal. E.g.::

    >>> z4 = zarr.zeros((10000, 10000), dtype='i4')
    >>> z4.chunks
    (313, 313)

.. _tutorial_tips_blosc:
    
Configuring Blosc
~~~~~~~~~~~~~~~~~

The Blosc compressor is able to use multiple threads internally to
accelerate compression and decompression. By default, Zarr allows
Blosc to use up to 8 internal threads. The number of Blosc threads can
be changed to increase or decrease this number, e.g.::

    >>> from zarr import blosc
    >>> blosc.set_nthreads(2)
    8

When a Zarr array is being used within a multi-threaded program, Zarr
automatically switches to using Blosc in a single-threaded
"contextual" mode. This is generally better as it allows multiple
program threads to use Blosc simultaneously and prevents CPU thrashing
from too many active threads. If you want to manually override this
behaviour, set the value of the ``blosc.use_threads`` variable to
``True`` (Blosc always uses multiple internal threads) or ``False``
(Blosc always runs in single-threaded contextual mode). To re-enable
automatic switching, set ``blosc.use_threads`` to ``None``.
