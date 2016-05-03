Tutorial
========

Zarr provides classes and functions for working with N-dimensional
arrays that behave like NumPy arrays but whose data is divided into
chunks and compressed. If you are already familiar with HDF5 then Zarr
provides similar functionality, but with some additional flexibility.

Creating an array
-----------------

Zarr has a number of convenience functions for creating arrays. For example::

    >>> import zarr
    >>> z = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4')
    >>> z
    zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
    compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
    nbytes: 381.5M; nbytes_stored: 317; ratio: 1261829.7; initialized: 0/100
    store: builtins.dict

The code above creates a 2-dimensional array of 32-bit integers with 10000 rows and 10000
columns, divided into chunks where each chunk has 1000 rows and 1000
columns (and so there will be 100 chunks in total).

For a complete list of array creation routines see the :mod:`zarr.creation` module documentation.

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
first created, none of the chunks are initialised. Writing data into
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

Resizing and appending
----------------------

A Zarr array can be resized, which means that any of its dimensions can be
increased or decreased in length. For example::

    >>> z = zarr.zeros(shape=(10000, 10000), chunks=(1000, 1000))
    >>> z[:] = 42
    >>> z.resize(20000, 10000)
    >>> z
    zarr.core.Array((20000, 10000), float64, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 1.5G; nbytes_stored: 5.9M; ratio: 259.9; initialized: 100/200
      store: builtins.dict

Note that when an array is resized, the underlying data are not rearranged in
any way. If one or more dimensions are shrunk, any chunks falling outside the
new array shape will be deleted from the underlying store.

For convenience, Zarr arrays also provide an ``append()`` method, which can be
used to append data to any axis. E.g.::

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

Persistent arrays
-----------------

In the examples above, data for each chunk of the array was stored in
memory. Zarr arrays can also be stored on a file system, enabling
persistence of data between sessions. For example::

    >>> z1 = zarr.open('example.zarr', mode='w', shape=(10000, 10000),
    ...                chunks=(1000, 1000), dtype='i4', fill_value=0)
    >>> z1
    zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 381.5M; nbytes_stored: 317; ratio: 1261829.7; initialized: 0/100
      store: zarr.storage.DirectoryStore

The array above will store its configuration metadata and all compressed chunk
data in a directory called 'example.zarr' relative to the current working
directory. The :func:`zarr.creation.open` function provides a convenient way
to create new persistent arrays or open existing arrays. Note that there is no
need to close an array, and data are automatically flushed to disk.

Persistent arrays support the same interface for reading and writing data,
e.g.::

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

Compression
-----------

@@TODO discuss available compressors

Parallel computing
------------------

@@TODO discuss synchronization

@@TODO use_context blosc with dask

User attributes
---------------

@@TODO

Tips and tricks
---------------

@@TODO copying an array via __setitem__

@@TODO changing the order to improve compression ratio

@@TODO using other storage, e.g., ZipFile via zict

@@TODO tips on chunk shape and size
