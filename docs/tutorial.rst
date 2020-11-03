.. _tutorial:

Tutorial
========

Zarr provides classes and functions for working with N-dimensional arrays that
behave like NumPy arrays but whose data is divided into chunks and each chunk is
compressed. If you are already familiar with HDF5 then Zarr arrays provide
similar functionality, but with some additional flexibility.

.. _tutorial_create:

Creating an array
-----------------

Zarr has several functions for creating arrays. For example::

    >>> import zarr
    >>> z = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4')
    >>> z
    <zarr.core.Array (10000, 10000) int32>

The code above creates a 2-dimensional array of 32-bit integers with 10000 rows
and 10000 columns, divided into chunks where each chunk has 1000 rows and 1000
columns (and so there will be 100 chunks in total).

For a complete list of array creation routines see the :mod:`zarr.creation`
module documentation.

.. _tutorial_array:

Reading and writing data
------------------------

Zarr arrays support a similar interface to NumPy arrays for reading and writing
data. For example, the entire array can be filled with a scalar value::

    >>> z[:] = 42

Regions of the array can also be written to, e.g.::

    >>> import numpy as np
    >>> z[0, :] = np.arange(10000)
    >>> z[:, 0] = np.arange(10000)

The contents of the array can be retrieved by slicing, which will load the
requested region into memory as a NumPy array, e.g.::

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

In the examples above, compressed data for each chunk of the array was stored in
main memory. Zarr arrays can also be stored on a file system, enabling
persistence of data between sessions. For example::

    >>> z1 = zarr.open('data/example.zarr', mode='w', shape=(10000, 10000),
    ...                chunks=(1000, 1000), dtype='i4')

The array above will store its configuration metadata and all compressed chunk
data in a directory called 'data/example.zarr' relative to the current working
directory. The :func:`zarr.convenience.open` function provides a convenient way
to create a new persistent array or continue working with an existing
array. Note that although the function is called "open", there is no need to
close an array: data are automatically flushed to disk, and files are
automatically closed whenever an array is modified.

Persistent arrays support the same interface for reading and writing data,
e.g.::

    >>> z1[:] = 42
    >>> z1[0, :] = np.arange(10000)
    >>> z1[:, 0] = np.arange(10000)

Check that the data have been written and can be read again::

    >>> z2 = zarr.open('data/example.zarr', mode='r')
    >>> np.all(z1[:] == z2[:])
    True

If you are just looking for a fast and convenient way to save NumPy arrays to
disk then load back into memory later, the functions
:func:`zarr.convenience.save` and :func:`zarr.convenience.load` may be
useful. E.g.::

    >>> a = np.arange(10)
    >>> zarr.save('data/example.zarr', a)
    >>> zarr.load('data/example.zarr')
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

Please note that there are a number of other options for persistent array
storage, see the section on :ref:`tutorial_storage` below.

.. _tutorial_resize:

Resizing and appending
----------------------

A Zarr array can be resized, which means that any of its dimensions can be
increased or decreased in length. For example::

    >>> z = zarr.zeros(shape=(10000, 10000), chunks=(1000, 1000))
    >>> z[:] = 42
    >>> z.resize(20000, 10000)
    >>> z.shape
    (20000, 10000)

Note that when an array is resized, the underlying data are not rearranged in
any way. If one or more dimensions are shrunk, any chunks falling outside the
new array shape will be deleted from the underlying store.

For convenience, Zarr arrays also provide an ``append()`` method, which can be
used to append data to any axis. E.g.::

    >>> a = np.arange(10000000, dtype='i4').reshape(10000, 1000)
    >>> z = zarr.array(a, chunks=(1000, 100))
    >>> z.shape
    (10000, 1000)
    >>> z.append(a)
    (20000, 1000)
    >>> z.append(np.vstack([a, a]), axis=1)
    (20000, 2000)
    >>> z.shape
    (20000, 2000)

.. _tutorial_compress:

Compressors
-----------

A number of different compressors can be used with Zarr. A separate package
called NumCodecs_ is available which provides a common interface to various
compressor libraries including Blosc, Zstandard, LZ4, Zlib, BZ2 and
LZMA. Different compressors can be provided via the ``compressor`` keyword
argument accepted by all array creation functions. For example::

    >>> from numcodecs import Blosc
    >>> compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    >>> data = np.arange(100000000, dtype='i4').reshape(10000, 10000)
    >>> z = zarr.array(data, chunks=(1000, 1000), compressor=compressor)
    >>> z.compressor
    Blosc(cname='zstd', clevel=3, shuffle=BITSHUFFLE, blocksize=0)

This array above will use Blosc as the primary compressor, using the Zstandard
algorithm (compression level 3) internally within Blosc, and with the
bit-shuffle filter applied.

When using a compressor, it can be useful to get some diagnostics on the
compression ratio. Zarr arrays provide a ``info`` property which can be used to
print some diagnostics, e.g.::

    >>> z.info
    Type               : zarr.core.Array
    Data type          : int32
    Shape              : (10000, 10000)
    Chunk shape        : (1000, 1000)
    Order              : C
    Read-only          : False
    Compressor         : Blosc(cname='zstd', clevel=3, shuffle=BITSHUFFLE,
                       : blocksize=0)
    Store type         : builtins.dict
    No. bytes          : 400000000 (381.5M)
    No. bytes stored   : 3379344 (3.2M)
    Storage ratio      : 118.4
    Chunks initialized : 100/100

If you don't specify a compressor, by default Zarr uses the Blosc
compressor. Blosc is generally very fast and can be configured in a variety of
ways to improve the compression ratio for different types of data. Blosc is in
fact a "meta-compressor", which means that it can use a number of different
compression algorithms internally to compress the data. Blosc also provides
highly optimized implementations of byte- and bit-shuffle filters, which can
improve compression ratios for some data. A list of the internal compression
libraries available within Blosc can be obtained via::

    >>> from numcodecs import blosc
    >>> blosc.list_compressors()
    ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']

In addition to Blosc, other compression libraries can also be used. For example,
here is an array using Zstandard compression, level 1::

    >>> from numcodecs import Zstd
    >>> z = zarr.array(np.arange(100000000, dtype='i4').reshape(10000, 10000),
    ...                chunks=(1000, 1000), compressor=Zstd(level=1))
    >>> z.compressor
    Zstd(level=1)

Here is an example using LZMA with a custom filter pipeline including LZMA's
built-in delta filter::

    >>> import lzma
    >>> lzma_filters = [dict(id=lzma.FILTER_DELTA, dist=4),
    ...                 dict(id=lzma.FILTER_LZMA2, preset=1)]
    >>> from numcodecs import LZMA
    >>> compressor = LZMA(filters=lzma_filters)
    >>> z = zarr.array(np.arange(100000000, dtype='i4').reshape(10000, 10000),
    ...                chunks=(1000, 1000), compressor=compressor)
    >>> z.compressor
    LZMA(format=1, check=-1, preset=None, filters=[{'dist': 4, 'id': 3}, {'id': 33, 'preset': 1}])

The default compressor can be changed by setting the value of the
``zarr.storage.default_compressor`` variable, e.g.::

    >>> import zarr.storage
    >>> from numcodecs import Zstd, Blosc
    >>> # switch to using Zstandard
    ... zarr.storage.default_compressor = Zstd(level=1)
    >>> z = zarr.zeros(100000000, chunks=1000000)
    >>> z.compressor
    Zstd(level=1)
    >>> # switch back to Blosc defaults
    ... zarr.storage.default_compressor = Blosc()

To disable compression, set ``compressor=None`` when creating an array, e.g.::

    >>> z = zarr.zeros(100000000, chunks=1000000, compressor=None)
    >>> z.compressor is None
    True

.. _tutorial_filters:

Filters
-------

In some cases, compression can be improved by transforming the data in some
way. For example, if nearby values tend to be correlated, then shuffling the
bytes within each numerical value or storing the difference between adjacent
values may increase compression ratio. Some compressors provide built-in filters
that apply transformations to the data prior to compression. For example, the
Blosc compressor has built-in implementations of byte- and bit-shuffle filters,
and the LZMA compressor has a built-in implementation of a delta
filter. However, to provide additional flexibility for implementing and using
filters in combination with different compressors, Zarr also provides a
mechanism for configuring filters outside of the primary compressor.

Here is an example using a delta filter with the Blosc compressor::

    >>> from numcodecs import Blosc, Delta
    >>> filters = [Delta(dtype='i4')]
    >>> compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
    >>> data = np.arange(100000000, dtype='i4').reshape(10000, 10000)
    >>> z = zarr.array(data, chunks=(1000, 1000), filters=filters, compressor=compressor)
    >>> z.info
    Type               : zarr.core.Array
    Data type          : int32
    Shape              : (10000, 10000)
    Chunk shape        : (1000, 1000)
    Order              : C
    Read-only          : False
    Filter [0]         : Delta(dtype='<i4')
    Compressor         : Blosc(cname='zstd', clevel=1, shuffle=SHUFFLE, blocksize=0)
    Store type         : builtins.dict
    No. bytes          : 400000000 (381.5M)
    No. bytes stored   : 1290562 (1.2M)
    Storage ratio      : 309.9
    Chunks initialized : 100/100

For more information about available filter codecs, see the `Numcodecs
<http://numcodecs.readthedocs.io/>`_ documentation.

.. _tutorial_groups:

Groups
------

Zarr supports hierarchical organization of arrays via groups. As with arrays,
groups can be stored in memory, on disk, or via other storage systems that
support a similar interface.

To create a group, use the :func:`zarr.group` function::

    >>> root = zarr.group()
    >>> root
    <zarr.hierarchy.Group '/'>

Groups have a similar API to the Group class from `h5py
<http://www.h5py.org/>`_.  For example, groups can contain other groups::

    >>> foo = root.create_group('foo')
    >>> bar = foo.create_group('bar')

Groups can also contain arrays, e.g.::

    >>> z1 = bar.zeros('baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4')
    >>> z1
    <zarr.core.Array '/foo/bar/baz' (10000, 10000) int32>

Arrays are known as "datasets" in HDF5 terminology. For compatibility with h5py,
Zarr groups also implement the ``create_dataset()`` and ``require_dataset()``
methods, e.g.::

    >>> z = bar.create_dataset('quux', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4')
    >>> z
    <zarr.core.Array '/foo/bar/quux' (10000, 10000) int32>

Members of a group can be accessed via the suffix notation, e.g.::

    >>> root['foo']
    <zarr.hierarchy.Group '/foo'>

The '/' character can be used to access multiple levels of the hierarchy in one
call, e.g.::

    >>> root['foo/bar']
    <zarr.hierarchy.Group '/foo/bar'>
    >>> root['foo/bar/baz']
    <zarr.core.Array '/foo/bar/baz' (10000, 10000) int32>

The :func:`zarr.hierarchy.Group.tree` method can be used to print a tree
representation of the hierarchy, e.g.::

    >>> root.tree()
    /
     └── foo
         └── bar
             ├── baz (10000, 10000) int32
             └── quux (10000, 10000) int32

The :func:`zarr.convenience.open` function provides a convenient way to create or
re-open a group stored in a directory on the file-system, with sub-groups stored in
sub-directories, e.g.::

    >>> root = zarr.open('data/group.zarr', mode='w')
    >>> root
    <zarr.hierarchy.Group '/'>
    >>> z = root.zeros('foo/bar/baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4')
    >>> z
    <zarr.core.Array '/foo/bar/baz' (10000, 10000) int32>

Groups can be used as context managers (in a ``with`` statement).
If the underlying store has a ``close`` method, it will be called on exit.

For more information on groups see the :mod:`zarr.hierarchy` and
:mod:`zarr.convenience` API docs.

.. _tutorial_diagnostics:

Array and group diagnostics
---------------------------

Diagnostic information about arrays and groups is available via the ``info``
property. E.g.::

    >>> root = zarr.group()
    >>> foo = root.create_group('foo')
    >>> bar = foo.zeros('bar', shape=1000000, chunks=100000, dtype='i8')
    >>> bar[:] = 42
    >>> baz = foo.zeros('baz', shape=(1000, 1000), chunks=(100, 100), dtype='f4')
    >>> baz[:] = 4.2
    >>> root.info
    Name        : /
    Type        : zarr.hierarchy.Group
    Read-only   : False
    Store type  : zarr.storage.MemoryStore
    No. members : 1
    No. arrays  : 0
    No. groups  : 1
    Groups      : foo

    >>> foo.info
    Name        : /foo
    Type        : zarr.hierarchy.Group
    Read-only   : False
    Store type  : zarr.storage.MemoryStore
    No. members : 2
    No. arrays  : 2
    No. groups  : 0
    Arrays      : bar, baz

    >>> bar.info
    Name               : /foo/bar
    Type               : zarr.core.Array
    Data type          : int64
    Shape              : (1000000,)
    Chunk shape        : (100000,)
    Order              : C
    Read-only          : False
    Compressor         : Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=0)
    Store type         : zarr.storage.MemoryStore
    No. bytes          : 8000000 (7.6M)
    No. bytes stored   : 33240 (32.5K)
    Storage ratio      : 240.7
    Chunks initialized : 10/10

    >>> baz.info
    Name               : /foo/baz
    Type               : zarr.core.Array
    Data type          : float32
    Shape              : (1000, 1000)
    Chunk shape        : (100, 100)
    Order              : C
    Read-only          : False
    Compressor         : Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=0)
    Store type         : zarr.storage.MemoryStore
    No. bytes          : 4000000 (3.8M)
    No. bytes stored   : 23943 (23.4K)
    Storage ratio      : 167.1
    Chunks initialized : 100/100

Groups also have the :func:`zarr.hierarchy.Group.tree` method, e.g.::

    >>> root.tree()
    /
     └── foo
         ├── bar (1000000,) int64
         └── baz (1000, 1000) float32

If you're using Zarr within a Jupyter notebook (requires
`ipytree <https://github.com/QuantStack/ipytree>`_), calling ``tree()`` will generate an
interactive tree representation, see the `repr_tree.ipynb notebook
<http://nbviewer.jupyter.org/github/zarr-developers/zarr-python/blob/master/notebooks/repr_tree.ipynb>`_
for more examples.

.. _tutorial_attrs:

User attributes
---------------

Zarr arrays and groups support custom key/value attributes, which can be useful for
storing application-specific metadata. For example::

    >>> root = zarr.group()
    >>> root.attrs['foo'] = 'bar'
    >>> z = root.zeros('zzz', shape=(10000, 10000))
    >>> z.attrs['baz'] = 42
    >>> z.attrs['qux'] = [1, 4, 7, 12]
    >>> sorted(root.attrs)
    ['foo']
    >>> 'foo' in root.attrs
    True
    >>> root.attrs['foo']
    'bar'
    >>> sorted(z.attrs)
    ['baz', 'qux']
    >>> z.attrs['baz']
    42
    >>> z.attrs['qux']
    [1, 4, 7, 12]

Internally Zarr uses JSON to store array attributes, so attribute values must be
JSON serializable.

.. _tutorial_indexing:

Advanced indexing
-----------------

As of version 2.2, Zarr arrays support several methods for advanced or "fancy"
indexing, which enable a subset of data items to be extracted or updated in an
array without loading the entire array into memory.

Note that although this functionality is similar to some of the advanced
indexing capabilities available on NumPy arrays and on h5py datasets, **the Zarr
API for advanced indexing is different from both NumPy and h5py**, so please
read this section carefully.  For a complete description of the indexing API,
see the documentation for the :class:`zarr.core.Array` class.

Indexing with coordinate arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Items from a Zarr array can be extracted by providing an integer array of
coordinates. E.g.::

    >>> z = zarr.array(np.arange(10))
    >>> z[:]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> z.get_coordinate_selection([1, 4])
    array([1, 4])

Coordinate arrays can also be used to update data, e.g.::

    >>> z.set_coordinate_selection([1, 4], [-1, -2])
    >>> z[:]
    array([ 0, -1,  2,  3, -2,  5,  6,  7,  8,  9])

For multidimensional arrays, coordinates must be provided for each dimension,
e.g.::

    >>> z = zarr.array(np.arange(15).reshape(3, 5))
    >>> z[:]
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> z.get_coordinate_selection(([0, 2], [1, 3]))
    array([ 1, 13])
    >>> z.set_coordinate_selection(([0, 2], [1, 3]), [-1, -2])
    >>> z[:]
    array([[ 0, -1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, -2, 14]])

For convenience, coordinate indexing is also available via the ``vindex``
property, e.g.::

    >>> z.vindex[[0, 2], [1, 3]]
    array([-1, -2])
    >>> z.vindex[[0, 2], [1, 3]] = [-3, -4]
    >>> z[:]
    array([[ 0, -3,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, -4, 14]])

Indexing with a mask array
~~~~~~~~~~~~~~~~~~~~~~~~~~

Items can also be extracted by providing a Boolean mask. E.g.::

    >>> z = zarr.array(np.arange(10))
    >>> z[:]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> sel = np.zeros_like(z, dtype=bool)
    >>> sel[1] = True
    >>> sel[4] = True
    >>> z.get_mask_selection(sel)
    array([1, 4])
    >>> z.set_mask_selection(sel, [-1, -2])
    >>> z[:]
    array([ 0, -1,  2,  3, -2,  5,  6,  7,  8,  9])

Here's a multidimensional example::

    >>> z = zarr.array(np.arange(15).reshape(3, 5))
    >>> z[:]
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> sel = np.zeros_like(z, dtype=bool)
    >>> sel[0, 1] = True
    >>> sel[2, 3] = True
    >>> z.get_mask_selection(sel)
    array([ 1, 13])
    >>> z.set_mask_selection(sel, [-1, -2])
    >>> z[:]
    array([[ 0, -1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, -2, 14]])

For convenience, mask indexing is also available via the ``vindex`` property,
e.g.::

    >>> z.vindex[sel]
    array([-1, -2])
    >>> z.vindex[sel] = [-3, -4]
    >>> z[:]
    array([[ 0, -3,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, -4, 14]])

Mask indexing is conceptually the same as coordinate indexing, and is
implemented internally via the same machinery. Both styles of indexing allow
selecting arbitrary items from an array, also known as point selection.

Orthogonal indexing
~~~~~~~~~~~~~~~~~~~

Zarr arrays also support methods for orthogonal indexing, which allows
selections to be made along each dimension of an array independently. For
example, this allows selecting a subset of rows and/or columns from a
2-dimensional array. E.g.::

    >>> z = zarr.array(np.arange(15).reshape(3, 5))
    >>> z[:]
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> z.get_orthogonal_selection(([0, 2], slice(None)))  # select first and third rows
    array([[ 0,  1,  2,  3,  4],
           [10, 11, 12, 13, 14]])
    >>> z.get_orthogonal_selection((slice(None), [1, 3]))  # select second and fourth columns
    array([[ 1,  3],
           [ 6,  8],
           [11, 13]])
    >>> z.get_orthogonal_selection(([0, 2], [1, 3]))       # select rows [0, 2] and columns [1, 4]
    array([[ 1,  3],
           [11, 13]])

Data can also be modified, e.g.::

    >>> z.set_orthogonal_selection(([0, 2], [1, 3]), [[-1, -2], [-3, -4]])
    >>> z[:]
    array([[ 0, -1,  2, -2,  4],
           [ 5,  6,  7,  8,  9],
           [10, -3, 12, -4, 14]])

For convenience, the orthogonal indexing functionality is also available via the
``oindex`` property, e.g.::

    >>> z = zarr.array(np.arange(15).reshape(3, 5))
    >>> z.oindex[[0, 2], :]  # select first and third rows
    array([[ 0,  1,  2,  3,  4],
           [10, 11, 12, 13, 14]])
    >>> z.oindex[:, [1, 3]]  # select second and fourth columns
    array([[ 1,  3],
           [ 6,  8],
           [11, 13]])
    >>> z.oindex[[0, 2], [1, 3]]  # select rows [0, 2] and columns [1, 4]
    array([[ 1,  3],
           [11, 13]])
    >>> z.oindex[[0, 2], [1, 3]] = [[-1, -2], [-3, -4]]
    >>> z[:]
    array([[ 0, -1,  2, -2,  4],
           [ 5,  6,  7,  8,  9],
           [10, -3, 12, -4, 14]])

Any combination of integer, slice, 1D integer array and/or 1D Boolean array can
be used for orthogonal indexing.

Indexing fields in structured arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All selection methods support a ``fields`` parameter which allows retrieving or
replacing data for a specific field in an array with a structured dtype. E.g.::

    >>> a = np.array([(b'aaa', 1, 4.2),
    ...               (b'bbb', 2, 8.4),
    ...               (b'ccc', 3, 12.6)],
    ...              dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
    >>> z = zarr.array(a)
    >>> z['foo']
    array([b'aaa', b'bbb', b'ccc'],
          dtype='|S3')
    >>> z['baz']
    array([  4.2,   8.4,  12.6])
    >>> z.get_basic_selection(slice(0, 2), fields='bar')
    array([1, 2], dtype=int32)
    >>> z.get_coordinate_selection([0, 2], fields=['foo', 'baz'])
    array([(b'aaa',   4.2), (b'ccc',  12.6)],
          dtype=[('foo', 'S3'), ('baz', '<f8')])

.. _tutorial_storage:

Storage alternatives
--------------------

Zarr can use any object that implements the ``MutableMapping`` interface from
the :mod:`collections` module in the Python standard library as the store for a
group or an array.

Some pre-defined storage classes are provided in the :mod:`zarr.storage`
module. For example, the :class:`zarr.storage.DirectoryStore` class provides a
``MutableMapping`` interface to a directory on the local file system. This is
used under the hood by the :func:`zarr.convenience.open` function. In other words,
the following code::

    >>> z = zarr.open('data/example.zarr', mode='w', shape=1000000, dtype='i4')

...is short-hand for::

    >>> store = zarr.DirectoryStore('data/example.zarr')
    >>> z = zarr.create(store=store, overwrite=True, shape=1000000, dtype='i4')

...and the following code::

    >>> root = zarr.open('data/example.zarr', mode='w')

...is short-hand for::

    >>> store = zarr.DirectoryStore('data/example.zarr')
    >>> root = zarr.group(store=store, overwrite=True)

Any other compatible storage class could be used in place of
:class:`zarr.storage.DirectoryStore` in the code examples above. For example,
here is an array stored directly into a Zip file, via the
:class:`zarr.storage.ZipStore` class::

    >>> store = zarr.ZipStore('data/example.zip', mode='w')
    >>> root = zarr.group(store=store)
    >>> z = root.zeros('foo/bar', shape=(1000, 1000), chunks=(100, 100), dtype='i4')
    >>> z[:] = 42
    >>> store.close()

Re-open and check that data have been written::

    >>> store = zarr.ZipStore('data/example.zip', mode='r')
    >>> root = zarr.group(store=store)
    >>> z = root['foo/bar']
    >>> z[:]
    array([[42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42],
           ...,
           [42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42],
           [42, 42, 42, ..., 42, 42, 42]], dtype=int32)
    >>> store.close()

Note that there are some limitations on how Zip files can be used, because items
within a Zip file cannot be updated in place. This means that data in the array
should only be written once and write operations should be aligned with chunk
boundaries. Note also that the ``close()`` method must be called after writing
any data to the store, otherwise essential records will not be written to the
underlying zip file.

Another storage alternative is the :class:`zarr.storage.DBMStore` class, added
in Zarr version 2.2. This class allows any DBM-style database to be used for
storing an array or group. Here is an example using a Berkeley DB B-tree
database for storage (requires `bsddb3
<https://www.jcea.es/programacion/pybsddb.htm>`_ to be installed)::

    >>> import bsddb3
    >>> store = zarr.DBMStore('data/example.bdb', open=bsddb3.btopen)
    >>> root = zarr.group(store=store, overwrite=True)
    >>> z = root.zeros('foo/bar', shape=(1000, 1000), chunks=(100, 100), dtype='i4')
    >>> z[:] = 42
    >>> store.close()

Also added in Zarr version 2.2 is the :class:`zarr.storage.LMDBStore` class which
enables the lightning memory-mapped database (LMDB) to be used for storing an array or
group (requires `lmdb <http://lmdb.readthedocs.io/>`_ to be installed)::

    >>> store = zarr.LMDBStore('data/example.lmdb')
    >>> root = zarr.group(store=store, overwrite=True)
    >>> z = root.zeros('foo/bar', shape=(1000, 1000), chunks=(100, 100), dtype='i4')
    >>> z[:] = 42
    >>> store.close()

In Zarr version 2.3 is the :class:`zarr.storage.SQLiteStore` class which
enables the SQLite database to be used for storing an array or group (requires
Python is built with SQLite support)::

    >>> store = zarr.SQLiteStore('data/example.sqldb')
    >>> root = zarr.group(store=store, overwrite=True)
    >>> z = root.zeros('foo/bar', shape=(1000, 1000), chunks=(100, 100), dtype='i4')
    >>> z[:] = 42
    >>> store.close()

Also added in Zarr version 2.3 are two storage classes for interfacing with server-client
databases. The :class:`zarr.storage.RedisStore` class interfaces `Redis <https://redis.io/>`_
(an in memory data structure store), and the :class:`zarr.storage.MongoDB` class interfaces
with `MongoDB <https://www.mongodb.com/>`_ (an object oriented NoSQL database). These stores
respectively require the `redis-py <https://redis-py.readthedocs.io>`_ and
`pymongo <https://api.mongodb.com/python/current/>`_ packages to be installed. 

For compatibility with the `N5 <https://github.com/saalfeldlab/n5>`_ data format, Zarr also provides
an N5 backend (this is currently an experimental feature). Similar to the zip storage class, an
:class:`zarr.n5.N5Store` can be instantiated directly::

    >>> store = zarr.N5Store('data/example.n5')
    >>> root = zarr.group(store=store)
    >>> z = root.zeros('foo/bar', shape=(1000, 1000), chunks=(100, 100), dtype='i4')
    >>> z[:] = 42

For convenience, the N5 backend will automatically be chosen when the filename
ends with `.n5`::

    >>> root = zarr.open('data/example.n5', mode='w')

Distributed/cloud storage
~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to use distributed storage systems. The Dask project has
implementations of the ``MutableMapping`` interface for Amazon S3 (`S3Map
<http://s3fs.readthedocs.io/en/latest/api.html#s3fs.mapping.S3Map>`_), Hadoop
Distributed File System (`HDFSMap
<http://hdfs3.readthedocs.io/en/latest/api.html#hdfs3.mapping.HDFSMap>`_) and
Google Cloud Storage (`GCSMap
<http://gcsfs.readthedocs.io/en/latest/api.html#gcsfs.mapping.GCSMap>`_), which
can be used with Zarr.

Here is an example using S3Map to read an array created previously::

    >>> import s3fs
    >>> import zarr
    >>> s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name='eu-west-2'))
    >>> store = s3fs.S3Map(root='zarr-demo/store', s3=s3, check=False)
    >>> root = zarr.group(store=store)
    >>> z = root['foo/bar/baz']
    >>> z
    <zarr.core.Array '/foo/bar/baz' (21,) |S1>
    >>> z.info
    Name               : /foo/bar/baz
    Type               : zarr.core.Array
    Data type          : |S1
    Shape              : (21,)
    Chunk shape        : (7,)
    Order              : C
    Read-only          : False
    Compressor         : Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=0)
    Store type         : fsspec.mapping.FSMap
    No. bytes          : 21
    Chunks initialized : 3/3
    >>> z[:]
    array([b'H', b'e', b'l', b'l', b'o', b' ', b'f', b'r', b'o', b'm', b' ',
           b't', b'h', b'e', b' ', b'c', b'l', b'o', b'u', b'd', b'!'],
          dtype='|S1')
    >>> z[:].tostring()
    b'Hello from the cloud!'

Zarr now also has a builtin storage backend for Azure Blob Storage.
The class is :class:`zarr.storage.ABSStore` (requires
`azure-storage-blob <https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python>`_
to be installed)::

    >>> store = zarr.ABSStore(container='test', prefix='zarr-testing', blob_service_kwargs={'is_emulated': True})  # doctest: +SKIP
    >>> root = zarr.group(store=store, overwrite=True)  # doctest: +SKIP
    >>> z = root.zeros('foo/bar', shape=(1000, 1000), chunks=(100, 100), dtype='i4')  # doctest: +SKIP
    >>> z[:] = 42  # doctest: +SKIP

When using an actual storage account, provide ``account_name`` and
``account_key`` arguments to :class:`zarr.storage.ABSStore`, the
above client is just testing against the emulator. Please also note
that this is an experimental feature.

Note that retrieving data from a remote service via the network can be significantly
slower than retrieving data from a local file system, and will depend on network latency
and bandwidth between the client and server systems. If you are experiencing poor
performance, there are several things you can try. One option is to increase the array
chunk size, which will reduce the number of chunks and thus reduce the number of network
round-trips required to retrieve data for an array (and thus reduce the impact of network
latency). Another option is to try to increase the compression ratio by changing
compression options or trying a different compressor (which will reduce the impact of
limited network bandwidth).

As of version 2.2, Zarr also provides the :class:`zarr.storage.LRUStoreCache`
which can be used to implement a local in-memory cache layer over a remote
store. E.g.::

    >>> s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name='eu-west-2'))
    >>> store = s3fs.S3Map(root='zarr-demo/store', s3=s3, check=False)
    >>> cache = zarr.LRUStoreCache(store, max_size=2**28)
    >>> root = zarr.group(store=cache)
    >>> z = root['foo/bar/baz']
    >>> from timeit import timeit
    >>> # first data access is relatively slow, retrieved from store
    ... timeit('print(z[:].tostring())', number=1, globals=globals())  # doctest: +SKIP
    b'Hello from the cloud!'
    0.1081731989979744
    >>> # second data access is faster, uses cache
    ... timeit('print(z[:].tostring())', number=1, globals=globals())  # doctest: +SKIP
    b'Hello from the cloud!'
    0.0009490990014455747

If you are still experiencing poor performance with distributed/cloud storage,
please raise an issue on the GitHub issue tracker with any profiling data you
can provide, as there may be opportunities to optimise further either within
Zarr or within the mapping interface to the storage.

IO with ``fsspec``
~~~~~~~~~~~~~~~~~~

As of version 2.5, zarr supports passing URLs directly to `fsspec`_,
and having it create the "mapping" instance automatically. This means, that
for all of the backend storage implementations `supported by fsspec`_,
you can skip importing and configuring the storage explicitly.
For example::

    >>> g = zarr.open_group("s3://zarr-demo/store", storage_options={'anon': True})   # doctest: +SKIP
    >>> g['foo/bar/baz'][:].tobytes()  # doctest: +SKIP
    b'Hello from the cloud!'

The provision of the protocol specifier "s3://" will select the correct backend.
Notice the kwargs ``storage_options``, used to pass parameters to that backend.

As of version 2.6, write mode and complex URLs are also supported, such as::

    >>> g = zarr.open_group("simplecache::s3://zarr-demo/store",
    ...                     storage_options={"s3": {'anon': True}})  # doctest: +SKIP
    >>> g['foo/bar/baz'][:].tobytes()  # downloads target file  # doctest: +SKIP
    b'Hello from the cloud!'
    >>> g['foo/bar/baz'][:].tobytes()  # uses cached file  # doctest: +SKIP
    b'Hello from the cloud!'

The second invocation here will be much faster. Note that the ``storage_options``
have become more complex here, to account for the two parts of the supplied
URL.

.. _fsspec: https://filesystem-spec.readthedocs.io/en/latest/

.. _supported by fsspec: https://filesystem-spec.readthedocs.io/en/latest/api.html#built-in-implementations

.. _tutorial_copy:

Consolidating metadata
~~~~~~~~~~~~~~~~~~~~~~

Since there is a significant overhead for every connection to a cloud object
store such as S3, the pattern described in the previous section may incur
significant latency while scanning the metadata of the array hierarchy, even
though each individual metadata object is small.  For cases such as these, once
the data are static and can be regarded as read-only, at least for the
metadata/structure of the array hierarchy, the many metadata objects can be
consolidated into a single one via
:func:`zarr.convenience.consolidate_metadata`. Doing this can greatly increase
the speed of reading the array metadata, e.g.::

   >>> zarr.consolidate_metadata(store)  # doctest: +SKIP

This creates a special key with a copy of all of the metadata from all of the
metadata objects in the store.

Later, to open a Zarr store with consolidated metadata, use
:func:`zarr.convenience.open_consolidated`, e.g.::

   >>> root = zarr.open_consolidated(store)  # doctest: +SKIP

This uses the special key to read all of the metadata in a single call to the
backend storage.

Note that, the hierarchy could still be opened in the normal way and altered,
causing the consolidated metadata to become out of sync with the real state of
the array hierarchy. In this case,
:func:`zarr.convenience.consolidate_metadata` would need to be called again.

To protect against consolidated metadata accidentally getting out of sync, the
root group returned by :func:`zarr.convenience.open_consolidated` is read-only
for the metadata, meaning that no new groups or arrays can be created, and
arrays cannot be resized. However, data values with arrays can still be updated.

Copying/migrating data
----------------------

If you have some data in an HDF5 file and would like to copy some or all of it
into a Zarr group, or vice-versa, the :func:`zarr.convenience.copy` and
:func:`zarr.convenience.copy_all` functions can be used. Here's an example
copying a group named 'foo' from an HDF5 file to a Zarr group::

    >>> import h5py
    >>> import zarr
    >>> import numpy as np
    >>> source = h5py.File('data/example.h5', mode='w')
    >>> foo = source.create_group('foo')
    >>> baz = foo.create_dataset('bar/baz', data=np.arange(100), chunks=(50,))
    >>> spam = source.create_dataset('spam', data=np.arange(100, 200), chunks=(30,))
    >>> zarr.tree(source)
    /
     ├── foo
     │   └── bar
     │       └── baz (100,) int64
     └── spam (100,) int64
    >>> dest = zarr.open_group('data/example.zarr', mode='w')
    >>> from sys import stdout
    >>> zarr.copy(source['foo'], dest, log=stdout)
    copy /foo
    copy /foo/bar
    copy /foo/bar/baz (100,) int64
    all done: 3 copied, 0 skipped, 800 bytes copied
    (3, 0, 800)
    >>> dest.tree()  # N.B., no spam
    /
     └── foo
         └── bar
             └── baz (100,) int64
    >>> source.close()

If rather than copying a single group or array you would like to copy all
groups and arrays, use :func:`zarr.convenience.copy_all`, e.g.::

    >>> source = h5py.File('data/example.h5', mode='r')
    >>> dest = zarr.open_group('data/example2.zarr', mode='w')
    >>> zarr.copy_all(source, dest, log=stdout)
    copy /foo
    copy /foo/bar
    copy /foo/bar/baz (100,) int64
    copy /spam (100,) int64
    all done: 4 copied, 0 skipped, 1,600 bytes copied
    (4, 0, 1600)
    >>> dest.tree()
    /
     ├── foo
     │   └── bar
     │       └── baz (100,) int64
     └── spam (100,) int64

If you need to copy data between two Zarr groups, the
:func:`zarr.convenience.copy` and :func:`zarr.convenience.copy_all` functions can
be used and provide the most flexibility. However, if you want to copy data
in the most efficient way possible, without changing any configuration options,
the :func:`zarr.convenience.copy_store` function can be used. This function
copies data directly between the underlying stores, without any decompression or
re-compression, and so should be faster. E.g.::

    >>> import zarr
    >>> import numpy as np
    >>> store1 = zarr.DirectoryStore('data/example.zarr')
    >>> root = zarr.group(store1, overwrite=True)
    >>> baz = root.create_dataset('foo/bar/baz', data=np.arange(100), chunks=(50,))
    >>> spam = root.create_dataset('spam', data=np.arange(100, 200), chunks=(30,))
    >>> root.tree()
    /
     ├── foo
     │   └── bar
     │       └── baz (100,) int64
     └── spam (100,) int64
    >>> from sys import stdout
    >>> store2 = zarr.ZipStore('data/example.zip', mode='w')
    >>> zarr.copy_store(store1, store2, log=stdout)
    copy .zgroup
    copy foo/.zgroup
    copy foo/bar/.zgroup
    copy foo/bar/baz/.zarray
    copy foo/bar/baz/0
    copy foo/bar/baz/1
    copy spam/.zarray
    copy spam/0
    copy spam/1
    copy spam/2
    copy spam/3
    all done: 11 copied, 0 skipped, 1,138 bytes copied
    (11, 0, 1138)
    >>> new_root = zarr.group(store2)
    >>> new_root.tree()
    /
     ├── foo
     │   └── bar
     │       └── baz (100,) int64
     └── spam (100,) int64
    >>> new_root['foo/bar/baz'][:]
    array([ 0,  1,  2,  ..., 97, 98, 99])
    >>> store2.close()  # zip stores need to be closed

.. _tutorial_strings:

String arrays
-------------

There are several options for storing arrays of strings.

If your strings are all ASCII strings, and you know the maximum length of the string in
your array, then you can use an array with a fixed-length bytes dtype. E.g.::

    >>> z = zarr.zeros(10, dtype='S6')
    >>> z
    <zarr.core.Array (10,) |S6>
    >>> z[0] = b'Hello'
    >>> z[1] = b'world!'
    >>> z[:]
    array([b'Hello', b'world!', b'', b'', b'', b'', b'', b'', b'', b''],
          dtype='|S6')

A fixed-length unicode dtype is also available, e.g.::

    >>> greetings = ['¡Hola mundo!', 'Hej Världen!', 'Servus Woid!', 'Hei maailma!',
    ...              'Xin chào thế giới', 'Njatjeta Botë!', 'Γεια σου κόσμε!',
    ...              'こんにちは世界', '世界，你好！', 'Helló, világ!', 'Zdravo svete!',
    ...              'เฮลโลเวิลด์']
    >>> text_data = greetings * 10000
    >>> z = zarr.array(text_data, dtype='U20')
    >>> z
    <zarr.core.Array (120000,) <U20>
    >>> z[:]
    array(['¡Hola mundo!', 'Hej Världen!', 'Servus Woid!', ...,
           'Helló, világ!', 'Zdravo svete!', 'เฮลโลเวิลด์'],
          dtype='<U20')

For variable-length strings, the ``object`` dtype can be used, but a codec must be
provided to encode the data (see also :ref:`tutorial_objects` below). At the time of
writing there are four codecs available that can encode variable length string
objects: :class:`numcodecs.VLenUTF8`, :class:`numcodecs.JSON`, :class:`numcodecs.MsgPack`.
and :class:`numcodecs.Pickle`. E.g. using ``VLenUTF8``::

    >>> import numcodecs
    >>> z = zarr.array(text_data, dtype=object, object_codec=numcodecs.VLenUTF8())
    >>> z
    <zarr.core.Array (120000,) object>
    >>> z.filters
    [VLenUTF8()]
    >>> z[:]
    array(['¡Hola mundo!', 'Hej Världen!', 'Servus Woid!', ...,
           'Helló, világ!', 'Zdravo svete!', 'เฮลโลเวิลด์'], dtype=object)

As a convenience, ``dtype=str`` (or ``dtype=unicode`` on Python 2.7) can be used, which
is a short-hand for ``dtype=object, object_codec=numcodecs.VLenUTF8()``, e.g.::

    >>> z = zarr.array(text_data, dtype=str)
    >>> z
    <zarr.core.Array (120000,) object>
    >>> z.filters
    [VLenUTF8()]
    >>> z[:]
    array(['¡Hola mundo!', 'Hej Världen!', 'Servus Woid!', ...,
           'Helló, világ!', 'Zdravo svete!', 'เฮลโลเวิลด์'], dtype=object)

Variable-length byte strings are also supported via ``dtype=object``. Again an
``object_codec`` is required, which can be one of :class:`numcodecs.VLenBytes` or
:class:`numcodecs.Pickle`. For convenience, ``dtype=bytes`` (or ``dtype=str`` on Python
2.7) can be used as a short-hand for ``dtype=object, object_codec=numcodecs.VLenBytes()``,
e.g.::

    >>> bytes_data = [g.encode('utf-8') for g in greetings] * 10000
    >>> z = zarr.array(bytes_data, dtype=bytes)
    >>> z
    <zarr.core.Array (120000,) object>
    >>> z.filters
    [VLenBytes()]
    >>> z[:]
    array([b'\xc2\xa1Hola mundo!', b'Hej V\xc3\xa4rlden!', b'Servus Woid!',
           ..., b'Hell\xc3\xb3, vil\xc3\xa1g!', b'Zdravo svete!',
           b'\xe0\xb9\x80\xe0\xb8\xae\xe0\xb8\xa5\xe0\xb9\x82\xe0\xb8\xa5\xe0\xb9\x80\xe0\xb8\xa7\xe0\xb8\xb4\xe0\xb8\xa5\xe0\xb8\x94\xe0\xb9\x8c'], dtype=object)

If you know ahead of time all the possible string values that can occur, you could
also use the :class:`numcodecs.Categorize` codec to encode each unique string value as an
integer. E.g.::

    >>> categorize = numcodecs.Categorize(greetings, dtype=object)
    >>> z = zarr.array(text_data, dtype=object, object_codec=categorize)
    >>> z
    <zarr.core.Array (120000,) object>
    >>> z.filters
    [Categorize(dtype='|O', astype='|u1', labels=['¡Hola mundo!', 'Hej Världen!', 'Servus Woid!', ...])]
    >>> z[:]
    array(['¡Hola mundo!', 'Hej Världen!', 'Servus Woid!', ...,
           'Helló, világ!', 'Zdravo svete!', 'เฮลโลเวิลด์'], dtype=object)


.. _tutorial_objects:

Object arrays
-------------

Zarr supports arrays with an "object" dtype. This allows arrays to contain any type of
object, such as variable length unicode strings, or variable length arrays of numbers, or
other possibilities. When creating an object array, a codec must be provided via the
``object_codec`` argument. This codec handles encoding (serialization) of Python objects.
The best codec to use will depend on what type of objects are present in the array.

At the time of writing there are three codecs available that can serve as a general
purpose object codec and support encoding of a mixture of object types:
:class:`numcodecs.JSON`, :class:`numcodecs.MsgPack`. and :class:`numcodecs.Pickle`.

For example, using the JSON codec::

    >>> z = zarr.empty(5, dtype=object, object_codec=numcodecs.JSON())
    >>> z[0] = 42
    >>> z[1] = 'foo'
    >>> z[2] = ['bar', 'baz', 'qux']
    >>> z[3] = {'a': 1, 'b': 2.2}
    >>> z[:]
    array([42, 'foo', list(['bar', 'baz', 'qux']), {'a': 1, 'b': 2.2}, None], dtype=object)

Not all codecs support encoding of all object types. The
:class:`numcodecs.Pickle` codec is the most flexible, supporting encoding any type
of Python object. However, if you are sharing data with anyone other than yourself, then
Pickle is not recommended as it is a potential security risk. This is because malicious
code can be embedded within pickled data. The JSON and MsgPack codecs do not have any
security issues and support encoding of unicode strings, lists and dictionaries.
MsgPack is usually faster for both encoding and decoding.

Ragged arrays
~~~~~~~~~~~~~

If you need to store an array of arrays, where each member array can be of any length
and stores the same primitive type (a.k.a. a ragged array), the
:class:`numcodecs.VLenArray` codec can be used, e.g.::

    >>> z = zarr.empty(4, dtype=object, object_codec=numcodecs.VLenArray(int))
    >>> z
    <zarr.core.Array (4,) object>
    >>> z.filters
    [VLenArray(dtype='<i8')]
    >>> z[0] = np.array([1, 3, 5])
    >>> z[1] = np.array([4])
    >>> z[2] = np.array([7, 9, 14])
    >>> z[:]
    array([array([1, 3, 5]), array([4]), array([ 7,  9, 14]),
           array([], dtype=int64)], dtype=object)

As a convenience, ``dtype='array:T'`` can be used as a short-hand for
``dtype=object, object_codec=numcodecs.VLenArray('T')``, where 'T' can be any NumPy
primitive dtype such as 'i4' or 'f8'. E.g.::

    >>> z = zarr.empty(4, dtype='array:i8')
    >>> z
    <zarr.core.Array (4,) object>
    >>> z.filters
    [VLenArray(dtype='<i8')]
    >>> z[0] = np.array([1, 3, 5])
    >>> z[1] = np.array([4])
    >>> z[2] = np.array([7, 9, 14])
    >>> z[:]
    array([array([1, 3, 5]), array([4]), array([ 7,  9, 14]),
           array([], dtype=int64)], dtype=object)

.. _tutorial_chunks:

Chunk optimizations
-------------------

.. _tutorial_chunks_shape:

Chunk size and shape
~~~~~~~~~~~~~~~~~~~~

In general, chunks of at least 1 megabyte (1M) uncompressed size seem to provide
better performance, at least when using the Blosc compression library.

The optimal chunk shape will depend on how you want to access the data. E.g.,
for a 2-dimensional array, if you only ever take slices along the first
dimension, then chunk across the second dimenson. If you know you want to chunk
across an entire dimension you can use ``None`` or ``-1`` within the ``chunks``
argument, e.g.::

    >>> z1 = zarr.zeros((10000, 10000), chunks=(100, None), dtype='i4')
    >>> z1.chunks
    (100, 10000)

Alternatively, if you only ever take slices along the second dimension, then
chunk across the first dimension, e.g.::

    >>> z2 = zarr.zeros((10000, 10000), chunks=(None, 100), dtype='i4')
    >>> z2.chunks
    (10000, 100)

If you require reasonable performance for both access patterns then you need to
find a compromise, e.g.::

    >>> z3 = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4')
    >>> z3.chunks
    (1000, 1000)

If you are feeling lazy, you can let Zarr guess a chunk shape for your data by
providing ``chunks=True``, although please note that the algorithm for guessing
a chunk shape is based on simple heuristics and may be far from optimal. E.g.::

    >>> z4 = zarr.zeros((10000, 10000), chunks=True, dtype='i4')
    >>> z4.chunks
    (625, 625)

If you know you are always going to be loading the entire array into memory, you
can turn off chunks by providing ``chunks=False``, in which case there will be
one single chunk for the array::

    >>> z5 = zarr.zeros((10000, 10000), chunks=False, dtype='i4')
    >>> z5.chunks
    (10000, 10000)

.. _tutorial_chunks_order:

Chunk memory layout
~~~~~~~~~~~~~~~~~~~

The order of bytes **within each chunk** of an array can be changed via the
``order`` keyword argument, to use either C or Fortran layout. For
multi-dimensional arrays, these two layouts may provide different compression
ratios, depending on the correlation structure within the data. E.g.::

    >>> a = np.arange(100000000, dtype='i4').reshape(10000, 10000).T
    >>> c = zarr.array(a, chunks=(1000, 1000))
    >>> c.info
    Type               : zarr.core.Array
    Data type          : int32
    Shape              : (10000, 10000)
    Chunk shape        : (1000, 1000)
    Order              : C
    Read-only          : False
    Compressor         : Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=0)
    Store type         : builtins.dict
    No. bytes          : 400000000 (381.5M)
    No. bytes stored   : 6696010 (6.4M)
    Storage ratio      : 59.7
    Chunks initialized : 100/100
    >>> f = zarr.array(a, chunks=(1000, 1000), order='F')
    >>> f.info
    Type               : zarr.core.Array
    Data type          : int32
    Shape              : (10000, 10000)
    Chunk shape        : (1000, 1000)
    Order              : F
    Read-only          : False
    Compressor         : Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=0)
    Store type         : builtins.dict
    No. bytes          : 400000000 (381.5M)
    No. bytes stored   : 4684636 (4.5M)
    Storage ratio      : 85.4
    Chunks initialized : 100/100

In the above example, Fortran order gives a better compression ratio. This is an
artifical example but illustrates the general point that changing the order of
bytes within chunks of an array may improve the compression ratio, depending on
the structure of the data, the compression algorithm used, and which compression
filters (e.g., byte-shuffle) have been applied.

.. _tutorial_sync:

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
with thread synchronization::

    >>> z = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4',
    ...                 synchronizer=zarr.ThreadSynchronizer())
    >>> z
    <zarr.core.Array (10000, 10000) int32>

This array is safe to read or write within a multi-threaded program.

Zarr also provides support for process synchronization via file locking,
provided that all processes have access to a shared file system, and provided
that the underlying file system supports file locking (which is not the case for
some networked file systems). E.g.::

    >>> synchronizer = zarr.ProcessSynchronizer('data/example.sync')
    >>> z = zarr.open_array('data/example', mode='w', shape=(10000, 10000),
    ...                     chunks=(1000, 1000), dtype='i4',
    ...                     synchronizer=synchronizer)
    >>> z
    <zarr.core.Array (10000, 10000) int32>

This array is safe to read or write from multiple processes.

When using multiple processes to parallelize reads or writes on arrays using the Blosc
compression library, it may be necessary to set ``numcodecs.blosc.use_threads = False``,
as otherwise Blosc may share incorrect global state amongst processes causing programs
to hang. See also the section on :ref:`tutorial_tips_blosc` below.

Please note that support for parallel computing is an area of ongoing research
and development. If you are using Zarr for parallel computing, we welcome
feedback, experience, discussion, ideas and advice, particularly about issues
related to data integrity and performance.

.. _tutorial_pickle:

Pickle support
--------------

Zarr arrays and groups can be pickled, as long as the underlying store object can be
pickled. Instances of any of the storage classes provided in the :mod:`zarr.storage`
module can be pickled, as can the built-in ``dict`` class which can also be used for
storage.

Note that if an array or group is backed by an in-memory store like a ``dict`` or
:class:`zarr.storage.MemoryStore`, then when it is pickled all of the store data will be
included in the pickled data. However, if an array or group is backed by a persistent
store like a :class:`zarr.storage.DirectoryStore`, :class:`zarr.storage.ZipStore` or
:class:`zarr.storage.DBMStore` then the store data **are not** pickled. The only thing
that is pickled is the necessary parameters to allow the store to re-open any
underlying files or databases upon being unpickled.

E.g., pickle/unpickle an in-memory array::

    >>> import pickle
    >>> z1 = zarr.array(np.arange(100000))
    >>> s = pickle.dumps(z1)
    >>> len(s) > 5000  # relatively large because data have been pickled
    True
    >>> z2 = pickle.loads(s)
    >>> z1 == z2
    True
    >>> np.all(z1[:] == z2[:])
    True

E.g., pickle/unpickle an array stored on disk::

    >>> z3 = zarr.open('data/walnuts.zarr', mode='w', shape=100000, dtype='i8')
    >>> z3[:] = np.arange(100000)
    >>> s = pickle.dumps(z3)
    >>> len(s) < 200  # small because no data have been pickled
    True
    >>> z4 = pickle.loads(s)
    >>> z3 == z4
    True
    >>> np.all(z3[:] == z4[:])
    True

.. _tutorial_datetime:

Datetimes and timedeltas
------------------------

NumPy's ``datetime64`` ('M8') and ``timedelta64`` ('m8') dtypes are supported for Zarr
arrays, as long as the units are specified. E.g.::

    >>> z = zarr.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='M8[D]')
    >>> z
    <zarr.core.Array (3,) datetime64[D]>
    >>> z[:]
    array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64[D]')
    >>> z[0]
    numpy.datetime64('2007-07-13')
    >>> z[0] = '1999-12-31'
    >>> z[:]
    array(['1999-12-31', '2006-01-13', '2010-08-13'], dtype='datetime64[D]')

.. _tutorial_tips:

Usage tips
----------

.. _tutorial_tips_copy:

Copying large arrays
~~~~~~~~~~~~~~~~~~~~

Data can be copied between large arrays without needing much memory, e.g.::

    >>> z1 = zarr.empty((10000, 10000), chunks=(1000, 1000), dtype='i4')
    >>> z1[:] = 42
    >>> z2 = zarr.empty_like(z1)
    >>> z2[:] = z1

Internally the example above works chunk-by-chunk, extracting only the data from
``z1`` required to fill each chunk in ``z2``. The source of the data (``z1``)
could equally be an h5py Dataset.

.. _tutorial_tips_blosc:

Configuring Blosc
~~~~~~~~~~~~~~~~~

The Blosc compressor is able to use multiple threads internally to accelerate
compression and decompression. By default, Blosc uses up to 8
internal threads. The number of Blosc threads can be changed to increase or
decrease this number, e.g.::

    >>> from numcodecs import blosc
    >>> blosc.set_nthreads(2)  # doctest: +SKIP
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

Please note that if Zarr is being used within a multi-process program, Blosc may not
be safe to use in multi-threaded mode and may cause the program to hang. If using Blosc
in a multi-process program then it is recommended to set ``blosc.use_threads = False``.
