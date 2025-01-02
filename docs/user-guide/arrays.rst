.. _user-guide-arrays:

Working with arrays
===================

.. ipython:: python
   :suppress:

   rm -r data/

Creating an array
-----------------

Zarr has several functions for creating arrays. For example:

.. ipython:: python

   import zarr

   store = {}
   # TODO: replace with `create_array` after #2463
   z = zarr.create(
       store=store,
       mode="w",
       shape=(10000, 10000),
       chunks=(1000, 1000),
       dtype="i4"
   )
   z

The code above creates a 2-dimensional array of 32-bit integers with 10000 rows
and 10000 columns, divided into chunks where each chunk has 1000 rows and 1000
columns (and so there will be 100 chunks in total). The data is written to a
:class:`zarr.storage.MemoryStore` (e.g. an in-memory dict). See
:ref:`user-guide-persist` for details on storing arrays in other stores.

For a complete list of array creation routines see the :mod:`zarr`
module documentation.

.. _user-guide-array:

Reading and writing data
------------------------

Zarr arrays support a similar interface to `NumPy <https://numpy.org/doc/stable/>`_
arrays for reading and writing data. For example, the entire array can be filled
with a scalar value:

.. ipython:: python

   z[:] = 42

Regions of the array can also be written to, e.g.:

.. ipython:: python

   import numpy as np

   z[0, :] = np.arange(10000)
   z[:, 0] = np.arange(10000)

The contents of the array can be retrieved by slicing, which will load the
requested region into memory as a NumPy array, e.g.:

.. ipython:: python

   z[0, 0]
   z[-1, -1]
   z[0, :]
   z[:, 0]
   z[:]

Read more about NumPy-style indexing can be found in the
`NumPy documentation <https://numpy.org/doc/stable/user/basics.indexing.html>`_.

.. _user-guide-persist:

Persistent arrays
-----------------

In the examples above, compressed data for each chunk of the array was stored in
main memory. Zarr arrays can also be stored on a file system, enabling
persistence of data between sessions. To do this, we can change the store
argument to point to a filesystem path:

.. ipython:: python

   # TODO: replace with `open_array` after #2463
   z1 = zarr.open(
       store='data/example-2.zarr',
       mode='w',
       shape=(10000, 10000),
       chunks=(1000, 1000),
       dtype='i4'
   )

The array above will store its configuration metadata and all compressed chunk
data in a directory called ``'data/example-2.zarr'`` relative to the current working
directory. The :func:`zarr.open` function provides a convenient way
to create a new persistent array or continue working with an existing
array. Note that although the function is called "open", there is no need to
close an array: data are automatically flushed to disk, and files are
automatically closed whenever an array is modified.

Persistent arrays support the same interface for reading and writing data,
e.g.:

.. ipython:: python

   z1[:] = 42
   z1[0, :] = np.arange(10000)
   z1[:, 0] = np.arange(10000)

Check that the data have been written and can be read again:

.. ipython:: python

   # TODO: replace with `open_array` after #2463
   z2 = zarr.open('data/example-2.zarr', mode='r')
   np.all(z1[:] == z2[:])

If you are just looking for a fast and convenient way to save NumPy arrays to
disk then load back into memory later, the functions
:func:`zarr.save` and :func:`zarr.load` may be
useful. E.g.:

.. ipython:: python

   a = np.arange(10)
   zarr.save('data/example-3.zarr', a)
   zarr.load('data/example-3.zarr')

Please note that there are a number of other options for persistent array
storage, see the :ref:`Storage Guide <user-guide-storage>` guide for more details.

.. _user-guide-resize:

Resizing and appending
----------------------

A Zarr array can be resized, which means that any of its dimensions can be
increased or decreased in length. For example:

.. ipython:: python

   z = zarr.zeros(
      store="data/example-4.zarr",
      shape=(10000, 10000),
      chunks=(1000, 1000)
   )
   z[:] = 42
   z.shape
   z.resize((20000, 10000))
   z.shape

Note that when an array is resized, the underlying data are not rearranged in
any way. If one or more dimensions are shrunk, any chunks falling outside the
new array shape will be deleted from the underlying store.

:func:`zarr.Array.append` is provided as a convenience function, which can be
used to append data to any axis. E.g.:

.. ipython:: python

   a = np.arange(10000000, dtype='i4').reshape(10000, 1000)
   # TODO: replace with create_array after #2463
   z = zarr.array(store="data/example-5", data=a, chunks=(1000, 100))
   z.shape
   z.append(a)
   z.append(np.vstack([a, a]), axis=1)
   z.shape

.. _user-guide-compress:

Compressors
-----------

A number of different compressors can be used with Zarr. A separate package
called NumCodecs_ is available which provides a common interface to various
compressor libraries including Blosc, Zstandard, LZ4, Zlib, BZ2 and
LZMA. Different compressors can be provided via the ``compressor`` keyword
argument accepted by all array creation functions. For example:

.. ipython:: python

   from numcodecs import Blosc

   compressor = None  # TODO: Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
   data = np.arange(100000000, dtype='i4').reshape(10000, 10000)
   # TODO: remove zarr_format and replace with create_array after #2463
   z = zarr.array(store="data/example-6.zarr", data=data, chunks=(1000, 1000), compressor=compressor, zarr_format=2)
   None  # TODO: z.compressor

This array above will use Blosc as the primary compressor, using the Zstandard
algorithm (compression level 3) internally within Blosc, and with the
bit-shuffle filter applied.

When using a compressor, it can be useful to get some diagnostics on the
compression ratio. Zarr arrays provide the :property:`zarr.Array.info` property
which can be used to print useful diagnostics, e.g.:

.. ipython:: python

   z.info

The :func:`zarr.Array.info_complete` method inspects the underlying store and
prints additional diagnostics, e.g.:

.. ipython:: python

   z.info_complete()

.. note::
   :func:`zarr.Array.info_complete` will inspect the underlying store and may
   be slow for large arrays. Use :property:`zarr.Array.info` if detailed storage
   statistics are not needed.

If you don't specify a compressor, by default Zarr uses the Blosc
compressor. Blosc is generally very fast and can be configured in a variety of
ways to improve the compression ratio for different types of data. Blosc is in
fact a "meta-compressor", which means that it can use a number of different
compression algorithms internally to compress the data. Blosc also provides
highly optimized implementations of byte- and bit-shuffle filters, which can
improve compression ratios for some data. A list of the internal compression
libraries available within Blosc can be obtained via:

.. ipython:: python

   from numcodecs import blosc

   blosc.list_compressors()

In addition to Blosc, other compression libraries can also be used. For example,
here is an array using Zstandard compression, level 1:

.. ipython:: python

   from numcodecs import Zstd
   # TODO: remove zarr_format and replace with create_array after #2463
   z = zarr.array(
       store="data/example-7.zarr",
       data=np.arange(100000000, dtype='i4').reshape(10000, 10000),
       chunks=(1000, 1000),
       compressor=Zstd(level=1),
       zarr_format=2  # TODO: remove zarr_format
   )
   None  # TODO: z.compressor

Here is an example using LZMA with a custom filter pipeline including LZMA's
built-in delta filter:

.. ipython:: python

   import lzma
   from numcodecs import LZMA

   lzma_filters = [dict(id=lzma.FILTER_DELTA, dist=4), dict(id=lzma.FILTER_LZMA2, preset=1)]
   compressor = LZMA(filters=lzma_filters)
   # TODO: remove zarr_format and replace with create_array after #2463
   z = zarr.array(
       np.arange(100000000, dtype='i4').reshape(10000, 10000),
       chunks=(1000, 1000),
       compressor=compressor,
       zarr_format=2
   )
   None  # TODO: z.compressor

The default compressor can be changed by setting the value of the using Zarr's
:ref:`user-guide-config`, e.g.:

.. ipython:: python

   with zarr.config.set({'array.v2_default_compressor.numeric': 'blosc'}):
       z = zarr.zeros(100000000, chunks=1000000, zarr_format=2)
   z.metadata.filters
   z.metadata.compressor


To disable compression, set ``compressor=None`` when creating an array, e.g.:

.. ipython:: python

   # TODO: remove zarr_format
   z = zarr.zeros(100000000, chunks=1000000, compressor=None, zarr_format=2)
   z
.. _user-guide-filters:

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

Here is an example using a delta filter with the Blosc compressor:

.. ipython:: python

   from numcodecs import Blosc, Delta

   filters = [Delta(dtype='i4')]
   compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
   data = np.arange(100000000, dtype='i4').reshape(10000, 10000)
   # TODO: remove zarr_format and replace with create_array after #2463
   z = zarr.array(data, chunks=(1000, 1000), filters=filters, compressor=compressor, zarr_format=2)
   z.info

For more information about available filter codecs, see the `Numcodecs
<https://numcodecs.readthedocs.io/>`_ documentation.

.. _user-guide-indexing:

Advanced indexing
-----------------

Zarr arrays support several methods for advanced or "fancy"
indexing, which enable a subset of data items to be extracted or updated in an
array without loading the entire array into memory.

Note that although this functionality is similar to some of the advanced
indexing capabilities available on NumPy arrays and on h5py datasets, **the Zarr
API for advanced indexing is different from both NumPy and h5py**, so please
read this section carefully.  For a complete description of the indexing API,
see the documentation for the :class:`zarr.Array` class.

Indexing with coordinate arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Items from a Zarr array can be extracted by providing an integer array of
coordinates. E.g.:

.. ipython:: python

   # TODO: replace with create_array after #2463
   z = zarr.array(np.arange(10) ** 2)
   z[:]
   z.get_coordinate_selection([2, 5])

Coordinate arrays can also be used to update data, e.g.:

.. ipython:: python

   z.set_coordinate_selection([2, 5], [-1, -2])
   z[:]

For multidimensional arrays, coordinates must be provided for each dimension,
e.g.:

.. ipython:: python

   # TODO: replace with create_array after #2463
   z = zarr.array(np.arange(15).reshape(3, 5))
   z[:]
   z.get_coordinate_selection(([0, 2], [1, 3]))
   z.set_coordinate_selection(([0, 2], [1, 3]), [-1, -2])
   z[:]

For convenience, coordinate indexing is also available via the ``vindex``
property, as well as the square bracket operator, e.g.:

.. ipython:: python

   z.vindex[[0, 2], [1, 3]]
   z.vindex[[0, 2], [1, 3]] = [-3, -4]
   z[:]
   z[[0, 2], [1, 3]]

When the indexing arrays have different shapes, they are broadcast together.
That is, the following two calls are equivalent:

.. ipython:: python

   z[1, [1, 3]]
   z[[1, 1], [1, 3]]

Indexing with a mask array
~~~~~~~~~~~~~~~~~~~~~~~~~~

Items can also be extracted by providing a Boolean mask. E.g.:

.. ipython:: python

   # TODO: replace with create_array after #2463
   z = zarr.array(np.arange(10) ** 2)
   z[:]
   sel = np.zeros_like(z, dtype=bool)
   sel[2] = True
   sel[5] = True
   z.get_mask_selection(sel)
   z.set_mask_selection(sel, [-1, -2])
   z[:]

Here's a multidimensional example:

.. ipython:: python

   # TODO: replace with create_array after #2463
   z = zarr.array(np.arange(15).reshape(3, 5))
   z[:]
   sel = np.zeros_like(z, dtype=bool)
   sel[0, 1] = True
   sel[2, 3] = True
   z.get_mask_selection(sel)
   z.set_mask_selection(sel, [-1, -2])
   z[:]

For convenience, mask indexing is also available via the ``vindex`` property,
e.g.:

.. ipython:: python

   z.vindex[sel]
   z.vindex[sel] = [-3, -4]
   z[:]

Mask indexing is conceptually the same as coordinate indexing, and is
implemented internally via the same machinery. Both styles of indexing allow
selecting arbitrary items from an array, also known as point selection.

Orthogonal indexing
~~~~~~~~~~~~~~~~~~~

Zarr arrays also support methods for orthogonal indexing, which allows
selections to be made along each dimension of an array independently. For
example, this allows selecting a subset of rows and/or columns from a
2-dimensional array. E.g.:

.. ipython:: python

   # TODO: replace with create_array after #2463
   z = zarr.array(np.arange(15).reshape(3, 5))
   z[:]
   z.get_orthogonal_selection(([0, 2], slice(None)))  # select first and third rows
   z.get_orthogonal_selection((slice(None), [1, 3]))  # select second and fourth columns
   z.get_orthogonal_selection(([0, 2], [1, 3]))  # select rows [0, 2] and columns [1, 4]

Data can also be modified, e.g.:

.. ipython:: python

   z.set_orthogonal_selection(([0, 2], [1, 3]), [[-1, -2], [-3, -4]])
   z[:]
For convenience, the orthogonal indexing functionality is also available via the
``oindex`` property, e.g.:

.. ipython:: python

   # TODO: replace with create_array after #2463
   z = zarr.array(np.arange(15).reshape(3, 5))
   z.oindex[[0, 2], :]  # select first and third rows
   z.oindex[:, [1, 3]]  # select second and fourth columns
   z.oindex[[0, 2], [1, 3]]  # select rows [0, 2] and columns [1, 4]
   z.oindex[[0, 2], [1, 3]] = [[-1, -2], [-3, -4]]
   z[:]

Any combination of integer, slice, 1D integer array and/or 1D Boolean array can
be used for orthogonal indexing.

If the index contains at most one iterable, and otherwise contains only slices and integers,
orthogonal indexing is also available directly on the array:

.. ipython:: python

   # TODO: replace with create_array after #2463
   z = zarr.array(np.arange(15).reshape(3, 5))
   np.all(z.oindex[[0, 2], :] == z[[0, 2], :])

Block Indexing
~~~~~~~~~~~~~~

Zarr also support block indexing, which allows selections of whole chunks based on their
logical indices along each dimension of an array. For example, this allows selecting
a subset of chunk aligned rows and/or columns from a 2-dimensional array. E.g.:

.. ipython:: python

   # TODO: replace with create_array after #2463
   z = zarr.array(np.arange(100).reshape(10, 10), chunks=(3, 3))

Retrieve items by specifying their block coordinates:

.. ipython:: python

   z.get_block_selection(1)

Equivalent slicing:

.. ipython:: python

   z[3:6]

For convenience, the block selection functionality is also available via the
`blocks` property, e.g.:

.. ipython:: python

   z.blocks[1]

Block index arrays may be multidimensional to index multidimensional arrays.
For example:

.. ipython:: python

   z.blocks[0, 1:3]

Data can also be modified. Let's start by a simple 2D array:

.. ipython:: python

   z = zarr.zeros((6, 6), dtype=int, chunks=2)

Set data for a selection of items:

.. ipython:: python

   z.set_block_selection((1, 0), 1)
   z[...]

For convenience, this functionality is also available via the ``blocks`` property.
E.g.:

.. ipython:: python

   z.blocks[:, 2] = 7
   z[...]

Any combination of integer and slice can be used for block indexing:

.. ipython:: python

   z.blocks[2, 1:3]

   # TODO: replace with create_group after #2463
   root = zarr.group('data/example-12.zarr')
   foo = root.create_array(name='foo', shape=(1000, 100), chunks=(10, 10), dtype='f4')
   bar = root.create_array(name='foo/bar', shape=(100,), dtype='i4')
   foo[:, :] = np.random.random((1000, 100))
   bar[:] = np.arange(100)
   root.tree()

.. _user-guide-sharding:

Sharding
--------

Coming soon.


Missing features in 3.0
-----------------------


The following features have not been ported to 3.0 yet.

.. _user-guide-objects:

Object arrays
~~~~~~~~~~~~~

See the Zarr-Python 2 documentation on `Object arrays <https://zarr.readthedocs.io/en/support-v2/tutorial.html#object-arrays>`_ for more details.

.. _user-guide-strings:

Fixed-length string arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~

See the Zarr-Python 2 documentation on `Fixed-length string arrays <https://zarr.readthedocs.io/en/support-v2/tutorial.html#string-arrays>`_ for more details.

.. _user-guide-datetime:

Datetime and Timedelta arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the Zarr-Python 2 documentation on `Datetime and Timedelta <https://zarr.readthedocs.io/en/support-v2/tutorial.html#datetimes-and-timedeltas>`_ for more details.

.. _user-guide-copy:

Copying and migrating data
~~~~~~~~~~~~~~~~~~~~~~~~~~

See the Zarr-Python 2 documentation on `Copying and migrating data <https://zarr.readthedocs.io/en/support-v2/tutorial.html#copying-migrating-data>`_ for more details.
