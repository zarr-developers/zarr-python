
Working With Arrays
===================

Creating an array
-----------------

Zarr has several functions for creating arrays. For example:

.. ipython:: python

   import zarr

   z = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4')
   z

The code above creates a 2-dimensional array of 32-bit integers with 10000 rows
and 10000 columns, divided into chunks where each chunk has 1000 rows and 1000
columns (and so there will be 100 chunks in total).

For a complete list of array creation routines see the :mod:`zarr.api.synchronous`
module documentation.

.. _tutorial_array:

Reading and writing data
------------------------

Zarr arrays support a similar interface to NumPy arrays for reading and writing
data. For example, the entire array can be filled with a scalar value:

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

.. _tutorial_persist:

Persistent arrays
-----------------

In the examples above, compressed data for each chunk of the array was stored in
main memory. Zarr arrays can also be stored on a file system, enabling
persistence of data between sessions. For example:

.. ipython:: python

   z1 = zarr.open('data/example.zarr', mode='w', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4')

The array above will store its configuration metadata and all compressed chunk
data in a directory called 'data/example.zarr' relative to the current working
directory. The :func:`zarr.api.synchronous.open` function provides a convenient way
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

   z2 = zarr.open('data/example.zarr', mode='r')
   np.all(z1[:] == z2[:])

If you are just looking for a fast and convenient way to save NumPy arrays to
disk then load back into memory later, the functions
:func:`zarr.convenience.save` and :func:`zarr.convenience.load` may be
useful. E.g.:

.. ipython:: python
   :suppress:

   In [144]: rm -r data/example.zarr

.. ipython:: python

   a = np.arange(10)
   zarr.save('data/example.zarr', a)
   zarr.load('data/example.zarr')

Please note that there are a number of other options for persistent array
storage, see the section on :ref:`tutorial_storage` below.

.. _tutorial_resize:

Resizing and appending
----------------------

A Zarr array can be resized, which means that any of its dimensions can be
increased or decreased in length. For example:

.. ipython:: python

   z = zarr.zeros(shape=(10000, 10000), chunks=(1000, 1000))
   z[:] = 42
   z.resize((20000, 10000))
   z.shape

Note that when an array is resized, the underlying data are not rearranged in
any way. If one or more dimensions are shrunk, any chunks falling outside the
new array shape will be deleted from the underlying store.

For convenience, Zarr arrays also provide an ``append()`` method, which can be
used to append data to any axis. E.g.:

.. ipython:: python

   a = np.arange(10000000, dtype='i4').reshape(10000, 1000)
   z = zarr.array(a, chunks=(1000, 100))
   z.shape
   z.append(a)
   z.append(np.vstack([a, a]), axis=1)
   z.shape

.. _tutorial_compress:

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
   # TODO: remove zarr_format
   z = zarr.array(data, chunks=(1000, 1000), compressor=compressor, zarr_format=2)
   None  # TODO: z.compressor

This array above will use Blosc as the primary compressor, using the Zstandard
algorithm (compression level 3) internally within Blosc, and with the
bit-shuffle filter applied.

When using a compressor, it can be useful to get some diagnostics on the
compression ratio. Zarr arrays provide a ``info`` property which can be used to
print some diagnostics, e.g.:

.. ipython:: python

   z.info

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

   # TODO: remove zarr_format
   z = zarr.array(np.arange(100000000, dtype='i4').reshape(10000, 10000), chunks=(1000, 1000), compressor=Zstd(level=1), zarr_format=2)
   None  # TODO: z.compressor

Here is an example using LZMA with a custom filter pipeline including LZMA's
built-in delta filter:

.. ipython:: python

   import lzma
   from numcodecs import LZMA

   lzma_filters = [dict(id=lzma.FILTER_DELTA, dist=4), dict(id=lzma.FILTER_LZMA2, preset=1)]
   compressor = LZMA(filters=lzma_filters)
   # TODO: remove zarr_format
   z = zarr.array(np.arange(100000000, dtype='i4').reshape(10000, 10000), chunks=(1000, 1000), compressor=compressor, zarr_format=2)
   None  # TODO: z.compressor

The default compressor can be changed by setting the value of the
``zarr.storage.default_compressor`` variable, e.g.:

.. ipython:: python

   import zarr.storage

   None # TODO: set default compressor via config

.. TODO
..     >>> from numcodecs import Zstd, Blosc
..     >>> # switch to using Zstandard
..     ... zarr.storage.default_compressor = Zstd(level=1)
..     >>> z = zarr.zeros(100000000, chunks=1000000)
..     >>> z.compressor
..     Zstd(level=1)
..     >>> # switch back to Blosc defaults
..     ... zarr.storage.default_compressor = Blosc()

To disable compression, set ``compressor=None`` when creating an array, e.g.:

.. ipython:: python

   # TODO: remove zarr_format
   z = zarr.zeros(100000000, chunks=1000000, compressor=None, zarr_format=2)
   None  # TODO: z.compressor is None

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

Here is an example using a delta filter with the Blosc compressor:

.. ipython:: python

   from numcodecs import Blosc, Delta

   filters = [Delta(dtype='i4')]
   compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
   data = np.arange(100000000, dtype='i4').reshape(10000, 10000)
   # TODO: remove zarr_format
   z = zarr.array(data, chunks=(1000, 1000), filters=filters, compressor=compressor, zarr_format=2)
   z.info

For more information about available filter codecs, see the `Numcodecs
<https://numcodecs.readthedocs.io/>`_ documentation.

.. _tutorial_indexing:

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

.. Items can also be extracted by providing a Boolean mask. E.g.:

.. ipython:: python

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

   z = zarr.array(np.arange(15).reshape(3, 5))
   np.all(z.oindex[[0, 2], :] == z[[0, 2], :])

Block Indexing
~~~~~~~~~~~~~~

Zarr also support block indexing, which allows selections of whole chunks based on their
logical indices along each dimension of an array. For example, this allows selecting
a subset of chunk aligned rows and/or columns from a 2-dimensional array. E.g.:

.. ipython:: python

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

Indexing fields in structured arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All selection methods support a ``fields`` parameter which allows retrieving or
replacing data for a specific field in an array with a structured dtype. E.g.:

.. ipython:: python

   a = np.array([(b'aaa', 1, 4.2), (b'bbb', 2, 8.4), (b'ccc', 3, 12.6)], dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
   None  # TODO: z = zarr.array(a)
   None  # TODO: z['foo']
   None  # TODO: z['baz']
   None  # TODO: z.get_basic_selection(slice(0, 2), fields='bar')
   None  # TODO: z.get_coordinate_selection([0, 2], fields=['foo', 'baz'])

.. .. _tutorial_strings:

.. String arrays
.. -------------

.. There are several options for storing arrays of strings.

.. If your strings are all ASCII strings, and you know the maximum length of the string in
.. your array, then you can use an array with a fixed-length bytes dtype. E.g.:

..     >>> z = zarr.zeros(10, dtype='S6')
..     >>> z
..     <zarr.Array (10,) |S6>
..     >>> z[0] = b'Hello'
..     >>> z[1] = b'world!'
..     >>> z[:]
..     array([b'Hello', b'world!', b'', b'', b'', b'', b'', b'', b'', b''],
..           dtype='|S6')

.. A fixed-length unicode dtype is also available, e.g.:

..     >>> greetings = ['¡Hola mundo!', 'Hej Världen!', 'Servus Woid!', 'Hei maailma!',
..     ...              'Xin chào thế giới', 'Njatjeta Botë!', 'Γεια σου κόσμε!',
..     ...              'こんにちは世界', '世界，你好！', 'Helló, világ!', 'Zdravo svete!',
..     ...              'เฮลโลเวิลด์']
..     >>> text_data = greetings * 10000
..     >>> z = zarr.array(text_data, dtype='U20')
..     >>> z
..     <zarr.Array (120000,) <U20>
..     >>> z[:]
..     array(['¡Hola mundo!', 'Hej Världen!', 'Servus Woid!', ...,
..            'Helló, világ!', 'Zdravo svete!', 'เฮลโลเวิลด์'],
..           dtype='<U20')

.. For variable-length strings, the ``object`` dtype can be used, but a codec must be
.. provided to encode the data (see also :ref:`tutorial_objects` below). At the time of
.. writing there are four codecs available that can encode variable length string
.. objects: :class:`numcodecs.vlen.VLenUTF8`, :class:`numcodecs.json.JSON`,
.. :class:`numcodecs.msgpacks.MsgPack`. and :class:`numcodecs.pickles.Pickle`.
.. E.g. using ``VLenUTF8``:

..     >>> import numcodecs
..     >>> z = zarr.array(text_data, dtype=object, object_codec=numcodecs.VLenUTF8())
..     >>> z
..     <zarr.Array (120000,) object>
..     >>> z.filters
..     [VLenUTF8()]
..     >>> z[:]
..     array(['¡Hola mundo!', 'Hej Världen!', 'Servus Woid!', ...,
..            'Helló, világ!', 'Zdravo svete!', 'เฮลโลเวิลด์'], dtype=object)

.. As a convenience, ``dtype=str`` (or ``dtype=unicode`` on Python 2.7) can be used, which
.. is a short-hand for ``dtype=object, object_codec=numcodecs.VLenUTF8()``, e.g.:

..     >>> z = zarr.array(text_data, dtype=str)
..     >>> z
..     <zarr.Array (120000,) object>
..     >>> z.filters
..     [VLenUTF8()]
..     >>> z[:]
..     array(['¡Hola mundo!', 'Hej Världen!', 'Servus Woid!', ...,
..            'Helló, világ!', 'Zdravo svete!', 'เฮลโลเวิลด์'], dtype=object)

.. Variable-length byte strings are also supported via ``dtype=object``. Again an
.. ``object_codec`` is required, which can be one of :class:`numcodecs.vlen.VLenBytes` or
.. :class:`numcodecs.pickles.Pickle`. For convenience, ``dtype=bytes`` (or ``dtype=str`` on Python
.. 2.7) can be used as a short-hand for ``dtype=object, object_codec=numcodecs.VLenBytes()``,
.. e.g.:

..     >>> bytes_data = [g.encode('utf-8') for g in greetings] * 10000
..     >>> z = zarr.array(bytes_data, dtype=bytes)
..     >>> z
..     <zarr.Array (120000,) object>
..     >>> z.filters
..     [VLenBytes()]
..     >>> z[:]
..     array([b'\xc2\xa1Hola mundo!', b'Hej V\xc3\xa4rlden!', b'Servus Woid!',
..            ..., b'Hell\xc3\xb3, vil\xc3\xa1g!', b'Zdravo svete!',
..            b'\xe0\xb9\x80\xe0\xb8\xae\xe0\xb8\xa5\xe0\xb9\x82\xe0\xb8\xa5\xe0\xb9\x80\xe0\xb8\xa7\xe0\xb8\xb4\xe0\xb8\xa5\xe0\xb8\x94\xe0\xb9\x8c'], dtype=object)

.. If you know ahead of time all the possible string values that can occur, you could
.. also use the :class:`numcodecs.categorize.Categorize` codec to encode each unique string value as an
.. integer. E.g.:

.. .. ipython::

..    In [1]: categorize = numcodecs.Categorize(greetings, dtype=object)

..    In [1]: z = zarr.array(text_data, dtype=object, object_codec=categorize)

..    In [1]: z

..    In [1]: z.filters

..    In [1]: z[:]

.. .. _tutorial_objects:

.. Object arrays
.. -------------

.. Zarr supports arrays with an "object" dtype. This allows arrays to contain any type of
.. object, such as variable length unicode strings, or variable length arrays of numbers, or
.. other possibilities. When creating an object array, a codec must be provided via the
.. ``object_codec`` argument. This codec handles encoding (serialization) of Python objects.
.. The best codec to use will depend on what type of objects are present in the array.

.. At the time of writing there are three codecs available that can serve as a general
.. purpose object codec and support encoding of a mixture of object types:
.. :class:`numcodecs.json.JSON`, :class:`numcodecs.msgpacks.MsgPack`. and :class:`numcodecs.pickles.Pickle`.

.. For example, using the JSON codec:

.. .. ipython::

..    In [1]: z = zarr.empty(5, dtype=object, object_codec=numcodecs.JSON())

..    In [1]: z[0] = 42

..    In [1]: z[1] = 'foo'

..    In [1]: z[2] = ['bar', 'baz', 'qux']

..    In [1]: z[3] = {'a': 1, 'b': 2.2}

..    In [1]: z[:]

.. Not all codecs support encoding of all object types. The
.. :class:`numcodecs.pickles.Pickle` codec is the most flexible, supporting encoding any type
.. of Python object. However, if you are sharing data with anyone other than yourself, then
.. Pickle is not recommended as it is a potential security risk. This is because malicious
.. code can be embedded within pickled data. The JSON and MsgPack codecs do not have any
.. security issues and support encoding of unicode strings, lists and dictionaries.
.. MsgPack is usually faster for both encoding and decoding.

.. Ragged arrays
.. ~~~~~~~~~~~~~

.. If you need to store an array of arrays, where each member array can be of any length
.. and stores the same primitive type (a.k.a. a ragged array), the
.. :class:`numcodecs.vlen.VLenArray` codec can be used, e.g.:

.. .. ipython::

..    In [1]: z = zarr.empty(4, dtype=object, object_codec=numcodecs.VLenArray(int))

..    In [1]: z

..    In [1]: z.filters

..    In [1]: z[0] = np.array([1, 3, 5])

..    In [1]: z[1] = np.array([4])

..    In [1]: z[2] = np.array([7, 9, 14])

..    In [1]: z[:]

.. As a convenience, ``dtype='array:T'`` can be used as a short-hand for
.. ``dtype=object, object_codec=numcodecs.VLenArray('T')``, where 'T' can be any NumPy
.. primitive dtype such as 'i4' or 'f8'. E.g.:

.. .. ipython::

..    In [1]: z = zarr.empty(4, dtype='array:i8')

..    In [1]: z

..    In [1]: z.filters

..    In [1]: z[0] = np.array([1, 3, 5])

..    In [1]: z[1] = np.array([4])

..    In [1]: z[2] = np.array([7, 9, 14])

..    In [1]: z[:]

.. .. _tutorial_chunks:

.. Chunk optimizations
.. -------------------

.. .. _tutorial_chunks_shape:

.. Chunk size and shape
.. ~~~~~~~~~~~~~~~~~~~~

.. In general, chunks of at least 1 megabyte (1M) uncompressed size seem to provide
.. better performance, at least when using the Blosc compression library.

.. The optimal chunk shape will depend on how you want to access the data. E.g.,
.. for a 2-dimensional array, if you only ever take slices along the first
.. dimension, then chunk across the second dimension. If you know you want to chunk
.. across an entire dimension you can use ``None`` or ``-1`` within the ``chunks``
.. argument, e.g.:

.. .. ipython::

..    In [1]: z1 = zarr.zeros((10000, 10000), chunks=(100, None), dtype='i4')
..     >>> z1.chunks
..     (100, 10000)

.. Alternatively, if you only ever take slices along the second dimension, then
.. chunk across the first dimension, e.g.:

.. .. ipython::

..    In [1]: z2 = zarr.zeros((10000, 10000), chunks=(None, 100), dtype='i4')

..    In [1]: z2.chunks
..     (10000, 100)

.. If you require reasonable performance for both access patterns then you need to
.. find a compromise, e.g.:

.. .. ipython::

..    In [1]: z3 = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4')

..    In [1]: z3.chunks
..     (1000, 1000)

.. If you are feeling lazy, you can let Zarr guess a chunk shape for your data by
.. providing ``chunks=True``, although please note that the algorithm for guessing
.. a chunk shape is based on simple heuristics and may be far from optimal. E.g.:

.. .. ipython::

..    In [1]: z4 = zarr.zeros((10000, 10000), chunks=True, dtype='i4')

..    In [1]: z4.chunks
..     (625, 625)

.. If you know you are always going to be loading the entire array into memory, you
.. can turn off chunks by providing ``chunks=False``, in which case there will be
.. .. one single chunk for the array::

.. .. ipython::

..    In [1]: z5 = zarr.zeros((10000, 10000), chunks=False, dtype='i4')

..    In [1]: z5.chunks
..     (10000, 10000)

.. .. _tutorial_chunks_order:

.. Chunk memory layout
.. ~~~~~~~~~~~~~~~~~~~

.. The order of bytes **within each chunk** of an array can be changed via the
.. ``order`` keyword argument, to use either C or Fortran layout. For
.. multi-dimensional arrays, these two layouts may provide different compression
.. ratios, depending on the correlation structure within the data. E.g.:

.. .. ipython::

..    In [1]: a = np.arange(100000000, dtype='i4').reshape(10000, 10000).T

..    In [1]: c = zarr.array(a, chunks=(1000, 1000))

..    In [1]: c.info

..    In [1]: f = zarr.array(a, chunks=(1000, 1000), order='F')

..    In [1]: f.info

.. In the above example, Fortran order gives a better compression ratio. This is an
.. artificial example but illustrates the general point that changing the order of
.. bytes within chunks of an array may improve the compression ratio, depending on
.. the structure of the data, the compression algorithm used, and which compression
.. filters (e.g., byte-shuffle) have been applied.

.. .. _tutorial_chunks_empty_chunks:

.. Empty chunks
.. ~~~~~~~~~~~~

.. As of version 2.11, it is possible to configure how Zarr handles the storage of
.. chunks that are "empty" (i.e., every element in the chunk is equal to the array's fill value).
.. When creating an array with ``write_empty_chunks=False``,
.. Zarr will check whether a chunk is empty before compression and storage. If a chunk is empty,
.. then Zarr does not store it, and instead deletes the chunk from storage
.. if the chunk had been previously stored.

.. This optimization prevents storing redundant objects and can speed up reads, but the cost is
.. added computation during array writes, since the contents of
.. each chunk must be compared to the fill value, and these advantages are contingent on the content of the array.
.. If you know that your data will form chunks that are almost always non-empty, then there is no advantage to the optimization described above.
.. In this case, creating an array with ``write_empty_chunks=True`` (the default) will instruct Zarr to write every chunk without checking for emptiness.

.. The following example illustrates the effect of the ``write_empty_chunks`` flag on
.. the time required to write an array with different values.:

.. .. ipython::

..    In [1]: import zarr

..    In [1]: import numpy as np

..    In [1]: import time

..    In [1]: from tempfile import TemporaryDirectory

..    In [1]: def timed_write(write_empty_chunks):
..                """
..                Measure the time required and number of objects created when writing
..                to a Zarr array with random ints or fill value.
..                """
..                chunks = (8192,)
..                shape = (chunks[0] * 1024,)
..                data = np.random.randint(0, 255, shape)
..                dtype = 'uint8'

..                with TemporaryDirectory() as store:
..                    arr = zarr.open(store,
..                                    shape=shape,
..                                    chunks=chunks,
..                                    dtype=dtype,
..                                    write_empty_chunks=write_empty_chunks,
..                                    fill_value=0,
..                                    mode='w')
..                    # initialize all chunks
..                    arr[:] = 100
..                    result = []
..                    for value in (data, arr.fill_value):
..                        start = time.time()
..                        arr[:] = value
..                        elapsed = time.time() - start
..                        result.append((elapsed, arr.nchunks_initialized))

..                    return result

..    In [1]: for write_empty_chunks in (True, False):
..                full, empty = timed_write(write_empty_chunks)
..                print(f'\nwrite_empty_chunks={write_empty_chunks}:\n\tRandom Data: {full[0]:.4f}s, {full[1]} objects stored\n\t Empty Data: {empty[0]:.4f}s, {empty[1]} objects stored\n')

.. In this example, writing random data is slightly slower with ``write_empty_chunks=True``,
.. but writing empty data is substantially faster and generates far fewer objects in storage.

.. .. _tutorial_rechunking:

.. Changing chunk shapes (rechunking)
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. Sometimes you are not free to choose the initial chunking of your input data, or
.. you might have data saved with chunking which is not optimal for the analysis you
.. have planned. In such cases it can be advantageous to re-chunk the data. For small
.. datasets, or when the mismatch between input and output chunks is small
.. such that only a few chunks of the input dataset need to be read to create each
.. chunk in the output array, it is sufficient to simply copy the data to a new array
.. with the desired chunking, e.g.:

.. .. ipython::

..    In [1]: a = zarr.zeros((10000, 10000), chunks=(100,100), dtype='uint16', store='a.zarr')

..    In [1]: b = zarr.array(a, chunks=(100, 200), store='b.zarr')

.. If the chunk shapes mismatch, however, a simple copy can lead to non-optimal data
.. access patterns and incur a substantial performance hit when using
.. file based stores. One of the most pathological examples is
.. switching from column-based chunking to row-based chunking e.g.:

.. .. ipython::

..    In [1]: a = zarr.zeros((10000,10000), chunks=(10000, 1), dtype='uint16', store='a.zarr')

..    In [1]: b = zarr.array(a, chunks=(1,10000), store='b.zarr')

.. which will require every chunk in the input data set to be repeatedly read when creating
.. each output chunk. If the entire array will fit within memory, this is simply resolved
.. by forcing the entire input array into memory as a numpy array before converting
.. back to zarr with the desired chunking.

.. .. ipython::

..    In [1]: a = zarr.zeros((10000,10000), chunks=(10000, 1), dtype='uint16', store='a.zarr')

..    In [1]: b = a[...]

..    In [1]: c = zarr.array(b, chunks=(1,10000), store='c.zarr')

.. For data sets which have mismatched chunks and which do not fit in memory, a
.. more sophisticated approach to rechunking, such as offered by the
.. `rechunker <https://github.com/pangeo-data/rechunker>`_ package and discussed
.. `here <https://medium.com/pangeo/rechunker-the-missing-link-for-chunked-array-analytics-5b2359e9dc11>`_
.. may offer a substantial improvement in performance.

.. .. _tutorial_sync:

.. Parallel computing and synchronization
.. --------------------------------------

.. Zarr arrays have been designed for use as the source or sink for data in
.. parallel computations. By data source we mean that multiple concurrent read
.. operations may occur. By data sink we mean that multiple concurrent write
.. operations may occur, with each writer updating a different region of the
.. array. Zarr arrays have **not** been designed for situations where multiple
.. readers and writers are concurrently operating on the same array.

.. Both multi-threaded and multi-process parallelism are possible. The bottleneck
.. for most storage and retrieval operations is compression/decompression, and the
.. Python global interpreter lock (GIL) is released wherever possible during these
.. operations, so Zarr will generally not block other Python threads from running.

.. When using a Zarr array as a data sink, some synchronization (locking) may be
.. required to avoid data loss, depending on how data are being updated. If each
.. worker in a parallel computation is writing to a separate region of the array,
.. and if region boundaries are perfectly aligned with chunk boundaries, then no
.. synchronization is required. However, if region and chunk boundaries are not
.. perfectly aligned, then synchronization is required to avoid two workers
.. attempting to modify the same chunk at the same time, which could result in data
.. loss.

.. To give a simple example, consider a 1-dimensional array of length 60, ``z``,
.. divided into three chunks of 20 elements each. If three workers are running and
.. each attempts to write to a 20 element region (i.e., ``z[0:20]``, ``z[20:40]``
.. and ``z[40:60]``) then each worker will be writing to a separate chunk and no
.. synchronization is required. However, if two workers are running and each
.. attempts to write to a 30 element region (i.e., ``z[0:30]`` and ``z[30:60]``)
.. then it is possible both workers will attempt to modify the middle chunk at the
.. same time, and synchronization is required to prevent data loss.

.. Zarr provides support for chunk-level synchronization. E.g., create an array
.. with thread synchronization:

.. .. ipython::

..    In [1]: z = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4', synchronizer=zarr.ThreadSynchronizer())

..    In [1]: z

.. This array is safe to read or write within a multi-threaded program.

.. Zarr also provides support for process synchronization via file locking,
.. provided that all processes have access to a shared file system, and provided
.. that the underlying file system supports file locking (which is not the case for
.. some networked file systems). E.g.:

.. .. ipython::

..    In [1]: synchronizer = zarr.ProcessSynchronizer('data/example.sync')

..    In [1]: z = zarr.open_array('data/example', mode='w', shape=(10000, 10000),
..     ...                     chunks=(1000, 1000), dtype='i4',
..     ...                     synchronizer=synchronizer)

..    In [1]: z
..     <zarr.Array (10000, 10000) int32>

.. This array is safe to read or write from multiple processes.

.. When using multiple processes to parallelize reads or writes on arrays using the Blosc
.. compression library, it may be necessary to set ``numcodecs.blosc.use_threads = False``,
.. as otherwise Blosc may share incorrect global state amongst processes causing programs
.. to hang. See also the section on :ref:`tutorial_tips_blosc` below.

.. Please note that support for parallel computing is an area of ongoing research
.. and development. If you are using Zarr for parallel computing, we welcome
.. feedback, experience, discussion, ideas and advice, particularly about issues
.. related to data integrity and performance.

.. .. _tutorial_pickle:

.. Pickle support
.. --------------

.. Zarr arrays and groups can be pickled, as long as the underlying store object can be
.. pickled. Instances of any of the storage classes provided in the :mod:`zarr.storage`
.. module can be pickled, as can the built-in ``dict`` class which can also be used for
.. storage.

.. Note that if an array or group is backed by an in-memory store like a ``dict`` or
.. :class:`zarr.storage.MemoryStore`, then when it is pickled all of the store data will be
.. included in the pickled data. However, if an array or group is backed by a persistent
.. store like a :class:`zarr.storage.DirectoryStore`, :class:`zarr.storage.ZipStore` or
.. :class:`zarr.storage.DBMStore` then the store data **are not** pickled. The only thing
.. that is pickled is the necessary parameters to allow the store to re-open any
.. underlying files or databases upon being unpickled.

.. E.g., pickle/unpickle an in-memory array:

.. .. ipython::

..    In [1]: import pickle

..    In [1]: z1 = zarr.array(np.arange(100000))

..    In [1]: s = pickle.dumps(z1)

..    In [1]: len(s) > 5000  # relatively large because data have been pickled

..    In [1]: z2 = pickle.loads(s)

..    In [1]: z1 == z2

..    In [1]: np.all(z1[:] == z2[:])

.. E.g., pickle/unpickle an array stored on disk:

.. .. ipython::

..    In [1]: z3 = zarr.open('data/walnuts.zarr', mode='w', shape=100000, dtype='i8')

..    In [1]: z3[:] = np.arange(100000)

..    In [1]: s = pickle.dumps(z3)

..    In [1]: len(s) < 200  # small because no data have been pickled

..    In [1]: z4 = pickle.loads(s)

..    In [1]: z3 == z4

..    In [1]: np.all(z3[:] == z4[:])

.. .. _tutorial_datetime:

.. Datetimes and timedeltas
.. ------------------------

.. NumPy's ``datetime64`` ('M8') and ``timedelta64`` ('m8') dtypes are supported for Zarr
.. arrays, as long as the units are specified. E.g.:

.. .. ipython::

..    In [1]: z = zarr.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='M8[D]')

..    In [1]: z

..    In [1]: z[:]

..    In [1]: z[0]

..    In [1]: z[0] = '1999-12-31'

..    In [1]: z[:]

.. .. _tutorial_tips:

.. Usage tips
.. ----------

.. .. _tutorial_tips_copy:

.. Copying large arrays
.. ~~~~~~~~~~~~~~~~~~~~

.. Data can be copied between large arrays without needing much memory, e.g.:

.. .. ipython::

..    In [1]: z1 = zarr.empty((10000, 10000), chunks=(1000, 1000), dtype='i4')

..    In [1]: z1[:] = 42

..    In [1]: z2 = zarr.empty_like(z1)

..    In [1]: z2[:] = z1

.. Internally the example above works chunk-by-chunk, extracting only the data from
.. ``z1`` required to fill each chunk in ``z2``. The source of the data (``z1``)
.. could equally be an h5py Dataset.

.. .. _tutorial_tips_blosc:

.. Configuring Blosc
.. ~~~~~~~~~~~~~~~~~

.. The Blosc compressor is able to use multiple threads internally to accelerate
.. compression and decompression. By default, Blosc uses up to 8
.. internal threads. The number of Blosc threads can be changed to increase or
.. decrease this number, e.g.:

.. .. ipython::

..    In [1]: from numcodecs import blosc

..    In [1]: blosc.set_nthreads(2)  # doctest: +SKIP

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
