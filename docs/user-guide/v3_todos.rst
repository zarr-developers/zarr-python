3.0 TO DOs
==========

.. warning::
   As noted in the `3.0 Migration Guide <user-guide/v3_migration>`_, there are still a few
   features that were present in Zarr-Python 2 that are not yet ported to Zarr-Python 3.
   This section summarizes the remaining features that are not yet ported to Zarr-Python 3
   but is not meant to be used as documentation as existing features.

Indexing fields in structured arrays
------------------------------------

All selection methods support a ``fields`` parameter which allows retrieving or
replacing data for a specific field in an array with a structured dtype. E.g.:

.. ipython:: python
   :verbatim:

   a = np.array([(b'aaa', 1, 4.2), (b'bbb', 2, 8.4), (b'ccc', 3, 12.6)], dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
   z = zarr.array(a)
   z['foo']
   z['baz']
   z.get_basic_selection(slice(0, 2), fields='bar')
   z.get_coordinate_selection([0, 2], fields=['foo', 'baz'])

.. _tutorial_strings:

String arrays
-------------

There are several options for storing arrays of strings.

If your strings are all ASCII strings, and you know the maximum length of the string in
your array, then you can use an array with a fixed-length bytes dtype. E.g.:

.. ipython:: python
   :verbatim:

   z = zarr.zeros(10, dtype='S6')
   z
   z[0] = b'Hello'
   z[1] = b'world!'
   z[:]

A fixed-length unicode dtype is also available, e.g.:

.. ipython:: python
   :verbatim:

   greetings = ['¡Hola mundo!', 'Hej Världen!', 'Servus Woid!', 'Hei maailma!',
                'Xin chào thế giới', 'Njatjeta Botë!', 'Γεια σου κόσμε!',
                'こんにちは世界', '世界，你好！', 'Helló, világ!', 'Zdravo svete!',
                'เฮลโลเวิลด์']
   text_data = greetings * 10000
   z = zarr.array(text_data, dtype='U20')
   z
   z[:]

For variable-length strings, the ``object`` dtype can be used, but a codec must be
provided to encode the data (see also :ref:`tutorial_objects` below). At the time of
writing there are four codecs available that can encode variable length string
objects: :class:`numcodecs.vlen.VLenUTF8`, :class:`numcodecs.json.JSON`,
:class:`numcodecs.msgpacks.MsgPack`. and :class:`numcodecs.pickles.Pickle`.
E.g. using ``VLenUTF8``:

.. ipython:: python
   :verbatim:

   import numcodecs
   z = zarr.array(text_data, dtype=object, object_codec=numcodecs.VLenUTF8())
   z
   z.filters
   z[:]

As a convenience, ``dtype=str`` (or ``dtype=unicode`` on Python 2.7) can be used, which
is a short-hand for ``dtype=object, object_codec=numcodecs.VLenUTF8()``, e.g.:

.. ipython:: python
   :verbatim:

   z = zarr.array(text_data, dtype=str)
   z
   z.filters
   z[:]

Variable-length byte strings are also supported via ``dtype=object``. Again an
``object_codec`` is required, which can be one of :class:`numcodecs.vlen.VLenBytes` or
:class:`numcodecs.pickles.Pickle`. For convenience, ``dtype=bytes`` (or ``dtype=str`` on Python
2.7) can be used as a short-hand for ``dtype=object, object_codec=numcodecs.VLenBytes()``,
e.g.:

.. ipython:: python
   :verbatim:

   bytes_data = [g.encode('utf-8') for g in greetings] * 10000
   z = zarr.array(bytes_data, dtype=bytes)
   z
   z.filters
   z[:]

If you know ahead of time all the possible string values that can occur, you could
also use the :class:`numcodecs.categorize.Categorize` codec to encode each unique string value as an
integer. E.g.:

.. ipython:: python
   :verbatim:

   categorize = numcodecs.Categorize(greetings, dtype=object)
   z = zarr.array(text_data, dtype=object, object_codec=categorize)
   z
   z.filters
   z[:]

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
:class:`numcodecs.json.JSON`, :class:`numcodecs.msgpacks.MsgPack`. and :class:`numcodecs.pickles.Pickle`.

For example, using the JSON codec:

.. ipython:: python
   :verbatim:

   z = zarr.empty(5, dtype=object, object_codec=numcodecs.JSON())
   z[0] = 42
   z[1] = 'foo'
   z[2] = ['bar', 'baz', 'qux']
   z[3] = {'a': 1, 'b': 2.2}
   z[:]

Not all codecs support encoding of all object types. The
:class:`numcodecs.pickles.Pickle` codec is the most flexible, supporting encoding any type
of Python object. However, if you are sharing data with anyone other than yourself, then
Pickle is not recommended as it is a potential security risk. This is because malicious
code can be embedded within pickled data. The JSON and MsgPack codecs do not have any
security issues and support encoding of unicode strings, lists and dictionaries.
MsgPack is usually faster for both encoding and decoding.

Ragged arrays
~~~~~~~~~~~~~

If you need to store an array of arrays, where each member array can be of any length
and stores the same primitive type (a.k.a. a ragged array), the
:class:`numcodecs.vlen.VLenArray` codec can be used, e.g.:

.. ipython:: python
   :verbatim:

   z = zarr.empty(4, dtype=object, object_codec=numcodecs.VLenArray(int))
   z
   z.filters
   z[0] = np.array([1, 3, 5])
   z[1] = np.array([4])
   z[2] = np.array([7, 9, 14])
   z[:]

As a convenience, ``dtype='array:T'`` can be used as a short-hand for
``dtype=object, object_codec=numcodecs.VLenArray('T')``, where 'T' can be any NumPy
primitive dtype such as 'i4' or 'f8'. E.g.:

.. ipython:: python
   :verbatim:

   z = zarr.empty(4, dtype='array:i8')
   z
   z.filters
   z[0] = np.array([1, 3, 5])
   z[1] = np.array([4])
   z[2] = np.array([7, 9, 14])
   z[:]

.. _tutorial_datetime:

Datetimes and timedeltas
------------------------

NumPy's ``datetime64`` ('M8') and ``timedelta64`` ('m8') dtypes are supported for Zarr
arrays, as long as the units are specified. E.g.:

.. ipython:: python
   :verbatim:

   z = zarr.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='M8[D]')
   z
   z[:]
   z[0]
   z[0] = '1999-12-31'
   z[:]

.. _tutorial_tips:

Usage tips
----------

.. _tutorial_tips_copy:

Copying large arrays
~~~~~~~~~~~~~~~~~~~~

Data can be copied between large arrays without needing much memory, e.g.:

.. ipython:: python
   :verbatim:

   z1 = zarr.empty((10000, 10000), chunks=(1000, 1000), dtype='i4')
   z1[:] = 42
   z2 = zarr.empty_like(z1)
   z2[:] = z1

Internally the example above works chunk-by-chunk, extracting only the data from
``z1`` required to fill each chunk in ``z2``. The source of the data (``z1``)
could equally be an h5py Dataset.
