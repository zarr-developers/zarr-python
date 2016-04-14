zarr - Persistence
==================

This document describes the file organisation and formats used to save zarr
arrays on disk.

All data and metadata associated with a zarr array is stored within a
directory on the file system. Within this directory there are a number
of files and sub-directories storing different components of the data
and metadata. Here I'll refer to a directory containing a zarr array
as a root directory.

Configuration metadata
----------------------

Within a root directory, a file called "__zmeta__" contains essential
configuration metadata about the array. This comprises the shape of the
array, chunk shape, data type (dtype), compression library,
compression level, shuffle filter and default fill value for
uninitialised portions of the array. The format of this file is JSON.

Mandatory fields and allowed values are as follows:

* ``shape`` - list of integers - the size of each dimension of the array
* ``chunks`` - list of integers - the size of each dimension of a chunk, i.e., the chunk shape
* ``dtype`` - string or list of lists - a description of the data type, following Numpy convention
* ``fill_value`` - scalar value - value to use for uninitialised portions of the array
* ``cname`` - string - name of the compression library used
* ``clevel`` - integer - compression level
* ``shuffle`` - integer - shuffle filter (0 = no shuffle, 1 = byte shuffle, 2 = bit shuffle)

For example::

    >>> import zarr
    >>> z = zarr.open('example.zarr', mode='w', shape=(1000000, 1000),
    ...               chunks=(10000, 100), dtype='i4', fill_value=42,
    ...               cname='lz4', clevel=3, shuffle=1)
    >>> print(open('example.zarr/__zmeta__').read())
    {
        "chunks": [
            10000,
            100
        ],
        "clevel": 3,
        "cname": "lz4",
        "dtype": "<i4",
        "fill_value": 42,
        "shape": [
            1000000,
            1000
        ],
        "shuffle": 1
    }

User metadata (attributes)
--------------------------

Within a root directory, a file called "__zattr__" contains user
metadata associated with the array, i.e., user attributes. The format
of this file is JSON.

For example::
  
    >>> import zarr
    >>> z = zarr.open('example.zarr', mode='w', shape=(1000000, 1000),
    ...               chunks=(10000, 100), dtype='i4', fill_value=42,
    ...               cname='lz4', clevel=3, shuffle=1)
    >>> z.attrs['foo'] = 42
    >>> z.attrs['bar'] = 4.2
    >>> z.attrs['baz'] = 'quux'
    >>> print(open('example.zarr/__zattr__').read())

TODO add results above

Array data
----------

Within a root directory, a sub-directory called "__zdata__" contains
the array data. The array data is divided into chunks, each of which
is compressed using the [blosc meta-compression library](TODO). Each
chunk is stored in a separate file.

The chunk files are named according to the chunk indices. E.g., for a
2-dimensional array with shape (100, 100) and chunk shape (10, 10)
there will be 100 chunks in total. The file "0.0.blosc" stores data
for the chunk with indices (0, 0) within chunk rows and columns
respectively, i.e., the first chunk, containing data for the segment
of the array that would be obtained by the slice ``z[0:10, 0:10]``;
the file "4.2.blosc" stores the chunk in the fifth row third column,
containing data for the slize ``z[40:50, 20:30]``; etc.

Each chunk file is a binary file following the blosc version 1 format,
comprising a 16 byte header followed by the compressed data. The
header is organised as follows::

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
      ^   ^   ^   ^ |     nbytes    |   blocksize   |    cbytes     |
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------blosclz version
      +--------------blosc version

For more details on the header, see the [C-Blosc header
description](https://github.com/Blosc/c-blosc/blob/master/README_HEADER.rst).

If a file does not exist on the file system for any given chunk in an
array, that indicates the chunk has not been initialised, and the
chunk should be interpreted as completely filled with whatever value
has been configured as the fill value for the array. I.e., chunk files
are not required to exist.

For example::

    >>> import zarr
    >>> z = zarr.open('example.zarr', mode='w', shape=(1000000, 1000),
    ...               chunks=(10000, 100), dtype='i4', fill_value=42,
    ...               cname='lz4', clevel=3, shuffle=1)
    >>> import os
    >>> os.listdir('example.zarr/__zdata__')
    []
    >>> z[:] = 0
    >>> sorted(os.listdir('example.zarr/__zdata__'))[:5]
    ['0.0.blosc', '0.1.blosc', '0.2.blosc', '0.3.blosc', '0.4.blosc']
