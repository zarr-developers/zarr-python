zarr - Persistence
==================

TODO generalise this to any storage system supporting key/value access.

This document describes the file organisation and formats used to save zarr
arrays on disk.

All data and metadata associated with a zarr array are stored within a
directory on the file system. Within this directory there are a number
of files and sub-directories storing different components of the data
and metadata. Here I'll refer to a directory containing a zarr array
as a root directory.

Configuration metadata
----------------------

Within a root directory, a file called "meta" contains essential
configuration metadata about the array. The format of this file is
JSON. Mandatory fields and allowed values are as follows:

* ``shape`` - list of integers - the size of each dimension of the array
* ``chunks`` - list of integers - the size of each dimension of a chunk, i.e., the chunk shape
* ``dtype`` - string - a description of the data type, following Numpy convention
* ``fill_value`` - scalar value - value to use for uninitialised portions of the array
* ``compression`` - string - name of the primary compression library used
* ``compression_opts`` - compression options
* ``zarr_format`` - integer - specified the array format version

For example::

    >>> import zarr
    >>> z = zarr.open('example.zarr', mode='w', shape=(1000000, 1000),
    ...               chunks=(10000, 100), dtype='i4', fill_value=42,
    ...               compression='blosc', compression_opts=dict(cname='lz4',
    ...               clevel=3, shuffle=1))
    >>> print(open('example.zarr/meta').read())
    {
        "chunks": [
            10000,
            100
        ],
        "compression": "blosc",
        "compression_opts": {
            "clevel": 3,
            "cname": "lz4",
            "shuffle": 1
        },
        "dtype": "<i4",
        "fill_value": 42,
        "order": "C",
        "shape": [
            1000000,
            1000
        ],
        "zarr_format": 1
    }

User metadata (attributes)
--------------------------

Within a root directory, a file called "attrs" contains user
metadata associated with the array, i.e., user attributes. The format
of this file is JSON.

For example::

    >>> import zarr
    >>> z = zarr.open('example.zarr', mode='w', shape=(1000000, 1000),
    ...               chunks=(10000, 100), dtype='i4', fill_value=42,
    ...               compression='blosc', compression_opts=dict(cname='lz4',
    ...               clevel=3, shuffle=1))
    >>> z.attrs['foo'] = 42
    >>> z.attrs['bar'] = 4.2
    >>> z.attrs['baz'] = 'quux'
    >>> print(open('example.zarr/attrs').read())
    {
        "bar": 4.2,
        "baz": "quux",
        "foo": 42
    }

Array data
----------

The array data is divided into chunks, each of which
is compressed using the `blosc meta-compression library
<https://github.com/blosc/c-blosc>`_. Each chunk is stored in a
separate file within the root directory.

The chunk files are named according to the chunk indices. E.g., for a
2-dimensional array with shape (100, 100) and chunk shape (10, 10)
there will be 100 chunks in total. The file "0.0" stores data
for the chunk with indices (0, 0) within chunk rows and columns
respectively, i.e., the first chunk, containing data for the segment
of the array that would be obtained by the slice ``z[0:10, 0:10]``;
the file "4.2" stores the chunk in the fifth row third column,
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

For more details on the header, see the `c-blosc header description
<https://github.com/Blosc/c-blosc/blob/master/README_HEADER.rst>`_.

If a file does not exist on the file system for any given chunk in an
array, that indicates the chunk has not been initialised, and the
chunk should be interpreted as completely filled with whatever value
has been configured as the fill value for the array. I.e., chunk files
are not required to exist.

For example::

    >>> import zarr
    >>> z = zarr.open('example.zarr', mode='w', shape=(1000000, 1000),
    ...               chunks=(10000, 100), dtype='i4', fill_value=42,
    ...               compression='blosc', compression_opts=dict(cname='lz4',
    ...               clevel=3, shuffle=1))
    >>> import os
    >>> os.listdir('example.zarr')
    ['meta', 'attrs']
    >>> z[:] = 0
    >>> sorted(os.listdir('example.zarr'))[:5]
    ['0.0', '0.1', '0.2', '0.3', '0.4']
