zarr
====

A minimal implementation of chunked, compressed, N-dimensional arrays
for Python.

* Source code: https://github.com/alimanfoo/zarr
* Download: https://pypi.python.org/pypi/zarr

.. image:: https://travis-ci.org/alimanfoo/zarr.svg?branch=master
    :target: https://travis-ci.org/alimanfoo/zarr

Installation
------------

Installation requires Numpy and Cython pre-installed. Can only be
installed on Linux currently.

Install from PyPI::

    $ pip install -U zarr

Install from GitHub::

    $ pip install -U git+https://github.com/alimanfoo/zarr.git@master

Status
------

Experimental, proof-of-concept. This is alpha-quality software. Things
may break, change or disappear without warning.

Bug reports and suggestions welcome.

Design goals
------------

* Chunking in multiple dimensions
* Resize any dimension
* Concurrent reads
* Concurrent writes
* Release the GIL during compression and decompression

Usage
-----

Create an array:

.. code-block::
     
    >>> import numpy as np
    >>> import zarr
    >>> z = zarr.empty(shape=(10000, 1000), dtype='i4', chunks=(1000, 100))
    >>> z
    zarr.ext.SynchronizedArray((10000, 1000), int32, chunks=(1000, 100))
      cname: blosclz; clevel: 5; shuffle: 1 (BYTESHUFFLE)
      nbytes: 38.1M; cbytes: 0; initialized: 0/100

Fill it with some data:

.. code-block::
     
    >>> z[:] = np.arange(10000000, dtype='i4').reshape(10000, 1000)
    >>> z
    zarr.ext.SynchronizedArray((10000, 1000), int32, chunks=(1000, 100))
      cname: blosclz; clevel: 5; shuffle: 1 (BYTESHUFFLE)
      nbytes: 38.1M; cbytes: 2.0M; ratio: 19.3; initialized: 100/100

Obtain a NumPy array by slicing:

.. code-block::
     
    >>> z[:]
    array([[      0,       1,       2, ...,     997,     998,     999],
           [   1000,    1001,    1002, ...,    1997,    1998,    1999],
           [   2000,    2001,    2002, ...,    2997,    2998,    2999],
           ...,
           [9997000, 9997001, 9997002, ..., 9997997, 9997998, 9997999],
           [9998000, 9998001, 9998002, ..., 9998997, 9998998, 9998999],
           [9999000, 9999001, 9999002, ..., 9999997, 9999998, 9999999]], dtype=int32)
    >>> z[:100]
    array([[    0,     1,     2, ...,   997,   998,   999],
           [ 1000,  1001,  1002, ...,  1997,  1998,  1999],
           [ 2000,  2001,  2002, ...,  2997,  2998,  2999],
           ...,
           [97000, 97001, 97002, ..., 97997, 97998, 97999],
           [98000, 98001, 98002, ..., 98997, 98998, 98999],
           [99000, 99001, 99002, ..., 99997, 99998, 99999]], dtype=int32)
    >>> z[:, :100]
    array([[      0,       1,       2, ...,      97,      98,      99],
           [   1000,    1001,    1002, ...,    1097,    1098,    1099],
           [   2000,    2001,    2002, ...,    2097,    2098,    2099],
           ...,
           [9997000, 9997001, 9997002, ..., 9997097, 9997098, 9997099],
           [9998000, 9998001, 9998002, ..., 9998097, 9998098, 9998099],
           [9999000, 9999001, 9999002, ..., 9999097, 9999098, 9999099]], dtype=int32)

Resize the array and add more data:

.. code-block::
     
    >>> z.resize(20000, 1000)
    >>> z
    zarr.ext.SynchronizedArray((20000, 1000), int32, chunks=(1000, 100))
      cname: blosclz; clevel: 5; shuffle: 1 (BYTESHUFFLE)
      nbytes: 76.3M; cbytes: 2.0M; ratio: 38.5; initialized: 100/200
    >>> z[10000:, :] = np.arange(10000000, dtype='i4').reshape(10000, 1000)
    >>> z
    zarr.ext.SynchronizedArray((20000, 1000), int32, chunks=(1000, 100))
      cname: blosclz; clevel: 5; shuffle: 1 (BYTESHUFFLE)
      nbytes: 76.3M; cbytes: 4.0M; ratio: 19.3; initialized: 200/200

For convenience, an ``append()`` method is also available, which can be used to
append data to any axis:

.. code-block::
     
    >>> a = np.arange(10000000, dtype='i4').reshape(10000, 1000)
    >>> z = zarr.array(a, chunks=(1000, 100))
    >>> z.append(a+a)
    >>> z
    zarr.ext.SynchronizedArray((20000, 1000), int32, chunks=(1000, 100))
      cname: blosclz; clevel: 5; shuffle: 1 (BYTESHUFFLE)
      nbytes: 76.3M; cbytes: 3.6M; ratio: 21.2; initialized: 200/200
    >>> z.append(np.vstack([a, a]), axis=1)
    >>> z
    zarr.ext.SynchronizedArray((20000, 2000), int32, chunks=(1000, 100))
      cname: blosclz; clevel: 5; shuffle: 1 (BYTESHUFFLE)
      nbytes: 152.6M; cbytes: 7.6M; ratio: 20.2; initialized: 400/400

Persistence
-----------

Create a persistent array (data stored on disk):

.. code-block::

    >>> path = 'example.zarr'
    >>> z = zarr.open(path, mode='w', shape=(10000, 1000), dtype='i4', chunks=(1000, 100))
    >>> z[:] = np.arange(10000000, dtype='i4').reshape(10000, 1000)
    >>> z
    zarr.ext.SynchronizedPersistentArray((10000, 1000), int32, chunks=(1000, 100))
      cname: blosclz; clevel: 5; shuffle: 1 (BYTESHUFFLE)
      nbytes: 38.1M; cbytes: 2.0M; ratio: 19.3; initialized: 100/100
      mode: w; path: example.zarr

There is no need to close a persistent array. Data are automatically flushed
to disk.

If you're working with really big arrays, try the 'lazy' option:

.. code-block::

    >>> path = 'big.zarr'
    >>> z = zarr.open(path, mode='w', shape=(1e8, 1e7), dtype='i4', chunks=(1000, 1000), lazy=True)
    >>> z
    zarr.ext.SynchronizedLazyPersistentArray((100000000, 10000000), int32, chunks=(1000, 1000))
      cname: blosclz; clevel: 5; shuffle: 1 (BYTESHUFFLE)
      nbytes: 3.6P; cbytes: 0; initialized: 0/1000000000
      mode: w; path: big.zarr

See the `persistence documentation <PERSISTENCE.rst>`_ for more
details of the file format.

Tuning
------

``zarr`` is optimised for accessing and storing data in contiguous
slices, of the same size or larger than chunks. It is not and probably
never will be optimised for single item access.

Chunks sizes >= 1M are generally good. Optimal chunk shape will depend
on the correlation structure in your data.

``zarr`` is designed for use in parallel computations working
chunk-wise over data. Try it with `dask.array
<http://dask.pydata.org/en/latest/array.html>`_. If using in a
multi-threaded, set zarr to use blosc in contextual mode::

    >>> zarr.set_blosc_options(use_context=True)

Acknowledgments
---------------

``zarr`` uses `c-blosc <https://github.com/Blosc/c-blosc>`_ internally for
compression and decompression and borrows code heavily from
`bcolz <http://bcolz.blosc.org/>`_.
