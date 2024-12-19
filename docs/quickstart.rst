.. ipython::
   :suppress:

   In [999]: rm -r data/

   In [999]: import numpy as np

   In [999]: np.random.seed(0)

Quickstart
==========

Welcome to the Zarr-Python Quickstart guide! This page will help you get up and running with the Zarr library in Python to efficiently manage and analyze multi-dimensional arrays.

Zarr is a powerful library for storage of n-dimensional arrays, supporting chunking, compression, and various backends, making it a versatile choice for scientific and large-scale data.

Installation
------------

Zarr requires Python 3.10 or higher. You can install it via `pip`:

.. code-block:: bash

    pip install zarr

or `conda`:

.. code-block:: bash

    conda install --channel conda-forge zarr

Creating an Array
-----------------

To get started, you can create a simple Zarr array using the in-memory store:

.. ipython:: python

    import zarr
    import numpy as np

    # Create a 2D Zarr array
    z = zarr.zeros((100, 100), chunks=(10, 10), dtype='f4')

    # Assign data to the array
    z[:, :] = np.random.random((100, 100))

    print(z.info)

Here, we created a 2D array of shape ``(100, 100)``, chunked into blocks of ``(10, 10)``, and filled it with random floating-point data.

Persistent Storage
------------------

Zarr supports persistent storage to disk or cloud-compatible backends. To store arrays on the filesystem:

.. ipython:: python

    # Store the array in a directory on disk
    z = zarr.open(
        'data/example-1.zarr',
        mode='w', shape=(100, 100), chunks=(10, 10), dtype='f4'
    )
    z[:, :] = np.random.random((100, 100))

    print("Array stored at 'data/example-1.zarr'")

To open an existing array:

.. ipython:: python

    z = zarr.open('data/example-1.zarr', mode='r')
    print(z[:])

Hierarchical Groups
-------------------

Zarr allows creating hierarchical groups, similar to directories:

.. ipython:: python

    # Create a group and add arrays
    root = zarr.group('data/example-2.zarr')
    foo = root.create_array(name='foo', shape=(1000, 100), chunks=(10, 10), dtype='f4')
    bar = root.create_array(name='bar', shape=(100,), dtype='i4')

    # Assign values
    foo[:, :] = np.random.random((1000, 100))
    bar[:] = np.arange(100)

    root.tree()

This creates a group with two datasets: ``foo`` and ``bar``.

.. Compression and Filters
.. -----------------------

.. Zarr supports data compression and filters. For example, to use Blosc compression:

.. .. ipython:: python
..    :verbatim:

..     z = zarr.open('data/example-3.zarr', mode='w', shape=(100, 100), chunks=(10, 10), dtype='f4',
..                   compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE))
..     z[:, :] = np.random.random((100, 100))

..     print(z.info)

.. This compresses the data using the Zstandard codec with shuffle enabled for better compression.

Cloud Storage Backends
----------------------

Zarr integrates seamlessly with cloud storage such as Amazon S3 and Google Cloud Storage using external libraries like `s3fs` or `gcsfs`.

For example, to use S3:

.. ipython:: python
   :verbatim:

    import s3fs
    import zarr

    z = zarr.open("s3://example-bucket/foo", mode='w', shape=(100, 100), chunks=(10, 10))
    z[:, :] = np.random.random((100, 100))

Next Steps
----------

Now that you're familiar with the basics, explore the following resources:

- `User Guide <guide>`_
- `API Reference <api>`_
