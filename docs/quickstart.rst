.. ipython:: python
   :suppress:

   rm -r data/

.. ipython:: python
   :suppress:

   import numpy as np
   np.random.seed(0)

Quickstart
==========

Welcome to the Zarr-Python Quickstart guide! This page will help you get up and running with
the Zarr library in Python to efficiently manage and analyze multi-dimensional arrays.

Zarr is a powerful library for storage of n-dimensional arrays, supporting chunking,
compression, and various backends, making it a versatile choice for scientific and
large-scale data.

Installation
------------

Zarr requires Python 3.11 or higher. You can install it via `pip`:

.. code-block:: bash

    pip install zarr

or `conda`:

.. code-block:: bash

    conda install --channel conda-forge zarr

Creating an Array
-----------------

To get started, you can create a simple Zarr array:

.. ipython:: python

    import zarr
    import numpy as np

    # Create a 2D Zarr array
    z = zarr.create_array(
        store="data/example-1.zarr",
        shape=(100, 100),
        chunks=(10, 10),
        dtype="f4"
    )

    # Assign data to the array
    z[:, :] = np.random.random((100, 100))
    z.info

Here, we created a 2D array of shape ``(100, 100)``, chunked into blocks of
``(10, 10)``, and filled it with random floating-point data. This array was
written to a ``LocalStore`` in the ``data/example-1.zarr`` directory.

Compression and Filters
~~~~~~~~~~~~~~~~~~~~~~~

Zarr supports data compression and filters. For example, to use Blosc compression:

.. ipython:: python

    z = zarr.create_array(
        "data/example-3.zarr",
        mode="w", shape=(100, 100),
        chunks=(10, 10), dtype="f4",
        compressor=zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.SHUFFLE)
    )
    z[:, :] = np.random.random((100, 100))

    z.info

This compresses the data using the Zstandard codec with shuffle enabled for better compression.

Hierarchical Groups
-------------------

Zarr allows you to create hierarchical groups, similar to directories:

.. ipython:: python

    # Create nested groups and add arrays
    root = zarr.group("data/example-2.zarr")
    foo = root.create_group(name="foo")
    bar = root.create_array(
        name="bar", shape=(100, 10), chunks=(10, 10)
    )
    spam = foo.create_array(name="spam", shape=(10,), dtype="i4")

    # Assign values
    bar[:, :] = np.random.random((100, 10))
    spam[:] = np.arange(10)

    # print the hierarchy
    root.tree()

This creates a group with two datasets: ``foo`` and ``bar``.

Persistent Storage
------------------

Zarr supports persistent storage to disk or cloud-compatible backends. While examples above
utilized a :class:`zarr.storage.LocalStore`, a number of other storage options are available,
including the :class:`zarr.storage.ZipStore` and :class:`zarr.storage.FsspecStore`.

.. ipython:: python

    # Store the array in a ZIP file
    store = zarr.storage.ZipStore("data/example-3.zip", mode='w')

    z = zarr.create_array(
        store=store,
        mode="w",
        shape=(100, 100),
        chunks=(10, 10),
        dtype="f4"
    )

    # write to the array
    z[:, :] = np.random.random((100, 100))

    # the ZipStore must be explicitly closed
    store.close()

To open an existing array:

.. ipython:: python

    # Open the ZipStore in read-only mode
    store = zarr.storage.ZipStore("data/example-3.zip", read_only=True)

    z = zarr.open_array(store, mode='r')

    # read the data as a NumPy Array
    z[:]

Cloud Storage Backends
~~~~~~~~~~~~~~~~~~~~~~

Zarr integrates seamlessly with cloud storage such as Amazon S3 and Google Cloud Storage
using external libraries like `s3fs <https://s3fs.readthedocs.io>`_ or
`gcsfs <https://gcsfs.readthedocs.io>`_.

For example, to use S3:

.. ipython:: python
   :verbatim:

    import s3fs

    z = zarr.create_array("s3://example-bucket/foo", mode="w", shape=(100, 100), chunks=(10, 10))
    z[:, :] = np.random.random((100, 100))

Read more about Zarr's :ref:`tutorial_storage` options in the User Guide.

Next Steps
----------

Now that you're familiar with the basics, explore the following resources:

- `User Guide <user-guide>`_
- `API Reference <api>`_
