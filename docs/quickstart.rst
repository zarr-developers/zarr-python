.. only:: doctest

   >>> import shutil
   >>> shutil.rmtree('data', ignore_errors=True)
   >>>
   >>> import numpy as np
   >>> np.random.seed(0)

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

To get started, you can create a simple Zarr array::

    >>> import zarr
    >>> import numpy as np
    >>>
    >>> # Create a 2D Zarr array
    >>> z = zarr.create_array(
    ...    store="data/example-1.zarr",
    ...    shape=(100, 100),
    ...    chunks=(10, 10),
    ...    dtype="f4"
    ... )
    >>>
    >>> # Assign data to the array
    >>> z[:, :] = np.random.random((100, 100))
    >>> z.info
    Type               : Array
    Zarr format        : 3
    Data type          : DataType.float32
    Shape              : (100, 100)
    Chunk shape        : (10, 10)
    Order              : C
    Read-only          : False
    Store type         : LocalStore
    Codecs             : [{'endian': <Endian.little: 'little'>}, {'level': 0, 'checksum': False}]
    No. bytes          : 40000 (39.1K)

Here, we created a 2D array of shape ``(100, 100)``, chunked into blocks of
``(10, 10)``, and filled it with random floating-point data. This array was
written to a ``LocalStore`` in the ``data/example-1.zarr`` directory.

Compression and Filters
~~~~~~~~~~~~~~~~~~~~~~~

Zarr supports data compression and filters. For example, to use Blosc compression::

    >>> z = zarr.create_array(
    ...    "data/example-3.zarr",
    ...    mode="w", shape=(100, 100),
    ...    chunks=(10, 10), dtype="f4",
    ...    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle)
    ... )
    >>> z[:, :] = np.random.random((100, 100))
    >>>
    >>> z.info
    Type               : Array
    Zarr format        : 3
    Data type          : DataType.float32
    Shape              : (100, 100)
    Chunk shape        : (10, 10)
    Order              : C
    Read-only          : False
    Store type         : LocalStore
    Codecs             : [{'endian': <Endian.little: 'little'>}, {'level': 0, 'checksum': False}]
    No. bytes          : 40000 (39.1K)

This compresses the data using the Zstandard codec with shuffle enabled for better compression.

Hierarchical Groups
-------------------

Zarr allows you to create hierarchical groups, similar to directories::

    >>> # Create nested groups and add arrays
    >>> root = zarr.group("data/example-2.zarr")
    >>> foo = root.create_group(name="foo")
    >>> bar = root.create_array(
    ...     name="bar", shape=(100, 10), chunks=(10, 10), dtype="f4"
    ... )
    >>> spam = foo.create_array(name="spam", shape=(10,), dtype="i4")
    >>>
    >>> # Assign values
    >>> bar[:, :] = np.random.random((100, 10))
    >>> spam[:] = np.arange(10)
    >>>
    >>> # print the hierarchy
    >>> root.tree()
    /
    ├── bar (100, 10) float32
    └── foo
        └── spam (10,) int32
    <BLANKLINE>

This creates a group with two datasets: ``foo`` and ``bar``.

Batch Hierarchy Creation
~~~~~~~~~~~~~~~~~~~~~~~~

Zarr provides tools for creating a collection of arrays and groups with a single function call.
Suppose we want to copy existing groups and arrays into a new storage backend:

    >>> # Create nested groups and add arrays
    >>> root = zarr.group("data/example-3.zarr", attributes={'name': 'root'})
    >>> foo = root.create_group(name="foo")
    >>> bar = root.create_array(
    ...     name="bar", shape=(100, 10), chunks=(10, 10), dtype="f4"
    ... )
    >>> nodes = {'': root.metadata} | {k: v.metadata for k,v in root.members()}
    >>> print(nodes)
    >>> from zarr.storage import MemoryStore
    >>> new_nodes = dict(zarr.create_hierarchy(store=MemoryStore(), nodes=nodes))
    >>> new_root = new_nodes['']
    >>> assert new_root.attrs == root.attrs

Note that :func:`zarr.create_hierarchy` will only initialize arrays and groups -- copying array data must
be done in a separate step.

Persistent Storage
------------------

Zarr supports persistent storage to disk or cloud-compatible backends. While examples above
utilized a :class:`zarr.storage.LocalStore`, a number of other storage options are available.

Zarr integrates seamlessly with cloud object storage such as Amazon S3 and Google Cloud Storage
using external libraries like `s3fs <https://s3fs.readthedocs.io>`_ or
`gcsfs <https://gcsfs.readthedocs.io>`_::

    >>> import s3fs # doctest: +SKIP
    >>>
    >>> z = zarr.create_array("s3://example-bucket/foo", mode="w", shape=(100, 100), chunks=(10, 10), dtype="f4") # doctest: +SKIP
    >>> z[:, :] = np.random.random((100, 100)) # doctest: +SKIP

A single-file store can also be created using the the :class:`zarr.storage.ZipStore`::

    >>> # Store the array in a ZIP file
    >>> store = zarr.storage.ZipStore("data/example-3.zip", mode='w')
    >>>
    >>> z = zarr.create_array(
    ...     store=store,
    ...     mode="w",
    ...     shape=(100, 100),
    ...     chunks=(10, 10),
    ...     dtype="f4"
    ... )
    >>>
    >>> # write to the array
    >>> z[:, :] = np.random.random((100, 100))
    >>>
    >>> # the ZipStore must be explicitly closed
    >>> store.close()

To open an existing array from a ZIP file::

    >>> # Open the ZipStore in read-only mode
    >>> store = zarr.storage.ZipStore("data/example-3.zip", read_only=True)
    >>>
    >>> z = zarr.open_array(store, mode='r')
    >>>
    >>> # read the data as a NumPy Array
    >>> z[:]
    array([[0.66734236, 0.15667458, 0.98720884, ..., 0.36229587, 0.67443246,
            0.34315267],
        [0.65787303, 0.9544212 , 0.4830079 , ..., 0.33097172, 0.60423803,
            0.45621237],
        [0.27632037, 0.9947008 , 0.42434934, ..., 0.94860053, 0.6226942 ,
            0.6386924 ],
        ...,
        [0.12854576, 0.934397  , 0.19524333, ..., 0.11838563, 0.4967675 ,
            0.43074256],
        [0.82029045, 0.4671437 , 0.8090906 , ..., 0.7814118 , 0.42650765,
            0.95929915],
        [0.4335856 , 0.7565437 , 0.7828931 , ..., 0.48119593, 0.66220033,
            0.6652362 ]], shape=(100, 100), dtype=float32)

Read more about Zarr's storage options  in the :ref:`User Guide <user-guide-storage>`.

Next Steps
----------

Now that you're familiar with the basics, explore the following resources:

- `User Guide <user-guide>`_
- `API Reference <api>`_
