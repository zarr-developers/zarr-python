.. only:: doctest

   >>> import shutil
   >>> shutil.rmtree('data', ignore_errors=True)

.. _user-guide-storage:

Storage guide
=============

Zarr-Python supports multiple storage backends, including: local file systems,
Zip files, remote stores via fsspec_ (S3, HTTP, etc.), and in-memory stores. In
Zarr-Python 3, stores must implement the abstract store API from
:class:`zarr.abc.store.Store`.

.. note::
   Unlike Zarr-Python 2 where the store interface was built around a generic ``MutableMapping``
   API, Zarr-Python 3 utilizes a custom store API that utilizes Python's AsyncIO library.

Implicit Store Creation
-----------------------

In most cases, it is not required to create a ``Store`` object explicitly. Passing a string
to Zarr's top level API will result in the store being created automatically.:

   >>> import zarr
   >>>
   >>> # Implicitly create a writable LocalStore
   >>> zarr.create_group(store='data/foo/bar')
   <Group file://data/foo/bar>
   >>>
   >>> # Implicitly create a read-only FsspecStore
   >>> zarr.open_group(
   ...    store='s3://noaa-nwm-retro-v2-zarr-pds',
   ...    mode='r',
   ...    storage_options={'anon': True}
   ... )
   <Group <FsspecStore(S3FileSystem, noaa-nwm-retro-v2-zarr-pds)>>
   >>>
   >>> # Implicitly creates a MemoryStore
   >>> data = {}
   >>> zarr.create_group(store=data)
   <Group memory://...>

Explicit Store Creation
-----------------------

In some cases, it may be helpful to create a store instance directly. Zarr-Python offers four
built-in store: :class:`zarr.storage.LocalStore`, :class:`zarr.storage.FsspecStore`,
:class:`zarr.storage.ZipStore`, :class:`zarr.storage.MemoryStore`, and :class:`zarr.storage.ObjectStore`.

Local Store
~~~~~~~~~~~

The :class:`zarr.storage.LocalStore` stores data in a nested set of directories on a local
filesystem.:

   >>> store = zarr.storage.LocalStore('data/foo/bar', read_only=True)
   >>> zarr.open_group(store=store, mode='r')
   <Group file://data/foo/bar>

Zip Store
~~~~~~~~~

The :class:`zarr.storage.ZipStore` stores the contents of a Zarr hierarchy in a single
Zip file. The `Zip Store specification`_ is currently in draft form.:

   >>> store = zarr.storage.ZipStore('data.zip', mode='w')
   >>> zarr.create_array(store=store, shape=(2,), dtype='float64')
   <Array zip://data.zip shape=(2,) dtype=float64>

Remote Store
~~~~~~~~~~~~

The :class:`zarr.storage.FsspecStore` stores the contents of a Zarr hierarchy in following the same
logical layout as the ``LocalStore``, except the store is assumed to be on a remote storage system
such as cloud object storage (e.g. AWS S3, Google Cloud Storage, Azure Blob Store). The
:class:`zarr.storage.FsspecStore` is backed by `fsspec`_ and can support any backend
that implements the `AbstractFileSystem <https://filesystem-spec.readthedocs.io/en/stable/api.html#fsspec.spec.AbstractFileSystem>`_
API. ``storage_options`` can be used to configure the fsspec backend.:

   >>> store = zarr.storage.FsspecStore.from_url(
   ...    's3://noaa-nwm-retro-v2-zarr-pds',
   ...    read_only=True,
   ...    storage_options={'anon': True}
   ... )
   >>> zarr.open_group(store=store, mode='r')
   <Group <FsspecStore(S3FileSystem, noaa-nwm-retro-v2-zarr-pds)>>

The type of filesystem (e.g. S3, https, etc..) is inferred from the scheme of the url (e.g. s3 for "**s3**://noaa-nwm-retro-v2-zarr-pds").
In case a specific filesystem is needed, one can explicitly create it. For example to create a S3 filesystem:

   >>> import fsspec
   >>> fs = fsspec.filesystem(
   ...    's3', anon=True, asynchronous=True,
   ...    client_kwargs={'endpoint_url': "https://noaa-nwm-retro-v2-zarr-pds.s3.amazonaws.com"}
   ... )
   >>> store = zarr.storage.FsspecStore(fs)

Memory Store
~~~~~~~~~~~~

The :class:`zarr.storage.MemoryStore` a in-memory store that allows for serialization of
Zarr data (metadata and chunks) to a dictionary.:

   >>> data = {}
   >>> store = zarr.storage.MemoryStore(data)
   >>> # TODO: replace with create_array after #2463
   >>> zarr.create_array(store=store, shape=(2,), dtype='float64')
   <Array memory://... shape=(2,) dtype=float64>

Object Store
~~~~~~~~~~~~

:class:`zarr.storage.ObjectStore` stores the contents of the Zarr hierarchy using any ObjectStore
`storage implementation <https://developmentseed.org/obstore/latest/api/store/>`_, including AWS S3 (:class:`obstore.store.S3Store`), Google Cloud Storage (:class:`obstore.store.GCSStore`), and Azure Blob Storage (:class:`obstore.store.AzureStore`). This store is backed by `obstore <https://developmentseed.org/obstore/latest/>`_, which
builds on the production quality Rust library `object_store <https://docs.rs/object_store/latest/object_store/>`_.


   >>> from zarr.storage import ObjectStore
   >>> from obstore.store import MemoryStore
   >>>
   >>> store = ObjectStore(MemoryStore())
   >>> zarr.create_array(store=store, shape=(2,), dtype='float64')
   <Array object_store://... shape=(2,) dtype=float64>

Here's an example of using ObjectStore for accessing remote data:

   >>> from zarr.storage import ObjectStore
   >>> from obstore.store import S3Store
   >>>
   >>> s3_store = S3Store('noaa-nwm-retro-v2-zarr-pds', skip_signature=True, region="us-west-2")
   >>> store = zarr.storage.ObjectStore(store=s3_store, read_only=True)
   >>> group = zarr.open_group(store=store, mode='r')
   >>> group.info
   Name        :
   Type        : Group
   Zarr format : 2
   Read-only   : True
   Store type  : ObjectStore
   No. members : 12
   No. arrays  : 12
   No. groups  : 0

.. warning::
   The :class:`zarr.storage.ObjectStore` class is experimental.

URL-based Storage (ZEP 8)
-------------------------

Zarr-Python supports URL-based storage specification following `ZEP 8: Zarr URL Specification`_.
This allows you to specify complex storage configurations using a concise URL syntax with chained adapters.

ZEP 8 URLs use pipe (``|``) characters to chain storage adapters together:

   >>> # Basic ZIP file storage
   >>> zarr.open_array("file:zep8-data.zip|zip", mode='w', shape=(10, 10), chunks=(5, 5), dtype="f4")
   <Array zip://zep8-data.zip shape=(10, 10) dtype=float32>

The general syntax is::

   scheme:path|adapter1|adapter2|...

Where:

* ``scheme:path`` specifies the base storage location
* ``|adapter`` chains storage adapters to transform or wrap the storage

Common ZEP 8 URL patterns:

**Local ZIP files:**

   >>> # Create data in a ZIP file
   >>> z = zarr.open_array("file:example.zip|zip", mode='w', shape=(100, 100), chunks=(10, 10), dtype="i4")
   >>> import numpy as np
   >>> z[:, :] = np.random.randint(0, 100, size=(100, 100))

**Remote ZIP files:**

   >>> # Access ZIP file from S3 (requires s3fs)
   >>> zarr.open_array("s3://bucket/data.zip|zip", mode='r')  # doctest: +SKIP

**In-memory storage:**

   >>> # Create array in memory
   >>> z = zarr.open_array("memory:", mode='w', shape=(5, 5), dtype="f4")
   >>> z[:, :] = np.random.random((5, 5))

**With format specification:**

   >>> # Specify Zarr format version
   >>> zarr.create_array("file:data-v3.zip|zip|zarr3", shape=(10,), dtype="i4")  # doctest: +SKIP

**Debugging with logging:**

   >>> # Log all operations on any store type
   >>> z = zarr.open_array("memory:|log:", mode='w', shape=(5, 5), dtype="f4")  # doctest: +SKIP
   >>> # 2025-08-24 20:01:13,282 - LoggingStore(memory://...) - INFO -  Calling MemoryStore.set(zarr.json)
   >>>
   >>> # Log operations on ZIP files with custom log level
   >>> z = zarr.open_array("file:debug.zip|zip:|log:?log_level=INFO", mode='w')  # doctest: +SKIP
   >>>
   >>> # Log operations on remote cloud storage
   >>> z = zarr.open_array("s3://bucket/data.zarr|log:", mode='r')  # doctest: +SKIP

Available adapters:

* ``file`` - Local filesystem paths
* ``zip`` - ZIP file storage
* ``memory`` - In-memory storage
* ``s3``, ``gs``, ``gcs`` - Cloud storage (requires appropriate fsspec backends)
* ``log`` - Logging wrapper for debugging store operations
* ``zarr2``, ``zarr3`` - Format specification adapters

You can programmatically discover all available adapters using :func:`zarr.registry.list_store_adapters`:

   >>> import zarr
   >>> zarr.registry.list_store_adapters()  # doctest: +SKIP
   ['file', 'gcs', 'gs', 'https', 'memory', 's3', 'zip', ...]

Additional adapters can be implemented as described in the `extending guide <./extending.html#custom-store-adapters>`_.

.. note::
   When using ZEP 8 URLs with third-party libraries like xarray, the URL syntax allows
   seamless integration without requiring zarr-specific store creation.

.. _ZEP 8\: Zarr URL Specification: https://zarr.dev/zeps/draft/ZEP0008.html

.. _user-guide-custom-stores:

Developing custom stores
------------------------

Zarr-Python :class:`zarr.abc.store.Store` API is meant to be extended. The Store Abstract Base
Class includes all of the methods needed to be a fully operational store in Zarr Python.
Zarr also provides a test harness for custom stores: :class:`zarr.testing.store.StoreTests`.

.. _Zip Store Specification: https://github.com/zarr-developers/zarr-specs/pull/311
.. _fsspec: https://filesystem-spec.readthedocs.io
