Storage
=======

Zarr-Python supports multiple storage backends, including: local file systems,
Zip files, remote stores via ``fsspec`` (S3, HTTP, etc.), and in-memory stores. In
Zarr-Python 3, stores must implement the abstract store API from
:class:`zarr.abc.store.Store`.

.. note::
   Unlike Zarr-Python 2 where the store interface was built around a generic ``MutableMapping``
   API, Zarr-Python 3 utilizes a custom store API that utilizes Python's AsyncIO library.

Implicit Store Creation
-----------------------

In most cases, it is not required to create a ``Store`` object explicitly. Passing a string
to Zarr's top level API will result in the store being created automatically.

.. code-block:: python

   >>> import zarr
   >>> zarr.open("data/foo/bar", mode="r")  # implicitly creates a read-only LocalStore
   <Group file://data/foo/bar>
   >>> zarr.open("s3://foo/bar", mode="r")  # implicitly creates a read-only RemoteStore
   <Group s3://foo/bar>
   >>> data = {}
   >>> zarr.open(data, mode="w")  # implicitly creates a MemoryStore
   <Group memory://4791444288>

Explicit Store Creation
-----------------------

In some cases, it may be helpful to create a store instance directly. Zarr-Python offers four
built-in store: :class:`zarr.storage.LocalStore`, :class:`zarr.storage.RemoteStore`,
:class:`zarr.storage.ZipStore`, and :class:`zarr.storage.MemoryStore`.

Local Store
~~~~~~~~~~~

The :class:`zarr.storage.LocalStore` stores data in a nested set of directories on a local
filesystem.

.. code-block:: python

   >>> import zarr
   >>> store = zarr.storage.LocalStore("data/foo/bar", read_only=True)
   >>> zarr.open(store=store)
   <Group file://data/foo/bar>

Zip Store
~~~~~~~~~

The :class:`zarr.storage.ZipStore` stores the contents of a Zarr hierarchy in a single
Zip file. The `Zip Store specification_` is currently in draft form.

.. code-block:: python

   >>> import zarr
   >>> store = zarr.storage.ZipStore("data.zip", mode="w")
   >>> zarr.open(store=store, shape=(2,))
   <Array zip://data.zip shape=(2,) dtype=float64

Remote Store
~~~~~~~~~~~~

The :class:`zarr.storage.RemoteStore` stores the contents of a Zarr hierarchy in following the same
logical layout as the ``LocalStore``, except the store is assumed to be on a remote storage system
such as cloud object storage (e.g. AWS S3, Google Cloud Storage, Azure Blob Store). The
:class:`zarr.storage.RemoteStore` is backed by `Fsspec_` and can support any Fsspec backend
that implements the `AbstractFileSystem` API,

.. code-block:: python

   >>> import zarr
   >>> store = zarr.storage.RemoteStore.from_url("gs://foo/bar", read_only=True)
   >>> zarr.open(store=store)
   <Array <RemoteStore(GCSFileSystem, foo/bar)> shape=(10, 20) dtype=float32>

Memory Store
~~~~~~~~~~~~

The :class:`zarr.storage.RemoteStore` a in-memory store that allows for serialization of
Zarr data (metadata and chunks) to a dictionary.

.. code-block:: python

   >>> import zarr
   >>> data = {}
   >>> store = zarr.storage.MemoryStore(data)
   >>> zarr.open(store=store, shape=(2, ))
   <Array memory://4943638848 shape=(2,) dtype=float64>

Developing custom stores
------------------------

Zarr-Python :class:`zarr.abc.store.Store` API is meant to be extended. The Store Abstract Base
Class includes all of the methods needed to be a fully operational store in Zarr Python.
Zarr also provides a test harness for custom stores: :class:`zarr.testing.store.StoreTests`.

.. _Zip Store Specification: https://github.com/zarr-developers/zarr-specs/pull/311
.. _Fsspec: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#consolidated-metadata
