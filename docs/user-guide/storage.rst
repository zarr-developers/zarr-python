.. _tutorial_storage:

Storage
=======

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
to Zarr's top level API will result in the store being created automatically.

.. ipython:: python

   import zarr

   # implicitly create a writable LocalStore
   zarr.open_group("data/foo/bar", mode="w")

   # implicitly create a read-only FsspecStore
   zarr.open_group(
      "s3://noaa-nwm-retro-v2-zarr-pds",
      mode="r",
      storage_options={"anon": True}
   )

   # implicitly creates a MemoryStore
   data = {}
   zarr.open_group(data, mode="w")

Explicit Store Creation
-----------------------

In some cases, it may be helpful to create a store instance directly. Zarr-Python offers four
built-in store: :class:`zarr.storage.LocalStore`, :class:`zarr.storage.FsspecStore`,
:class:`zarr.storage.ZipStore`, and :class:`zarr.storage.MemoryStore`.

Local Store
~~~~~~~~~~~

The :class:`zarr.storage.LocalStore` stores data in a nested set of directories on a local
filesystem.

.. ipython:: python

   store = zarr.storage.LocalStore("data/foo/bar", read_only=True)
   zarr.open(store=store, mode='r')

Zip Store
~~~~~~~~~

The :class:`zarr.storage.ZipStore` stores the contents of a Zarr hierarchy in a single
Zip file. The `Zip Store specification`_ is currently in draft form.

.. ipython:: python

   store = zarr.storage.ZipStore("data.zip", mode="w")
   zarr.open(store=store, shape=(2,))

Remote Store
~~~~~~~~~~~~

The :class:`zarr.storage.FsspecStore` stores the contents of a Zarr hierarchy in following the same
logical layout as the ``LocalStore``, except the store is assumed to be on a remote storage system
such as cloud object storage (e.g. AWS S3, Google Cloud Storage, Azure Blob Store). The
:class:`zarr.storage.FsspecStore` is backed by `fsspec`_ and can support any backend
that implements the `AbstractFileSystem` API. ``storage_options`` can be used to configure
the fsspec backend.

.. ipython:: python

   store = zarr.storage.FsspecStore.from_url(
      "s3://noaa-nwm-retro-v2-zarr-pds",
      read_only=True,
      storage_options={"anon": True}
   )
   zarr.open_group(store=store, mode='r')

Memory Store
~~~~~~~~~~~~

The :class:`zarr.storage.MemoryStore` a in-memory store that allows for serialization of
Zarr data (metadata and chunks) to a dictionary.

.. ipython:: python

   data = {}
   store = zarr.storage.MemoryStore(data)
   zarr.open(store=store, shape=(2, ))

.. _user-guide-custom-stores:

Developing custom stores
------------------------

Zarr-Python :class:`zarr.abc.store.Store` API is meant to be extended. The Store Abstract Base
Class includes all of the methods needed to be a fully operational store in Zarr Python.
Zarr also provides a test harness for custom stores: :class:`zarr.testing.store.StoreTests`.

.. _Zip Store Specification: https://github.com/zarr-developers/zarr-specs/pull/311
.. _fsspec: https://filesystem-spec.readthedocs.io
