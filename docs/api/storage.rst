Storage (``zarr.storage``)
==========================
.. module:: zarr.storage

This module contains a single :class:`DirectoryStore` class providing
a ``MutableMapping`` interface to a directory on the file
system. However, note that any object implementing the
``MutableMapping`` interface can be used as a Zarr array store.

.. autofunction:: init_store

.. autoclass:: DirectoryStore
