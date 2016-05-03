Storage (``zarr.storage``)
==========================
.. module:: zarr.storage

Note that any object implementing the ``MutableMapping`` interface can be used
as a Zarr array store.

This module contains a single :class:`DirectoryStore` class providing a
``MutableMapping`` interface to a directory on the file system.

.. autoclass:: DirectoryStore
