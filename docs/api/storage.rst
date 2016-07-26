Storage (``zarr.storage``)
==========================
.. module:: zarr.storage

This module contains storage classes for use with Zarr arrays and groups.
However, note that any object implementing the ``MutableMapping`` interface
can be used as a Zarr array store.

.. autofunction:: init_array
.. autofunction:: init_group

.. autoclass:: MemoryStore
.. autoclass:: DirectoryStore
.. autoclass:: ZipStore
