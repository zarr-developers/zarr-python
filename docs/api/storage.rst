Storage (``zarr.storage``)
==========================
.. automodule:: zarr.storage

.. autoclass:: DictStore
.. autoclass:: DirectoryStore
.. autoclass:: TempStore
.. autoclass:: NestedDirectoryStore
.. autoclass:: ZipStore

    .. automethod:: close
    .. automethod:: flush

.. autoclass:: DBMStore

    .. automethod:: close
    .. automethod:: flush

.. autoclass:: LMDBStore

    .. automethod:: close
    .. automethod:: flush

.. autoclass:: SQLiteStore

    .. automethod:: close

.. autoclass:: MongoDBStore
.. autoclass:: RedisStore
.. autoclass:: LRUStoreCache

    .. automethod:: invalidate
    .. automethod:: invalidate_values
    .. automethod:: invalidate_keys

.. autoclass:: ABSStore

.. autoclass:: ConsolidatedMetadataStore

.. autofunction:: init_array
.. autofunction:: init_group
.. autofunction:: contains_array
.. autofunction:: contains_group
.. autofunction:: listdir
.. autofunction:: rmdir
.. autofunction:: getsize
.. autofunction:: rename
.. autofunction:: migrate_1to2
