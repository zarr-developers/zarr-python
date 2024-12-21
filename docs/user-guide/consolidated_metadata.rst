Consolidated Metadata
=====================

Zarr-Python implements the `Consolidated Metadata`_ extension to the Zarr Spec.
Consolidated metadata can reduce the time needed to load the metadata for an
entire hierarchy, especially when the metadata is being served over a network.
Consolidated metadata essentially stores all the metadata for a hierarchy in the
metadata of the root Group.

Usage
-----

If consolidated metadata is present in a Zarr Group's metadata then it is used
by default.  The initial read to open the group will need to communicate with
the store (reading from a file for a :class:`zarr.storage.LocalStore`, making a
network request for a :class:`zarr.storage.FsspecStore`). After that, any subsequent
metadata reads get child Group or Array nodes will *not* require reads from the store.

In Python, the consolidated metadata is available on the ``.consolidated_metadata``
attribute of the ``GroupMetadata`` object.

.. TODO: remove :okwarning: after warnings are removed

.. ipython:: python
   :okwarning:

   import zarr
   store = zarr.storage.MemoryStore()
   group = zarr.open_group(store=store)
   group.create_array(shape=(1,), name="a")
   group.create_array(shape=(2, 2), name="b")
   group.create_array(shape=(3, 3, 3), name="c")
   zarr.consolidate_metadata(store)

If we open that group, the Group's metadata has a :class:`zarr.core.group.ConsolidatedMetadata`
that can be used.

.. ipython:: python

   consolidated = zarr.open_group(store=store)
   consolidated.metadata.consolidated_metadata.metadata

Operations on the group to get children automatically use the consolidated metadata.

.. ipython:: python

   consolidated["a"]  # no read / HTTP request to the Store is required

With nested groups, the consolidated metadata is available on the children, recursively.

.. ipython:: python
   :okwarning:

    child = group.create_group("child", attributes={"kind": "child"})
    grandchild = child.create_group("child", attributes={"kind": "grandchild"})
    consolidated = zarr.consolidate_metadata(store)

    consolidated["child"].metadata.consolidated_metadata

Synchronization and Concurrency
-------------------------------

Consolidated metadata is intended for read-heavy use cases on slowly changing
hierarchies. For hierarchies where new nodes are constantly being added,
removed, or modified, consolidated metadata may not be desirable.

1. It will add some overhead to each update operation, since the metadata
   would need to be re-consolidated to keep it in sync with the store.
2. Readers using consolidated metadata will regularly see a "past" version
   of the metadata, at the time they read the root node with its consolidated
   metadata.

.. _Consolidated Metadata: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#consolidated-metadata