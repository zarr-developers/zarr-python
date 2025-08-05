.. _user-guide-consolidated-metadata:

Consolidated metadata
=====================

.. warning::
   The Consolidated Metadata feature in Zarr-Python is considered experimental for v3
   stores. `zarr-specs#309 <https://github.com/zarr-developers/zarr-specs/pull/309>`_
   has proposed a formal extension to the v3 specification to support consolidated metadata.

Zarr-Python implements the `Consolidated Metadata`_ for v2 and v3 stores.
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

   >>> import zarr
   >>> import warnings
   >>> warnings.filterwarnings("ignore", category=UserWarning)
   >>>
   >>> store = zarr.storage.MemoryStore()
   >>> group = zarr.create_group(store=store)
   >>> group.create_array(shape=(1,), name='a', dtype='float64')
   <Array memory://.../a shape=(1,) dtype=float64>
   >>> group.create_array(shape=(2, 2), name='b', dtype='float64')
   <Array memory://.../b shape=(2, 2) dtype=float64>
   >>> group.create_array(shape=(3, 3, 3), name='c', dtype='float64')
   <Array memory://.../c shape=(3, 3, 3) dtype=float64>
   >>> zarr.consolidate_metadata(store)
   <Group memory://...>

If we open that group, the Group's metadata has a :class:`zarr.core.group.ConsolidatedMetadata`
that can be used.:

   >>> consolidated = zarr.open_group(store=store)
   >>> consolidated_metadata = consolidated.metadata.consolidated_metadata.metadata
   >>> from pprint import pprint
   >>> pprint(dict(consolidated_metadata.items()))
   {'a': ArrayV3Metadata(shape=(1,),
                          data_type=Float64(endianness='little'),
                          chunk_grid=RegularChunkGrid(chunk_shape=(1,)),
                          chunk_key_encoding=DefaultChunkKeyEncoding(name='default',
                                                                     separator='/'),
                          fill_value=np.float64(0.0),
                          codecs=(BytesCodec(endian=<Endian.little: 'little'>),
                                  ZstdCodec(level=0, checksum=False)),
                          attributes={},
                          dimension_names=None,
                          zarr_format=3,
                          node_type='array',
                          storage_transformers=()),
     'b': ArrayV3Metadata(shape=(2, 2),
                          data_type=Float64(endianness='little'),
                          chunk_grid=RegularChunkGrid(chunk_shape=(2, 2)),
                          chunk_key_encoding=DefaultChunkKeyEncoding(name='default',
                                                                     separator='/'),
                          fill_value=np.float64(0.0),
                          codecs=(BytesCodec(endian=<Endian.little: 'little'>),
                                  ZstdCodec(level=0, checksum=False)),
                          attributes={},
                          dimension_names=None,
                          zarr_format=3,
                          node_type='array',
                          storage_transformers=()),
     'c': ArrayV3Metadata(shape=(3, 3, 3),
                          data_type=Float64(endianness='little'),
                          chunk_grid=RegularChunkGrid(chunk_shape=(3, 3, 3)),
                          chunk_key_encoding=DefaultChunkKeyEncoding(name='default',
                                                                     separator='/'),
                          fill_value=np.float64(0.0),
                          codecs=(BytesCodec(endian=<Endian.little: 'little'>),
                                  ZstdCodec(level=0, checksum=False)),
                          attributes={},
                          dimension_names=None,
                          zarr_format=3,
                          node_type='array',
                          storage_transformers=())}

Operations on the group to get children automatically use the consolidated metadata.:

   >>> consolidated['a']  # no read / HTTP request to the Store is required
   <Array memory://.../a shape=(1,) dtype=float64>

With nested groups, the consolidated metadata is available on the children, recursively.:

   >>> child = group.create_group('child', attributes={'kind': 'child'})
   >>> grandchild = child.create_group('child', attributes={'kind': 'grandchild'})
   >>> consolidated = zarr.consolidate_metadata(store)
   >>>
   >>> consolidated['child'].metadata.consolidated_metadata
   ConsolidatedMetadata(metadata={'child': GroupMetadata(attributes={'kind': 'grandchild'}, zarr_format=3, consolidated_metadata=ConsolidatedMetadata(metadata={}, kind='inline', must_understand=False), node_type='group')}, kind='inline', must_understand=False)

.. versionadded:: 3.1.1

    The keys in the consolidated metadata are sorted prior to writing. Keys are
    sorted in ascending order by path depth, where a path is defined as a sequence
    of strings joined by ``"/"``. For keys with the same path length, lexicographic
    order is used to break the tie.  This behaviour ensures deterministic metadata
    output for a given group.

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

.. _Consolidated Metadata: https://github.com/zarr-developers/zarr-specs/pull/309

Stores Without Support for Consolidated Metadata
------------------------------------------------

Some stores may want to opt out of the consolidated metadata mechanism. This
may be for several reasons like:

* They want to maintain read-write consistency, which is challenging with
  consolidated metadata.
* They have their own consolidated metadata mechanism.
* They offer good enough performance without need for consolidation.

This type of store can declare it doesn't want consolidation by implementing
`Store.supports_consolidated_metadata` and returning `False`. For stores that don't support
consolidation, Zarr will:

* Raise an error on `consolidate_metadata` calls, maintaining the store in
  its unconsolidated state.
* Raise an error in `AsyncGroup.open(..., use_consolidated=True)`
* Not use consolidated metadata in `AsyncGroup.open(..., use_consolidated=None)`
