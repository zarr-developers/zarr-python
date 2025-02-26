.. only:: doctest

   >>> import shutil
   >>> shutil.rmtree('data', ignore_errors=True)

.. _user-guide-groups:

Working with groups
===================

Zarr supports hierarchical organization of arrays via groups. As with arrays,
groups can be stored in memory, on disk, or via other storage systems that
support a similar interface.

To create a group, use the :func:`zarr.group` function::

   >>> import zarr
   >>> store = zarr.storage.MemoryStore()
   >>> root = zarr.create_group(store=store)
   >>> root
   <Group memory://...>

Groups have a similar API to the Group class from `h5py
<https://www.h5py.org/>`_.  For example, groups can contain other groups::

   >>> foo = root.create_group('foo')
   >>> bar = foo.create_group('bar')

Groups can also contain arrays, e.g.::

   >>> z1 = bar.create_array(name='baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
   >>> z1
   <Array memory://.../foo/bar/baz shape=(10000, 10000) dtype=int32>

Members of a group can be accessed via the suffix notation, e.g.::

   >>> root['foo']
   <Group memory://.../foo>

The '/' character can be used to access multiple levels of the hierarchy in one
call, e.g.::

   >>> root['foo/bar']
   <Group memory://.../foo/bar>
   >>> root['foo/bar/baz']
   <Array memory://.../foo/bar/baz shape=(10000, 10000) dtype=int32>

The :func:`zarr.Group.tree` method can be used to print a tree
representation of the hierarchy, e.g.::

   >>> root.tree()
   /
   └── foo
       └── bar
           └── baz (10000, 10000) int32
   <BLANKLINE>

The :func:`zarr.open_group` function provides a convenient way to create or
re-open a group stored in a directory on the file-system, with sub-groups stored in
sub-directories, e.g.::

   >>> root = zarr.open_group('data/group.zarr', mode='w')
   >>> root
   <Group file://data/group.zarr>
   >>>
   >>> z = root.create_array(name='foo/bar/baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
   >>> z
   <Array file://data/group.zarr/foo/bar/baz shape=(10000, 10000) dtype=int32>

.. TODO: uncomment after __enter__ and __exit__ are implemented
.. Groups can be used as context managers (in a ``with`` statement).
.. If the underlying store has a ``close`` method, it will be called on exit.

For more information on groups see the :class:`zarr.Group` API docs.

.. _user-guide-diagnostics:

Batch Group Creation
--------------------

You can also create multiple groups concurrently with a single function call. :func:`zarr.create_hierarchy` takes
a :class:`zarr.storage.Store` instance and a dict of ``key : metadata`` pairs, parses that dict, and
writes metadata documents to storage:

   >>> from zarr import create_hierarchy
   >>> from zarr.core.group import GroupMetadata
   >>> from zarr.storage import LocalStore
   >>> node_spec = {'a/b/c': GroupMetadata()}
   >>> nodes_created = dict(create_hierarchy(store=LocalStore(root='data'), nodes=node_spec))
   >>> print(sorted(nodes_created.items(), key=lambda kv: len(kv[0])))
   [('', <Group file://data>), ('a', <Group file://data/a>), ('a/b', <Group file://data/a/b>), ('a/b/c', <Group file://data/a/b/c>)]

Note that we only specified a single group named ``a/b/c``, but 4 groups were created. These additional groups
were created to ensure that the desired node ``a/b/c`` is connected to the root group ``''`` by a sequence
of intermediate groups. :func:`zarr.create_hierarchy` normalizes the ``nodes`` keyword argument to
ensure that the resulting hierarchy is complete, i.e. all groups or arrays are connected to the root
of the hierarchy via intermediate groups.

Because :func:`zarr.create_hierarchy` concurrently creates metadata documents, it's more efficient
than repeated calls to :func:`create_group` or :func:`create_array`, provided you can statically define
the metadata for the groups and arrays you want to create.

Array and group diagnostics
---------------------------

Diagnostic information about arrays and groups is available via the ``info``
property. E.g.::

   >>> store = zarr.storage.MemoryStore()
   >>> root = zarr.group(store=store)
   >>> foo = root.create_group('foo')
   >>> bar = foo.create_array(name='bar', shape=1000000, chunks=100000, dtype='int64')
   >>> bar[:] = 42
   >>> baz = foo.create_array(name='baz', shape=(1000, 1000), chunks=(100, 100), dtype='float32')
   >>> baz[:] = 4.2
   >>> root.info
   Name        :
   Type        : Group
   Zarr format : 3
   Read-only   : False
   Store type  : MemoryStore
   >>> foo.info
   Name        : foo
   Type        : Group
   Zarr format : 3
   Read-only   : False
   Store type  : MemoryStore
   >>> bar.info_complete()
   Type               : Array
   Zarr format        : 3
   Data type          : DataType.int64
   Shape              : (1000000,)
   Chunk shape        : (100000,)
   Order              : C
   Read-only          : False
   Store type         : MemoryStore
   Filters            : ()
   Serializer         : BytesCodec(endian=<Endian.little: 'little'>)
   Compressors        : (ZstdCodec(level=0, checksum=False),)
   No. bytes          : 8000000 (7.6M)
   No. bytes stored   : 1614
   Storage ratio      : 4956.6
   Chunks Initialized : 0
   >>> baz.info
   Type               : Array
   Zarr format        : 3
   Data type          : DataType.float32
   Shape              : (1000, 1000)
   Chunk shape        : (100, 100)
   Order              : C
   Read-only          : False
   Store type         : MemoryStore
   Filters            : ()
   Serializer         : BytesCodec(endian=<Endian.little: 'little'>)
   Compressors        : (ZstdCodec(level=0, checksum=False),)
   No. bytes          : 4000000 (3.8M)

Groups also have the :func:`zarr.Group.tree` method, e.g.::

   >>> root.tree()
   /
   └── foo
       ├── bar (1000000,) int64
       └── baz (1000, 1000) float32
   <BLANKLINE>

.. note::

   :func:`zarr.Group.tree` requires the optional `rich <https://rich.readthedocs.io/en/stable/>`_
   dependency. It can be installed with the ``[tree]`` extra.
