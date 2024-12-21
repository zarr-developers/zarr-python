
Working with groups
===================

.. _tutorial_groups:

Groups
------

Zarr supports hierarchical organization of arrays via groups. As with arrays,
groups can be stored in memory, on disk, or via other storage systems that
support a similar interface.

To create a group, use the :func:`zarr.group` function:

.. ipython:: python

   import zarr

   root = zarr.group()
   root

Groups have a similar API to the Group class from `h5py
<https://www.h5py.org/>`_.  For example, groups can contain other groups:

.. ipython:: python

   foo = root.create_group('foo')
   bar = foo.create_group('bar')

Groups can also contain arrays, e.g.:

.. ipython:: python

   z1 = bar.zeros(name='baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4')
   z1

Members of a group can be accessed via the suffix notation, e.g.:

.. ipython:: python

   root['foo']

The '/' character can be used to access multiple levels of the hierarchy in one
call, e.g.:

.. ipython:: python

   root['foo/bar']
   root['foo/bar/baz']

The :func:`zarr.Group.tree` method can be used to print a tree
representation of the hierarchy, e.g.:

.. ipython:: python

   root.tree()

The :func:`zarr.open` function provides a convenient way to create or
re-open a group stored in a directory on the file-system, with sub-groups stored in
sub-directories, e.g.:

.. ipython:: python
   :suppress:

   rm -r data/group.zarr

.. ipython:: python

   root = zarr.open_group('data/group.zarr', mode='w')
   root

   z = root.zeros(name='foo/bar/baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='i4')
   z

.. TODO: uncomment after __enter__ and __exit__ are implemented
.. Groups can be used as context managers (in a ``with`` statement).
.. If the underlying store has a ``close`` method, it will be called on exit.

For more information on groups see the :class:`zarr.Group` API docs.

.. _tutorial_diagnostics:

Array and group diagnostics
---------------------------

Diagnostic information about arrays and groups is available via the ``info``
property. E.g.:

.. ipython:: python

   root = zarr.group()

   foo = root.create_group('foo')

   bar = foo.zeros(name='bar', shape=1000000, chunks=100000, dtype='i8')

   bar[:] = 42

   baz = foo.zeros(name='baz', shape=(1000, 1000), chunks=(100, 100), dtype='f4')

   baz[:] = 4.2

   root.info

   foo.info

   bar.info_complete()

   baz.info

Groups also have the :func:`zarr.Group.tree` method, e.g.:

.. ipython:: python

   root.tree()

.. note::

   :func:`zarr.Group.tree` requires the optional `rich <https://rich.readthedocs.io/en/stable/>`_
   dependency. It can be installed with the ``[tree]`` extra.
