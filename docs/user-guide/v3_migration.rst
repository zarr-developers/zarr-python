.. _v3 migration guide:

3.0 Migration Guide
===================

Zarr-Python 3 represents a major refactor of the Zarr-Python codebase. Some of the
goals motivating this refactor included:

* adding support for the Zarr format 3 specification (along with the Zarr format 2 specification)
* cleaning up internal and user facing APIs
* improving performance (particularly in high latency storage environments like
  cloud object stores)

To accommodate this, Zarr-Python 3 introduces a number of changes to the API, including a number
of significant breaking changes and deprecations.

This page provides a guide explaining breaking changes and deprecations to help you
migrate your code from version 2 to version 3. If we have missed anything, please
open a `GitHub issue <https://github.com/zarr-developers/zarr-python/issues/new>`_
so we can improve this guide.

Compatibility target
--------------------

The goals described above necessitated some breaking changes to the API (hence the
major version update), but where possible we have maintained backwards compatibility
in the most widely used parts of the API. This in the :class:`zarr.Array` and
:class:`zarr.Group` classes and the "top-level API" (e.g. :func:`zarr.open_array` and
:func:`zarr.open_group`).

Getting ready for 3.0
---------------------

Before migrating to Zarr-Python 3, we suggest projects that depend on Zarr-Python take
the following actions in order:

1. Pin the supported Zarr-Python version to ``zarr>=2,<3``. This is a best practice
   and will protect your users from any incompatibilities that may arise during the
   release of Zarr-Python 3. This pin can be removed after migrating to Zarr-Python 3.
2. Limit your imports from the Zarr-Python package. Most of the primary API ``zarr.*``
   will be compatible in Zarr-Python 3. However, the following breaking API changes are
   planned:

   - ``numcodecs.*`` will no longer be available in ``zarr.*``. To migrate, import codecs
     directly from ``numcodecs``:

     .. code-block:: python

        from numcodecs import Blosc
        # instead of:
        # from zarr import Blosc

   - The ``zarr.v3_api_available`` feature flag is being removed. In Zarr-Python 3
     the v3 API is always available, so you shouldn't need to use this flag.
   - The following internal modules are being removed or significantly changed. If
     your application relies on imports from any of the below modules, you will need
     to either a) modify your application to no longer rely on these imports or b)
     vendor the parts of the specific modules that you need.

     * ``zarr.attrs`` has gone, with no replacement
     * ``zarr.codecs`` has gone, use ``numcodecs`` instead
     * ``zarr.context`` has gone, with no replacement
     * ``zarr.core`` remains but should be considered private API
     * ``zarr.hierarchy`` has gone, with no replacement (use ``zarr.Group`` inplace of ``zarr.hierarchy.Group``)
     * ``zarr.indexing`` has gone, with no replacement
     * ``zarr.meta`` has gone, with no replacement
     * ``zarr.meta_v1`` has gone, with no replacement
     * ``zarr.sync`` has gone, with no replacement
     * ``zarr.types`` has gone, with no replacement
     * ``zarr.util`` has gone, with no replacement
     * ``zarr.n5`` has gone, see below for an alternative N5 options

3. Test that your package works with version 3.
4. Update the pin to include ``zarr>=3,<4``.

Zarr-Python 2 support window
----------------------------

Zarr-Python 2.x is still available, though we recommend migrating to Zarr-Python 3 for
its performance improvements and new features. Security and bug fixes will be made to
the 2.x series for at least six months following the first Zarr-Python 3 release.
If you need to use the latest Zarr-Python 2 release, you can install it with:

.. code-block:: console

    $ pip install "zarr==2.*"

.. note::
   Development and maintenance of the 2.x release series has moved to the
   `support/v2 <https://github.com/zarr-developers/zarr-python/tree/support/v2>`_ branch.
   Issues and pull requests related to this branch are tagged with the
   `V2 <https://github.com/zarr-developers/zarr-python/labels/V2>`_ label.

Migrating to Zarr-Python 3
--------------------------

The following sections provide details on breaking changes in Zarr-Python 3.

The Array class
~~~~~~~~~~~~~~~

1. Disallow direct construction - the signature for initializing the ``Array`` class has changed
   significantly. Please use :func:`zarr.create_array` or :func:`zarr.open_array` instead of
   directly constructing the :class:`zarr.Array` class.

2. Defaulting to ``zarr_format=3`` - newly created arrays will use the version 3 of the
   Zarr specification. To continue using version 2, set ``zarr_format=2`` when creating arrays
   or set ``default_zarr_version=2`` in Zarr's :ref:`runtime configuration <user-guide-config>`.

The Group class
~~~~~~~~~~~~~~~

1. Disallow direct construction - use :func:`zarr.open_group` or :func:`zarr.create_group`
   instead of directly constructing the :class:`zarr.Group` class.
2. Most of the h5py compatibility methods are deprecated and will issue warnings if used.
   The following functions are drop in replacements that have the same signature and functionality:

   - Use :func:`zarr.Group.create_array` in place of :func:`zarr.Group.create_dataset`
   - Use :func:`zarr.Group.require_array` in place of :func:`zarr.Group.require_dataset`

The Store class
~~~~~~~~~~~~~~~

The Store API has changed significant in Zarr-Python 3. The most notable changes to the
Store API are:

Store Import Paths
^^^^^^^^^^^^^^^^^^
Several store implementations have moved from the top-level module to ``zarr.storage``:

.. code-block:: diff
   :caption: Store import changes from v2 to v3

   # Before (v2)
   - from zarr import MemoryStore, DirectoryStore
   + from zarr.storage import MemoryStore, LocalStore  # LocalStore replaces DirectoryStore

Common replacements:

+-------------------------+------------------------------------+
| v2 Import               | v3 Import                          |
+=========================+====================================+
| ``zarr.MemoryStore``    | ``zarr.storage.MemoryStore``       |
+-------------------------+------------------------------------+
| ``zarr.DirectoryStore`` | ``zarr.storage.LocalStore``        |
+-------------------------+------------------------------------+
| ``zarr.TempStore``      | Use ``tempfile.TemporaryDirectory``|
|                         | with ``LocalStore``                |
+-------------------------+------------------------------------+

1. Replaced the ``MutableMapping`` base class in favor of a custom abstract base class
   (:class:`zarr.abc.store.Store`).
2. Switched to an asynchronous interface for all store methods that result in IO. This
   change ensures that all store methods are non-blocking and are as performant as
   possible.

Beyond the changes store interface, a number of deprecated stores were also removed in
Zarr-Python 3. See :issue:`1274` for more details on the removal of these stores.

- ``N5Store`` - see https://github.com/zarr-developers/n5py for an alternative interface to
  N5 formatted data.
- ``ABSStore`` - use the :class:`zarr.storage.FsspecStore` instead along with fsspec's
  `adlfs backend <https://github.com/fsspec/adlfs>`_.

The following stores have been removed altogether. Users who need these stores will have to
implement their own version in zarr-python v3.

- ``DBMStore``
- ``LMDBStore``
- ``SQLiteStore``
- ``MongoDBStore``
- ``RedisStore``

At present, the latter five stores in this list do not have an equivalent in Zarr-Python 3.
If you are interested in developing a custom store that targets these backends, see
:ref:`developing custom stores <user-guide-custom-stores>` or open an
`issue <https://github.com/zarr-developers/zarr-python/issues>`_ to discuss your use case.

Dependencies
~~~~~~~~~~~~

When installing using ``pip``:

- The new ``remote`` dependency group can be used to install a supported version of
  ``fsspec``, required for remote data access.
- The new ``gpu`` dependency group can be used to install a supported version of
  ``cuda``, required for GPU functionality.
- The ``jupyter`` optional dependency group has been removed, since v3 contains no
  jupyter specific functionality.

Miscellaneous
~~~~~~~~~~~~~

- The keyword argument ``zarr_version`` available in most creation functions in :mod:`zarr`
  (e.g. :func:`zarr.create`, :func:`zarr.open`, :func:`zarr.group`, :func:`zarr.array`) has
  been deprecated in favor of ``zarr_format``.

🚧 Work in Progress 🚧
----------------------

Zarr-Python 3 is still under active development, and is not yet fully complete.
The following list summarizes areas of the codebase that we expect to build out
after the 3.0.0 release. If features listed below are important to your use case
of Zarr-Python, please open (or comment on) a
`GitHub issue <https://github.com/zarr-developers/zarr-python/issues/new>`_.

- The following functions / methods have not been ported to Zarr-Python 3 yet:

  * :func:`zarr.copy` (:issue:`2407`)
  * :func:`zarr.copy_all` (:issue:`2407`)
  * :func:`zarr.copy_store` (:issue:`2407`)
  * :func:`zarr.Group.move` (:issue:`2108`)

- The following features (corresponding to function arguments to functions in
  :mod:`zarr`) have not been ported to Zarr-Python 3 yet. Using these features
  will raise a warning or a ``NotImplementedError``:

  * ``cache_attrs``
  * ``cache_metadata``
  * ``chunk_store`` (:issue:`2495`)
  * ``meta_array``
  * ``object_codec`` (:issue:`2617`)
  * ``synchronizer`` (:issue:`1596`)
  * ``dimension_separator``

- The following features that were supported by Zarr-Python 2 have not been ported
  to Zarr-Python 3 yet:

  * Structured arrays / dtypes (:issue:`2134`)
  * Fixed-length string dtypes (:issue:`2347`)
  * Datetime and timedelta dtypes (:issue:`2616`)
  * Object dtypes (:issue:`2617`)
  * Ragged arrays (:issue:`2618`)
  * Groups and Arrays do not implement ``__enter__`` and ``__exit__`` protocols (:issue:`2619`)
  * Big Endian dtypes (:issue:`2324`)
  * Default filters for object dtypes for Zarr format 2 arrays (:issue:`2627`)
