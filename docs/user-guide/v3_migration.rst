3.0 Migration Guide
===================

Zarr-Python 3.0 introduces a number of changes to the API, including a number
of significant breaking changes and pending deprecations.

This page provides a guide highlighting the most notable changes to help you
migrate your code from version 2 to version 3.

Zarr-Python 3 represents a major refactor of the Zarr-Python codebase. Some of the
goals motivating this refactor included:

* adding support for the Zarr V3 specification (alongside the Zarr V2 specification)
* cleaning up internal and user facing APIs
* improving performance (particularly in high latency storage environments like
  cloud object store)

Compatibility target
--------------------

The goals described above necessitated some breaking changes to the API (hence the
major version update), but we have attempted to maintain ~95% backwards compatibility
in the most widely used parts of the API. This in the :class:`zarr.Array` and
:class:`zarr.Group` classes and the "top-level API" (e.g. :func:`zarr.open_array` and
:func:`zarr.open_group`).

Getting ready for 3.0
---------------------

Ahead of the 3.0 release, we suggest projects that depend on Zarr-Python take the
following actions:

1. Pin the supported Zarr-Python version to ``zarr>=2,<3``. This is a best practice
   and will protect your users from any incompatibilities that may arise during the
   release of Zarr-Python 3.0.
2. Limit your imports from the Zarr-Python package. Most of the primary API ``zarr.*``
   will be compatible in 3.0. However, the following breaking API changes are planned:

   - ``numcodecs.*`` will no longer be available in ``zarr.*``. To migrate, import codecs
     directly from ``numcodecs``:

     .. code-block:: python

        from numcodecs import Blosc
        # instead of:
        # from zarr import Blosc

   - The ``zarr.v3_api_available`` feature flag is being removed. In Zarr-Python 3.0
     the v3 API is always available, so you shouldn't need to use this flag.
   - The following internal modules are being removed or significantly changed. If
     your application relies on imports from any of the below modules, you will need
     to either a) modify your application to no longer rely on these imports or b)
     vendor the parts of the specific modules that you need.

     * ``zarr.attrs``
     * ``zarr.codecs``
     * ``zarr.context``
     * ``zarr.core``
     * ``zarr.hierarchy``
     * ``zarr.indexing``
     * ``zarr.meta``
     * ``zarr.meta_v1``
     * ``zarr.storage``
     * ``zarr.sync``
     * ``zarr.types``
     * ``zarr.util``
     * ``zarr.n5``

3. Test that your package works with v3. You can start testing against version 3 now
   (pre-releases are being published to PyPI weekly).
4. Update the pin to zarr >=3

Continue using Zarr-Python 2
----------------------------

Zarr-Python 2.x is still available, though we recommend migrating to Zarr-Python 3 for
its improvements and new features. Security and bug fixes will be made to the 2.x series
for at least 6 months following the first Zarr-Python 3 release.
If you need to use the latest Zarr-Python 2 release, you can install it with:

.. code-block:: console

    $ pip install "zarr==2.*"

Migration Guide
---------------

The following sections provide details on the most important changes in Zarr-Python 3.

The Array class
~~~~~~~~~~~~~~~

1. Disallow direct construction - use :func:`zarr.open_array` or :func:`zarr.create_array`
   instead of directly constructing the :class:`zarr.Array` class.

2. Defaulting to ``zarr_format=3`` - newly created arrays will use the version 3 of the 
   Zarr specification. To continue using version 2, set ``zarr_format=2`` when creating arrays.

The Group class
~~~~~~~~~~~~~~~

1. Disallow direct construction - use :func:`zarr.open_group` or :func:`zarr.create_group`
   instead of directly constructing the :class:`zarr.Group` class.
2. Deprecated most of the h5py compatibility methods. The following migration is suggested:

   - Use :func:`zarr.Group.create_array` in place of :func:`zarr.Group.create_dataset`
   - Use :func:`zarr.Group.require_array` in place of :func:`zarr.Group.require_dataset`

The Store class
~~~~~~~~~~~~~~~

Some of the biggest changes in Zarr-Python 3 are found in the ``Store`` class. The most notable changes to the Store API are:

1. Replaced the ``MutableMapping`` base class in favor of a custom abstract base class (:class:`zarr.abc.store.Store`).
2. Switched to a primarily Async interface.

Beyond the changes store interface, a number of deprecated stores were also removed in Zarr-Python 3:

- ``N5Store``
- ``DBMStore``
- ``LMDBStore``
- ``SQLiteStore``
- ``MongoDBStore``
- ``RedisStore``
- ``ABSStore``

Dependencies Changes
~~~~~~~~~~~~~~~~~~~~

- The new ``remote`` dependency group can be used to install a supported version of
  ``fsspec``, required for remote data access.
- The new ``gpu`` dependency group can be used to install a supported version of
  ``cuda``, required for GPU functionality.
- The ``jupyter`` optional dependency group has been removed, since v3 contains no
  jupyter specific functionality.

Configuration
~~~~~~~~~~~~~

There is a new configuration system based on `donfig <https://github.com/pytroll/donfig>`_, 
which can be accessed via :data:`zarr.config`. 
Configuration values can be set using code like the following:

.. code-block:: python

   import zarr
   zarr.config.set({"array.order": "F"})

Alternatively, configuration values can be set using environment variables, 
e.g. ``ZARR_ARRAY__ORDER=F``.

Configuration options include the following:

- Default Zarr format ``default_zarr_version``
- Default array order in memory ``array.order``
- Async and threading options, e.g. ``async.concurrency`` and ``threading.max_workers``
- Selections of implementations of codecs, codec pipelines and buffers
- TODO: write_empty_chunks PR 2429
- TODO: default codecs PR 2470

Miscellaneous
~~~~~~~~~~~~~

- The keyword argument ``zarr_version`` has been deprecated in favor of ``zarr_format``.

🚧 Work in Progress 🚧
~~~~~~~~~~~~~~~~~~~~~~

Zarr-Python 3 is still under active development, and is not yet fully complete.
The following list summarizes areas of the codebase that we expect to build out
after the 3.0 release:

- The following functions / methods have not been ported to Zarr-Python 3 yet:

  * :func:`zarr.copy`
  * :func:`zarr.copy_all`
  * :func:`zarr.copy_store`
  * :func:`zarr.Group.move`

- The following options in the top-level API have not been ported to Zarr-Python 3 yet.
  If these options are important to you, please open a
  `GitHub issue <https://github.com/zarr-developers/zarr-python/issues/new>` describing
  your use case.

  * ``cache_attrs``
  * ``cache_metadata``
  * ``chunk_store``
  * ``meta_array``
  * ``object_codec``
  * ``synchronizer``
  * ``dimension_separator``

- The following features have not been ported to Zarr-Python 3 yet:

  * Structured arrays / dtypes
  * Fixed-length strings
  * Object arrays
  * Ragged arrays
  * Datetimes and timedeltas
  * Groups and Arrays do not implement ``__enter__`` and ``__exit__`` protocols

