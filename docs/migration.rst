Zarr-Python 3 Migration Guide
=============================

Zarr-Python 3 introduces a number of changes to the API, including a number
of significant breaking changes and pending deprecations.

This page provides a guide highlighting the most notable changes to help you
migrate your code from Zarr-Python 2.x to Zarr-Python 3.x.

Compatibility target
--------------------

Zarr-Python 3 represents a major refactor of the Zarr-Python codebase. Some of the goals motivating this refactor included:

  - adding support for the V3 specification (alongside the V3 specification)
  - cleaning up internal and user facing APIs
  - improving performance (particularly in high latency storage environments like cloud object store)

Though these goals necessitated some breaking changes to the API (hence the major version update), we have tried to maintain
backwards compatibility in the most widely used parts of the API including the `Array` and `Group` classes and the top-level
API (e.g. `zarr.open_array` and `zarr.open_group`). It is worth noting that we significantly evolved the internal data model,
moving away from a model that was tightly coupled to the v2 specification, and to a more generic representation of Zarr objects.

Getting ready for 3.0
---------------------

Ahead of the 3.0 release, we suggest projects that depend on Zarr-Python take the following actions:

1. Pin the supported Zarr-Python version to ``zarr>=2,<3``. This is a best practice and will protect your users from any incompatibilities that may arise during the release of Zarr-Python 3.0.
2. Limit your imports from the Zarr-Python package. Most of the primary API ``zarr.*`` will be compatible in 3.0. However, the following breaking API changes are planned:
   
   - ``numcodecs.*`` will no longer be available in ``zarr.*``. (Suggested action: transition to importing codecs from ``numcodecs`` directly.)
   - The ``zarr.v3_api_available`` feature flag is being removed. (Suggested action: this experimental feature was deprecated in v2.18.)
   - The following internal modules are being removed or significant changed:
   
    - ``zarr.attrs``
    - ``zarr.codecs``
    - ``zarr.context``
    - ``zarr.core``
    - ``zarr.hierarchy``
    - ``zarr.indexing``
    - ``zarr.meta``
    - ``zarr.meta_v1``
    - ``zarr.storage``
    - ``zarr.sync``
    - ``zarr.types``
    - ``zarr.util``
    - ``zarr.n5``

Continue using Zarr-Python 2
----------------------------

Zarr-Python 2.x is still available, though we recommend migrating to Zarr-Python 3 for its improvements and new features.
Security and bug fixes will be made to the 2.x series for at least 6 months following the first Zarr-Python 3 release.

If you need to use the latest Zarr-Python 2 release, you can install it with:

    $ pip install "zarr==2.*"


Migration Guide
---------------

The following sections provide details on the most important changes in Zarr-Python 3.

Changes to the Array class
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Disallow direct construction

Changes to the Group class
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Disallow direct construction

Changes to the Store class
~~~~~~~~~~~~~~~~~~~~~~~~~~

Some of the biggest changes in Zarr-Python 3 are found in the `Store` class. The most notable changes to the Store API are:

1. Dropped the ``MutableMapping`` base class in favor of a custom abstract base class (``zarr.abc.store.Store``).
2. Switched to a primarily Async interface.

TODO

Miscellaneous changes
~~~~~~~~~~~~~~~~~~~~~

TODO
