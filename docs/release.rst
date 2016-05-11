Release notes
=============

.. _release_1.0.0:

1.0.0
-----

This release includes a complete re-organization of the code base. The
major version number has been bumped to indicate that there have been
backwards-incompatible changes to the API and the on-disk storage
format. However, Zarr is still in an early stage of development, so
please do not take the version number as an indicator of maturity.

Storage
~~~~~~~

The main motivation for re-organizing the code was to create an
abstraction layer between the core array logic and data storage (`#21
<https://github.com/alimanfoo/zarr/issues/21>`_). In this release, any
object that implements the ``MutableMapping`` interface can be used as
an array store. See the tutorial sections on :ref:`tutorial_persist`
and :ref:`tutorial_tips_storage`, the :ref:`spec_v1`, and the
:mod:`zarr.storage` module documentation for more information.

Please note also that the file organization and file name conventions
used when storing a Zarr array in a directory on the file system have
changed. Persistent Zarr arrays created using previous versions of the
software will not be compatible with this version. See the
:mod:`zarr.storage` API docs and the :ref:`spec_v1` for more
information.

Compression
~~~~~~~~~~~

An abstraction layer has also been created between the core array
logic and the code for compressing and decompressing array
chunks. This release still bundles the c-blosc library and uses Blosc
as the default compressor, however other compressors including zlib,
BZ2 and LZMA are also now supported via the Python standard
library. New compressors can also be dynamically registered for use
with Zarr. See the tutorial sections on :ref:`tutorial_compress` and
:ref:`tutorial_tips_blosc`, the :ref:`spec_v1`, and the
:mod:`zarr.compressors` module documentation for more information.

Synchronization
~~~~~~~~~~~~~~~

The synchronization code has also been refactored to create a layer of
abstraction, enabling Zarr arrays to be used in parallel computations
with a number of alternative synchronization methods. For more
information see the tutorial section on :ref:`tutorial_sync` and the
:mod:`zarr.sync` module documentation.

Changes to the Blosc extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy is no longer a build dependency for the :mod:`zarr.blosc` Cython
extension, so setup.py will run even if NumPy is not already
installed, and should automatically install NumPy as a runtime
dependency. Manual installation of NumPy prior to installing Zarr is
still recommended, however, as the automatic installation of NumPy may
fail or be sub-optimal on some platforms.

The :mod:`zarr.blosc` Cython extension is now optional and compilation
will only be attempted on posix systems; other systems will fall back
to a pure-Python installation (`#25
<https://github.com/alimanfoo/zarr/issues/25>`_). On these systems
only 'zlib', 'bz2' and 'lzma' compression will be available.

Some optimizations have been made within the :mod:`zarr.blosc`
extension to avoid unnecessary memory copies, giving a ~10-20%
performance improvement for multi-threaded compression operations.

The :mod:`zarr.blosc` extension now automatically detects whether it
is running within a single-threaded or multi-threaded program and
adapts its internal behaviour accordingly (`#27
<https://github.com/alimanfoo/zarr/issues/27>`_). There is no need for
the user to make any API calls to switch Blosc between contextual and
non-contextual (global lock) mode. See also the tutorial section on
:ref:`tutorial_tips_blosc`.

Other changes
~~~~~~~~~~~~~

The internal code for managing chunks has been rewritten to be more
efficient. Now no state is maintained for chunks outside of the array
store, meaning that chunks do not carry any extra memory overhead not
accounted for by the store. This negates the need for the "lazy"
option present in the previous release, and this has been removed.

The memory layout within chunks can now be set as either "C"
(row-major) or "F" (column-major), which can help to provide better
compression for some data (`#7
<https://github.com/alimanfoo/zarr/issues/7>`_). See the tutorial
section on :ref:`tutorial_tips_order` for more information.

A bug has been fixed within the ``__getitem__`` and ``__setitem__``
machinery for slicing arrays, to properly handle getting and setting
partial slices.

.. _release_0.4.0:

0.4.0
-----

See `v0.4.0 release notes on GitHub
<https://github.com/alimanfoo/zarr/releases/tag/v0.4.0>`_.

.. _release_0.3.0:

0.3.0
-----

See `v0.3.0 release notes on GitHub
<https://github.com/alimanfoo/zarr/releases/tag/v0.3.0>`_.

