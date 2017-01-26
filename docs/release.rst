Release notes
=============

.. _release_2.1.4:

2.1.4
-----

Resolved an issue where calling ``hasattr`` on a ``Group`` object erroneously returned a
``KeyError`` (`#88 <https://github.com/alimanfoo/zarr/issues/88>`_,
`#95 <https://github.com/alimanfoo/zarr/issues/95>`_,
`Vincent Schut <https://github.com/vincentschut>`_)

.. _release_2.1.3:

2.1.3
-----

Resolved an issue with :func:`zarr.creation.array` where dtype was given as
None (`#80 <https://github.com/alimanfoo/zarr/issues/80>`_).

.. _release_2.1.2:

2.1.2
-----

Resolved an issue when no compression is used and chunks are stored in memory
(`#79 <https://github.com/alimanfoo/zarr/issues/79>`_).

.. _release_2.1.1:

2.1.1
-----

Various minor improvements, including: ``Group`` objects support member access
via dot notation (``__getattr__``); fixed metadata caching for ``Array.shape``
property and derivatives; added ``Array.ndim`` property; fixed
``Array.__array__`` method arguments; fixed bug in pickling ``Array`` state;
fixed bug in pickling ``ThreadSynchronizer``.

.. _release_2.1.0:

2.1.0
-----

* Group objects now support member deletion via ``del`` statement
  (`#65 <https://github.com/alimanfoo/zarr/issues/65>`_).
* Added :class:`zarr.storage.TempStore` class for convenience to provide
  storage via a temporary directory
  (`#59 <https://github.com/alimanfoo/zarr/issues/59>`_).
* Fixed performance issues with :class:`zarr.storage.ZipStore` class
  (`#66 <https://github.com/alimanfoo/zarr/issues/27>`_).
* The Blosc extension has been modified to return bytes instead of array
  objects from compress and decompress function calls. This should
  improve compatibility and also provides a small performance increase for
  compressing high compression ratio data
  (`#55 <https://github.com/alimanfoo/zarr/issues/55>`_).
* Added ``overwrite`` keyword argument to array and group creation methods
  on the :class:`zarr.hierarchy.Group` class
  (`#71 <https://github.com/alimanfoo/zarr/issues/71>`_).
* Added ``cache_metadata`` keyword argument to array creation methods.
* The functions :func:`zarr.creation.open_array` and
  :func:`zarr.hierarchy.open_group` now accept any store as first argument
  (`#56 <https://github.com/alimanfoo/zarr/issues/56>`_).

.. _release_2.0.1:

2.0.1
-----

The bundled Blosc library has been upgraded to version 1.11.1.

.. _release_2.0.0:

2.0.0
-----

Hierarchies
~~~~~~~~~~~

Support has been added for organizing arrays into hierarchies via groups. See
the tutorial section on :ref:`tutorial_groups` and the :mod:`zarr.hierarchy`
API docs for more information.

Filters
~~~~~~~

Support has been added for configuring filters to preprocess chunk data prior
to compression. See the tutorial section on :ref:`tutorial_filters` and the
:mod:`zarr.codecs` API docs for more information.

Other changes
~~~~~~~~~~~~~

To accommodate support for hierarchies and filters, the Zarr metadata format
has been modified. See the :ref:`spec_v2` for more information. To migrate an
array stored using Zarr version 1.x, use the :func:`zarr.storage.migrate_1to2`
function.

The bundled Blosc library has been upgraded to version 1.11.0.

Acknowledgments
~~~~~~~~~~~~~~~

Thanks to Matthew Rocklin (mrocklin_), Stephan Hoyer (shoyer_) and
Francesc Alted (FrancescAlted_) for contributions and comments.

.. _release_1.1.0:

1.1.0
-----

* The bundled Blosc library has been upgraded to version 1.10.0. The 'zstd'
  internal compression library is now available within Blosc. See the tutorial
  section on :ref:`tutorial_compress` for an example.
* When using the Blosc compressor, the default internal compression library
  is now 'lz4'.
* The default number of internal threads for the Blosc compressor has been
  increased to a maximum of 8 (previously 4).
* Added convenience functions :func:`zarr.blosc.list_compressors` and
  :func:`zarr.blosc.get_nthreads`.

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

Acknowledgments
~~~~~~~~~~~~~~~

Thanks to Matthew Rocklin (mrocklin_), Stephan Hoyer (shoyer_),
Francesc Alted (FrancescAlted_), Anthony Scopatz (scopatz_) and Martin
Durant (martindurant_) for contributions and comments.

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

.. _mrocklin: https://github.com/mrocklin
.. _shoyer: https://github.com/shoyer
.. _scopatz: https://github.com/scopatz
.. _martindurant: https://github.com/martindurant
.. _FrancescAlted: https://github.com/FrancescAlted
