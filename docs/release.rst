Release notes
=============

.. _release_2.3.1:

2.3.1
-----

Bug fixes
~~~~~~~~~

* Makes ``azure-storage-blob`` optional for testing.
  By :user:`John Kirkham <jakirkham>`; :issue:`419`, :issue:`420`


.. _release_2.3.0:

2.3.0
-----

Enhancements
~~~~~~~~~~~~

* New storage backend, backed by Azure Blob Storage, class :class:`zarr.storage.ABSStore`.
  All data is stored as block blobs. By :user:`Shikhar Goenka <shikarsg>`, 
  :user:`Tim Crone <tjcrone>` and :user:`Zain Patel <mzjp2>`; :issue:`345`.

* Add "consolidated" metadata as an experimental feature: use
  :func:`zarr.convenience.consolidate_metadata` to copy all metadata from the various
  metadata keys within a dataset hierarchy under a single key, and
  :func:`zarr.convenience.open_consolidated` to use this single key. This can greatly
  cut down the number of calls to the storage backend, and so remove a lot of overhead
  for reading remote data.
  By :user:`Martin Durant <martindurant>`, :user:`Alistair Miles <alimanfoo>`,
  :user:`Ryan Abernathey <rabernat>`, :issue:`268`, :issue:`332`, :issue:`338`.

* Support has been added for structured arrays with sub-array shape and/or nested fields. By
  :user:`Tarik Onalan <onalant>`, :issue:`111`, :issue:`296`.

* Adds the SQLite-backed :class:`zarr.storage.SQLiteStore` class enabling an
  SQLite database to be used as the backing store for an array or group.
  By :user:`John Kirkham <jakirkham>`, :issue:`368`, :issue:`365`.

* Efficient iteration over arrays by decompressing chunkwise.
  By :user:`Jerome Kelleher <jeromekelleher>`, :issue:`398`, :issue:`399`.

* Adds the Redis-backed :class:`zarr.storage.RedisStore` class enabling a
  Redis database to be used as the backing store for an array or group.
  By :user:`Joe Hamman <jhamman>`, :issue:`299`, :issue:`372`.

* Adds the MongoDB-backed :class:`zarr.storage.MongoDBStore` class enabling a
  MongoDB database to be used as the backing store for an array or group.
  By :user:`Noah D Brenowitz <nbren12>`, :user:`Joe Hamman <jhamman>`,
  :issue:`299`, :issue:`372`, :issue:`401`.

* **New storage class for N5 containers**. The :class:`zarr.n5.N5Store` has been
  added, which uses :class:`zarr.storage.NestedDirectoryStore` to support
  reading and writing from and to N5 containers.
  By :user:`Jan Funke <funkey>` and :user:`John Kirkham <jakirkham>`

Bug fixes
~~~~~~~~~

* The implementation of the :class:`zarr.storage.DirectoryStore` class has been modified to
  ensure that writes are atomic and there are no race conditions where a chunk might appear
  transiently missing during a write operation. By :user:`sbalmer <sbalmer>`, :issue:`327`,
  :issue:`263`.

* Avoid raising in :class:`zarr.storage.DirectoryStore`'s ``__setitem__`` when file already exists.
  By :user:`Justin Swaney <jmswaney>`, :issue:`272`, :issue:`318`

* The required version of the `numcodecs <http://numcodecs.rtfd.io>`_ package has been upgraded
  to 0.6.2, which has enabled some code simplification and fixes a failing test involving
  msgpack encoding. By :user:`John Kirkham <jakirkham>`, :issue:`361`, :issue:`360`, :issue:`352`,
  :issue:`355`, :issue:`324`.

* Failing tests related to pickling/unpickling have been fixed. By :user:`Ryan Williams <ryan-williams>`,
  :issue:`273`, :issue:`308`.

* Corrects handling of ``NaT`` in ``datetime64`` and ``timedelta64`` in various
  compressors (by :user:`John Kirkham <jakirkham>`; :issue:`344`).

* Ensure ``DictStore`` contains only ``bytes`` to facilitate comparisons and protect against writes.
  By :user:`John Kirkham <jakirkham>`, :issue:`350`

* Test and fix an issue (w.r.t. fill values) when storing complex data to ``Array``.
  By :user:`John Kirkham <jakirkham>`, :issue:`363`

* Always use a ``tuple`` when indexing a NumPy ``ndarray``.
  By :user:`John Kirkham <jakirkham>`, :issue:`376`

* Ensure when ``Array`` uses a ``dict``-based chunk store that it only contains
  ``bytes`` to facilitate comparisons and protect against writes. Drop the copy
  for the no filter/compressor case as this handles that case.
  By :user:`John Kirkham <jakirkham>`, :issue:`359`

Maintenance
~~~~~~~~~~~

* Simplify directory creation and removal in ``DirectoryStore.rename``.
  By :user:`John Kirkham <jakirkham>`, :issue:`249`

* CI and test environments have been upgraded to include Python 3.7, drop Python 3.4, and
  upgrade all pinned package requirements. :user:`Alistair Miles <alimanfoo>`, :issue:`308`.

* Start using pyup.io to maintain dependencies.
  :user:`Alistair Miles <alimanfoo>`, :issue:`326`.

* Configure flake8 line limit generally.
  :user:`John Kirkham <jakirkham>`, :issue:`335`.

* Add missing coverage pragmas.
  :user:`John Kirkham <jakirkham>`, :issue:`343`, :issue:`355`.

* Fix missing backslash in docs.
  :user:`John Kirkham <jakirkham>`, :issue:`254`, :issue:`353`.

* Include tests for stores' ``popitem`` and ``pop`` methods.
  By :user:`John Kirkham <jakirkham>`, :issue:`378`, :issue:`380`.

* Include tests for different compressors, endianness, and attributes.
  By :user:`John Kirkham <jakirkham>`, :issue:`378`, :issue:`380`.

* Test validity of stores' contents.
  By :user:`John Kirkham <jakirkham>`, :issue:`359`, :issue:`408`.

Acknowledgments
~~~~~~~~~~~~~~~

.. _release_2.2.0:

2.2.0
-----

Enhancements
~~~~~~~~~~~~

* **Advanced indexing**. The ``Array`` class has several new methods and
  properties that enable a selection of items in an array to be retrieved or
  updated. See the :ref:`tutorial_indexing` tutorial section for more
  information. There is also a `notebook
  <https://github.com/zarr-developers/zarr/blob/master/notebooks/advanced_indexing.ipynb>`_
  with extended examples and performance benchmarks. :issue:`78`, :issue:`89`,
  :issue:`112`, :issue:`172`.

* **New package for compressor and filter codecs**. The classes previously
  defined in the :mod:`zarr.codecs` module have been factored out into a
  separate package called NumCodecs_. The NumCodecs_ package also includes
  several new codec classes not previously available in Zarr, including
  compressor codecs for Zstd and LZ4. This change is backwards-compatible with
  existing code, as all codec classes defined by NumCodecs are imported into the
  :mod:`zarr.codecs` namespace. However, it is recommended to import codecs from
  the new package, see the tutorial sections on :ref:`tutorial_compress` and
  :ref:`tutorial_filters` for examples. With contributions by
  :user:`John Kirkham <jakirkham>`; :issue:`74`, :issue:`102`, :issue:`120`,
  :issue:`123`, :issue:`139`.

* **New storage class for DBM-style databases**. The
  :class:`zarr.storage.DBMStore` class enables any DBM-style database such as gdbm,
  ndbm or Berkeley DB, to be used as the backing store for an array or group. See the
  tutorial section on :ref:`tutorial_storage` for some examples. :issue:`133`,
  :issue:`186`.

* **New storage class for LMDB databases**. The :class:`zarr.storage.LMDBStore` class
  enables an LMDB "Lightning" database to be used as the backing store for an array or
  group. :issue:`192`.

* **New storage class using a nested directory structure for chunk files**. The
  :class:`zarr.storage.NestedDirectoryStore` has been added, which is similar to
  the existing :class:`zarr.storage.DirectoryStore` class but nests chunk files
  for multidimensional arrays into sub-directories. :issue:`155`, :issue:`177`.

* **New tree() method for printing hierarchies**. The ``Group`` class has a new
  :func:`zarr.hierarchy.Group.tree` method which enables a tree representation of
  a group hierarchy to be printed. Also provides an interactive tree
  representation when used within a Jupyter notebook. See the
  :ref:`tutorial_diagnostics` tutorial section for examples. By
  :user:`John Kirkham <jakirkham>`; :issue:`82`, :issue:`140`, :issue:`184`.

* **Visitor API**. The ``Group`` class now implements the h5py visitor API, see
  docs for the :func:`zarr.hierarchy.Group.visit`,
  :func:`zarr.hierarchy.Group.visititems` and
  :func:`zarr.hierarchy.Group.visitvalues` methods. By
  :user:`John Kirkham <jakirkham>`, :issue:`92`, :issue:`122`.

* **Viewing an array as a different dtype**. The ``Array`` class has a new
  :func:`zarr.core.Array.astype` method, which is a convenience that enables an
  array to be viewed as a different dtype. By :user:`John Kirkham <jakirkham>`,
  :issue:`94`, :issue:`96`.

* **New open(), save(), load() convenience functions**. The function
  :func:`zarr.convenience.open` provides a convenient way to open a persistent
  array or group, using either a ``DirectoryStore`` or ``ZipStore`` as the backing
  store. The functions :func:`zarr.convenience.save` and
  :func:`zarr.convenience.load` are also available and provide a convenient way to
  save an entire NumPy array to disk and load back into memory later. See the
  tutorial section :ref:`tutorial_persist` for examples. :issue:`104`,
  :issue:`105`, :issue:`141`, :issue:`181`.

* **IPython completions**. The ``Group`` class now implements ``__dir__()`` and
  ``_ipython_key_completions_()`` which enables tab-completion for group members
  to be used in any IPython interactive environment. :issue:`170`.

* **New info property; changes to __repr__**. The ``Group`` and
  ``Array`` classes have a new ``info`` property which can be used to print
  diagnostic information, including compression ratio where available. See the
  tutorial section on :ref:`tutorial_diagnostics` for examples. The string
  representation (``__repr__``) of these classes has been simplified to ensure
  it is cheap and quick to compute in all circumstances. :issue:`83`,
  :issue:`115`, :issue:`132`, :issue:`148`.

* **Chunk options**. When creating an array, ``chunks=False`` can be specified,
  which will result in an array with a single chunk only. Alternatively,
  ``chunks=True`` will trigger an automatic chunk shape guess. See
  :ref:`tutorial_chunks` for more on the ``chunks`` parameter. :issue:`106`,
  :issue:`107`, :issue:`183`.

* **Zero-dimensional arrays** and are now supported; by
  :user:`Prakhar Goel <newt0311>`, :issue:`154`, :issue:`161`.

* **Arrays with one or more zero-length dimensions** are now fully supported; by
  :user:`Prakhar Goel <newt0311>`, :issue:`150`, :issue:`154`, :issue:`160`.

* **The .zattrs key is now optional** and will now only be created when the first
  custom attribute is set; :issue:`121`, :issue:`200`.

* **New Group.move() method** supports moving a sub-group or array to a different
  location within the same hierarchy. By :user:`John Kirkham <jakirkham>`,
  :issue:`191`, :issue:`193`, :issue:`196`.

* **ZipStore is now thread-safe**; :issue:`194`, :issue:`192`.

* **New Array.hexdigest() method** computes an ``Array``'s hash with ``hashlib``.
  By :user:`John Kirkham <jakirkham>`, :issue:`98`, :issue:`203`.

* **Improved support for object arrays**. In previous versions of Zarr,
  creating an array with ``dtype=object`` was possible but could under certain
  circumstances lead to unexpected errors and/or segmentation faults. To make it easier
  to properly configure an object array, a new ``object_codec`` parameter has been
  added to array creation functions. See the tutorial section on :ref:`tutorial_objects`
  for more information and examples. Also, runtime checks have been added in both Zarr
  and Numcodecs so that segmentation faults are no longer possible, even with a badly
  configured array. This API change is backwards compatible and previous code that created
  an object array and provided an object codec via the ``filters`` parameter will
  continue to work, however a warning will be raised to encourage use of the
  ``object_codec`` parameter. :issue:`208`, :issue:`212`.

* **Added support for datetime64 and timedelta64 data types**;
  :issue:`85`, :issue:`215`.

* **Array and group attributes are now cached by default** to improve performance with
  slow stores, e.g., stores accessing data via the network; :issue:`220`, :issue:`218`,
  :issue:`204`.

* **New LRUStoreCache class**. The class :class:`zarr.storage.LRUStoreCache` has been
  added and provides a means to locally cache data in memory from a store that may be
  slow, e.g., a store that retrieves data from a remote server via the network;
  :issue:`223`.

* **New copy functions**. The new functions :func:`zarr.convenience.copy` and
  :func:`zarr.convenience.copy_all` provide a way to copy groups and/or arrays
  between HDF5 and Zarr, or between two Zarr groups. The
  :func:`zarr.convenience.copy_store` provides a more efficient way to copy
  data directly between two Zarr stores. :issue:`87`, :issue:`113`,
  :issue:`137`, :issue:`217`.

Bug fixes
~~~~~~~~~

* Fixed bug where ``read_only`` keyword argument was ignored when creating an
  array; :issue:`151`, :issue:`179`.

* Fixed bugs when using a ``ZipStore`` opened in 'w' mode; :issue:`158`,
  :issue:`182`.

* Fill values can now be provided for fixed-length string arrays; :issue:`165`,
  :issue:`176`.

* Fixed a bug where the number of chunks initialized could be counted
  incorrectly; :issue:`97`, :issue:`174`.

* Fixed a bug related to the use of an ellipsis (...) in indexing statements;
  :issue:`93`, :issue:`168`, :issue:`172`.

* Fixed a bug preventing use of other integer types for indexing; :issue:`143`,
  :issue:`147`.

Documentation
~~~~~~~~~~~~~

* Some changes have been made to the :ref:`spec_v2` document to clarify
  ambiguities and add some missing information. These changes do not break compatibility
  with any of the material as previously implemented, and so the changes have been made
  in-place in the document without incrementing the document version number. See the
  section on :ref:`spec_v2_changes` in the specification document for more information.
* A new :ref:`tutorial_indexing` section has been added to the tutorial.
* A new :ref:`tutorial_strings` section has been added to the tutorial
  (:issue:`135`, :issue:`175`).
* The :ref:`tutorial_chunks` tutorial section has been reorganised and updated.
* The :ref:`tutorial_persist` and :ref:`tutorial_storage` tutorial sections have
  been updated with new examples (:issue:`100`, :issue:`101`, :issue:`103`).
* A new tutorial section on :ref:`tutorial_pickle` has been added (:issue:`91`).
* A new tutorial section on :ref:`tutorial_datetime` has been added.
* A new tutorial section on :ref:`tutorial_diagnostics` has been added.
* The tutorial sections on :ref:`tutorial_sync` and :ref:`tutorial_tips_blosc` have been
  updated to provide information about how to avoid program hangs when using the Blosc
  compressor with multiple processes (:issue:`199`, :issue:`201`).

Maintenance
~~~~~~~~~~~

* A data fixture has been included in the test suite to ensure data format
  compatibility is maintained; :issue:`83`, :issue:`146`.
* The test suite has been migrated from nosetests to pytest; :issue:`189`, :issue:`225`.
* Various continuous integration updates and improvements; :issue:`118`, :issue:`124`,
  :issue:`125`, :issue:`126`, :issue:`109`, :issue:`114`, :issue:`171`.
* Bump numcodecs dependency to 0.5.3, completely remove nose dependency, :issue:`237`.
* Fix compatibility issues with NumPy 1.14 regarding fill values for structured arrays,
  :issue:`222`, :issue:`238`, :issue:`239`.

Acknowledgments
~~~~~~~~~~~~~~~

Code was contributed to this release by :user:`Alistair Miles <alimanfoo>`, :user:`John
Kirkham <jakirkham>` and :user:`Prakhar Goel <newt0311>`.

Documentation was contributed to this release by :user:`Mamy Ratsimbazafy <mratsim>`
and :user:`Charles Noyes <CSNoyes>`.

Thank you to :user:`John Kirkham <jakirkham>`, :user:`Stephan Hoyer <shoyer>`,
:user:`Francesc Alted <FrancescAlted>`, and :user:`Matthew Rocklin <mrocklin>` for code
reviews and/or comments on pull requests.

.. _release_2.1.4:

2.1.4
-----

* Resolved an issue where calling ``hasattr`` on a ``Group`` object erroneously
  returned a ``KeyError``. By :user:`Vincent Schut <vincentschut>`; :issue:`88`,
  :issue:`95`.

.. _release_2.1.3:

2.1.3
-----

* Resolved an issue with :func:`zarr.creation.array` where dtype was given as
  None (:issue:`80`).

.. _release_2.1.2:

2.1.2
-----

* Resolved an issue when no compression is used and chunks are stored in memory
  (:issue:`79`).

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
  (:issue:`65`).
* Added :class:`zarr.storage.TempStore` class for convenience to provide
  storage via a temporary directory
  (:issue:`59`).
* Fixed performance issues with :class:`zarr.storage.ZipStore` class
  (:issue:`66`).
* The Blosc extension has been modified to return bytes instead of array
  objects from compress and decompress function calls. This should
  improve compatibility and also provides a small performance increase for
  compressing high compression ratio data
  (:issue:`55`).
* Added ``overwrite`` keyword argument to array and group creation methods
  on the :class:`zarr.hierarchy.Group` class
  (:issue:`71`).
* Added ``cache_metadata`` keyword argument to array creation methods.
* The functions :func:`zarr.creation.open_array` and
  :func:`zarr.hierarchy.open_group` now accept any store as first argument
  (:issue:`56`).

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

Thanks to :user:`Matthew Rocklin <mrocklin>`, :user:`Stephan Hoyer <shoyer>` and
:user:`Francesc Alted <FrancescAlted>` for contributions and comments.

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
abstraction layer between the core array logic and data storage (:issue:`21`).
In this release, any
object that implements the ``MutableMapping`` interface can be used as
an array store. See the tutorial sections on :ref:`tutorial_persist`
and :ref:`tutorial_storage`, the :ref:`spec_v1`, and the
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
adapts its internal behaviour accordingly (:issue:`27`). There is no need for
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
compression for some data (:issue:`7`). See the tutorial
section on :ref:`tutorial_chunks_order` for more information.

A bug has been fixed within the ``__getitem__`` and ``__setitem__``
machinery for slicing arrays, to properly handle getting and setting
partial slices.

Acknowledgments
~~~~~~~~~~~~~~~

Thanks to :user:`Matthew Rocklin <mrocklin>`, :user:`Stephan Hoyer <shoyer>`,
:user:`Francesc Alted <FrancescAlted>`, :user:`Anthony Scopatz <scopatz>` and
:user:`Martin Durant <martindurant>` for contributions and comments.

.. _release_0.4.0:

0.4.0
-----

See `v0.4.0 release notes on GitHub
<https://github.com/zarr-developers/zarr/releases/tag/v0.4.0>`_.

.. _release_0.3.0:

0.3.0
-----

See `v0.3.0 release notes on GitHub
<https://github.com/zarr-developers/zarr/releases/tag/v0.3.0>`_.

.. _NumCodecs: http://numcodecs.readthedocs.io/
