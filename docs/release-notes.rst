Release notes
=============

.. towncrier release notes start

3.0.6 (2025-03-20)
------------------

Bugfixes
~~~~~~~~

- Restore functionality of `del z.attrs['key']` to actually delete the key. (:issue:`2908`)


3.0.5 (2025-03-07)
------------------

Bugfixes
~~~~~~~~

- Fixed a bug where ``StorePath`` creation would not apply standard path normalization to the ``path`` parameter,
  which led to the creation of arrays and groups with invalid keys. (:issue:`2850`)
- Prevent update_attributes calls from deleting old attributes (:issue:`2870`)


Misc
~~~~

- :issue:`2796`

3.0.4 (2025-02-23)
------------------

Features
~~~~~~~~

- Adds functions for concurrently creating multiple arrays and groups. (:issue:`2665`)

Bugfixes
~~~~~~~~

- Fixed a bug where ``ArrayV2Metadata`` could save ``filters`` as an empty array. (:issue:`2847`)
- Fix a bug when setting values of a smaller last chunk. (:issue:`2851`)

Misc
~~~~

- :issue:`2828`


3.0.3 (2025-02-14)
------------------

Features
~~~~~~~~

- Improves performance of FsspecStore.delete_dir for remote filesystems supporting concurrent/batched deletes, e.g., s3fs. (:issue:`2661`)
- Added :meth:`zarr.config.enable_gpu` to update Zarr's configuration to use GPUs. (:issue:`2751`)
- Avoid reading chunks during writes where possible. :issue:`757` (:issue:`2784`)
- :py:class:`LocalStore` learned to ``delete_dir``. This makes array and group deletes more efficient. (:issue:`2804`)
- Add `zarr.testing.strategies.array_metadata` to generate ArrayV2Metadata and ArrayV3Metadata instances. (:issue:`2813`)
- Add arbitrary `shards` to Hypothesis strategy for generating arrays. (:issue:`2822`)


Bugfixes
~~~~~~~~

- Fixed bug with Zarr using device memory, instead of host memory, for storing metadata when using GPUs. (:issue:`2751`)
- The array returned by ``zarr.empty`` and an empty ``zarr.core.buffer.cpu.NDBuffer`` will now be filled with the
  specified fill value, or with zeros if no fill value is provided.
  This fixes a bug where Zarr format 2 data with no fill value was written with un-predictable chunk sizes. (:issue:`2755`)
- Fix zip-store path checking for stores with directories listed as files. (:issue:`2758`)
- Use removeprefix rather than replace when removing filename prefixes in `FsspecStore.list` (:issue:`2778`)
- Enable automatic removal of `needs release notes` with labeler action (:issue:`2781`)
- Use the proper label config (:issue:`2785`)
- Alters the behavior of ``create_array`` to ensure that any groups implied by the array's name are created if they do not already exist. Also simplifies the type signature for any function that takes an ArrayConfig-like object. (:issue:`2795`)
- Enitialise empty chunks to the default fill value during writing and add default fill values for datetime, timedelta, structured, and other (void* fixed size) data types (:issue:`2799`)
- Ensure utf8 compliant strings are used to construct numpy arrays in property-based tests (:issue:`2801`)
- Fix pickling for ZipStore (:issue:`2807`)
- Update numcodecs to not overwrite codec configuration ever. Closes :issue:`2800`. (:issue:`2811`)
- Fix fancy indexing (e.g. arr[5, [0, 1]]) with the sharding codec (:issue:`2817`)


Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Added new user guide on :ref:`user-guide-gpu`. (:issue:`2751`)


3.0.2 (2025-01-31)
------------------

Features
~~~~~~~~

- Test ``getsize()`` and ``getsize_prefix()`` in ``StoreTests``. (:issue:`2693`)
- Test that a ``ValueError`` is raised for invalid byte range syntax in ``StoreTests``. (:issue:`2693`)
- Separate instantiating and opening a store in ``StoreTests``. (:issue:`2693`)
- Add a test for using Stores as a context managers in ``StoreTests``. (:issue:`2693`)
- Implemented ``LogingStore.open()``. (:issue:`2693`)
- ``LoggingStore`` is now a generic class. (:issue:`2693`)
- Change StoreTest's ``test_store_repr``, ``test_store_supports_writes``,
  ``test_store_supports_partial_writes``, and ``test_store_supports_listing``
  to to be implemented using ``@abstractmethod``, rather raising ``NotImplementedError``. (:issue:`2693`)
- Test the error raised for invalid buffer arguments in ``StoreTests``. (:issue:`2693`)
- Test that data can be written to a store that's not yet open using the store.set method in ``StoreTests``. (:issue:`2693`)
- Adds a new function ``init_array`` for initializing an array in storage, and refactors ``create_array``
  to use ``init_array``. ``create_array`` takes two new parameters: ``data``, an optional array-like object, and ``write_data``, a bool which defaults to ``True``.
  If ``data`` is given to ``create_array``, then the ``dtype`` and ``shape`` attributes of ``data`` are used to define the
  corresponding attributes of the resulting Zarr array. Additionally, if ``data`` given and ``write_data`` is ``True``,
  then the values in ``data`` will be written to the newly created array. (:issue:`2761`)


Bugfixes
~~~~~~~~

- Wrap sync fsspec filesystems with ``AsyncFileSystemWrapper``. (:issue:`2533`)
- Added backwards compatibility for Zarr format 2 structured arrays. (:issue:`2681`)
- Update equality for ``LoggingStore`` and ``WrapperStore`` such that 'other' must also be a ``LoggingStore`` or ``WrapperStore`` respectively, rather than only checking the types of the stores they wrap. (:issue:`2693`)
- Ensure that ``ZipStore`` is open before getting or setting any values. (:issue:`2693`)
- Use stdout rather than stderr as the default stream for ``LoggingStore``. (:issue:`2693`)
- Match the errors raised by read only stores in ``StoreTests``. (:issue:`2693`)
- Fixed ``ZipStore`` to make sure the correct attributes are saved when instances are pickled.
  This fixes a previous bug that prevent using ``ZipStore`` with a ``ProcessPoolExecutor``. (:issue:`2762`)
- Updated the optional test dependencies to include ``botocore`` and ``fsspec``. (:issue:`2768`)
- Fixed the fsspec tests to skip if ``botocore`` is not installed.
  Previously they would have failed with an import error. (:issue:`2768`)
- Optimize full chunk writes. (:issue:`2782`)


Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Changed the machinery for creating changelog entries.
  Now individual entries should be added as files to the `changes` directory in the `zarr-python` repository, instead of directly to the changelog file. (:issue:`2736`)

Other
~~~~~

- Created a type alias ``ChunkKeyEncodingLike`` to model the union of ``ChunkKeyEncoding`` instances and the dict form of the
  parameters of those instances. ``ChunkKeyEncodingLike`` should be used by high-level functions to provide a convenient
  way for creating ``ChunkKeyEncoding`` objects. (:issue:`2763`)


3.0.1 (Jan. 17, 2025)
---------------------

Bug fixes
~~~~~~~~~
* Fixes ``order`` argument for Zarr format 2 arrays (:issue:`2679`).

* Fixes a bug that prevented reading Zarr format 2 data with consolidated
  metadata written using ``zarr-python`` version 2 (:issue:`2694`).

* Ensure that compressor=None results in no compression when writing Zarr
  format 2 data (:issue:`2708`).

* Fix for empty consolidated metadata dataset: backwards compatibility with
  Zarr-Python 2 (:issue:`2695`).

Documentation
~~~~~~~~~~~~~
* Add v3.0.0 release announcement banner (:issue:`2677`).

* Quickstart guide alignment with V3 API (:issue:`2697`).

* Fix doctest failures related to numcodecs 0.15 (:issue:`2727`).

Other
~~~~~
* Removed some unnecessary files from the source distribution
  to reduce its size. (:issue:`2686`).

* Enable codecov in GitHub actions (:issue:`2682`).

* Speed up hypothesis tests (:issue:`2650`).

* Remove multiple imports for an import name (:issue:`2723`).


.. _release_3.0.0:

3.0.0 (Jan. 9, 2025)
--------------------

3.0.0 is a new major release of Zarr-Python, with many breaking changes.
See the :ref:`v3 migration guide` for a listing of what's changed.

Normal release note service will resume with further releases in the 3.0.0
series.

Release notes for the zarr-python 2.x and 1.x releases can be found here:
https://zarr.readthedocs.io/en/support-v2/release.html
