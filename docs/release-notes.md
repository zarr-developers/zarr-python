# Release notes

## 3.0.8 (2025-05-19)

> **Warning**
> In versions 3.0.0 to 3.0.7 opening arrays or groups with `mode='a'` (the default for many builtin functions)
> would cause any existing paths in the store to be deleted. This is fixed in 3.0.8, and
> we recommend all users upgrade to avoid this bug that could cause unintentional data loss.

### Features

- Added a `print_debug_info` function for bug reports. ([#2913](https://github.com/zarr-developers/zarr-python/issues/2913))

### Bugfixes

- Fix a bug that prevented the number of initialized chunks being counted properly. ([#2862](https://github.com/zarr-developers/zarr-python/issues/2862))
- Fixed sharding with GPU buffers. ([#2978](https://github.com/zarr-developers/zarr-python/issues/2978))
- Fix structured `dtype` fill value serialization for consolidated metadata ([#2998](https://github.com/zarr-developers/zarr-python/issues/2998))
- It is now possible to specify no compressor when creating a zarr format 2 array.
  This can be done by passing `compressor=None` to the various array creation routines.

  The default behaviour of automatically choosing a suitable default compressor remains if the compressor argument is not given.
  To reproduce the behaviour in previous zarr-python versions when `compressor=None` was passed, pass `compressor='auto'` instead. ([#3039](https://github.com/zarr-developers/zarr-python/issues/3039))
- Fixed the typing of `dimension_names` arguments throughout so that it now accepts iterables that contain `None` alongside `str`. ([#3045](https://github.com/zarr-developers/zarr-python/issues/3045))
- Using various functions to open data with `mode='a'` no longer deletes existing data in the store. ([#3062](https://github.com/zarr-developers/zarr-python/issues/3062))
- Internally use `typesize` constructor parameter for `numcodecs.blosc.Blosc` to improve compression ratios back to the v2-package levels. ([#2962](https://github.com/zarr-developers/zarr-python/issues/2962))
- Specifying the memory order of Zarr format 2 arrays using the `order` keyword argument has been fixed. ([#2950](https://github.com/zarr-developers/zarr-python/issues/2950))

### Misc

- [#2972](https://github.com/zarr-developers/zarr-python/issues/2972), [#3027](https://github.com/zarr-developers/zarr-python/issues/3027), [#3049](https://github.com/zarr-developers/zarr-python/issues/3049)

## 3.0.7 (2025-04-22)

### Features

- Add experimental ObjectStore storage class based on obstore. ([#1661](https://github.com/zarr-developers/zarr-python/issues/1661))
- Add `zarr.from_array` using concurrent streaming of source data ([#2622](https://github.com/zarr-developers/zarr-python/issues/2622))

### Bugfixes

- 0-dimensional arrays are now returning a scalar. Therefore, the return type of `__getitem__` changed
  to NDArrayLikeOrScalar. This change is to make the behavior of 0-dimensional arrays consistent with
  `numpy` scalars. ([#2718](https://github.com/zarr-developers/zarr-python/issues/2718))
- Fix `fill_value` serialization for `NaN` in `ArrayV2Metadata` and add property-based testing of round-trip serialization ([#2802](https://github.com/zarr-developers/zarr-python/issues/2802))
- Fixes `ConsolidatedMetadata` serialization of `nan`, `inf`, and `-inf` to be
  consistent with the behavior of `ArrayMetadata`. ([#2996](https://github.com/zarr-developers/zarr-python/issues/2996))

### Improved Documentation

- Updated the 3.0 migration guide to include the removal of "." syntax for getting group members. ([#2991](https://github.com/zarr-developers/zarr-python/issues/2991), [#2997](https://github.com/zarr-developers/zarr-python/issues/2997))

### Misc

- Define a new versioning policy based on Effective Effort Versioning. This replaces the old Semantic
  Versioning-based policy. ([#2924](https://github.com/zarr-developers/zarr-python/issues/2924), [#2910](https://github.com/zarr-developers/zarr-python/issues/2910))
- Make warning filters in the tests more specific, so warnings emitted by tests added in the future
  are more likely to be caught instead of ignored. ([#2714](https://github.com/zarr-developers/zarr-python/issues/2714))
- Avoid an unnecessary memory copy when writing Zarr to a local file ([#2944](https://github.com/zarr-developers/zarr-python/issues/2944))

## 3.0.6 (2025-03-20)

### Bugfixes

- Restore functionality of `del z.attrs['key']` to actually delete the key. ([#2908](https://github.com/zarr-developers/zarr-python/issues/2908))

## 3.0.5 (2025-03-07)

### Bugfixes

- Fixed a bug where `StorePath` creation would not apply standard path normalization to the `path` parameter,
  which led to the creation of arrays and groups with invalid keys. ([#2850](https://github.com/zarr-developers/zarr-python/issues/2850))
- Prevent update_attributes calls from deleting old attributes ([#2870](https://github.com/zarr-developers/zarr-python/issues/2870))

### Misc

- [#2796](https://github.com/zarr-developers/zarr-python/issues/2796)

## 3.0.4 (2025-02-23)

### Features

- Adds functions for concurrently creating multiple arrays and groups. ([#2665](https://github.com/zarr-developers/zarr-python/issues/2665))

### Bugfixes

- Fixed a bug where `ArrayV2Metadata` could save `filters` as an empty array. ([#2847](https://github.com/zarr-developers/zarr-python/issues/2847))
- Fix a bug when setting values of a smaller last chunk. ([#2851](https://github.com/zarr-developers/zarr-python/issues/2851))

### Misc

- [#2828](https://github.com/zarr-developers/zarr-python/issues/2828)

## 3.0.3 (2025-02-14)

### Features

- Improves performance of FsspecStore.delete_dir for remote filesystems supporting concurrent/batched deletes, e.g., s3fs. ([#2661](https://github.com/zarr-developers/zarr-python/issues/2661))
- Added `zarr.config.enable_gpu` to update Zarr's configuration to use GPUs. ([#2751](https://github.com/zarr-developers/zarr-python/issues/2751))
- Avoid reading chunks during writes where possible. [#757](https://github.com/zarr-developers/zarr-python/issues/757) ([#2784](https://github.com/zarr-developers/zarr-python/issues/2784))
- `LocalStore` learned to `delete_dir`. This makes array and group deletes more efficient. ([#2804](https://github.com/zarr-developers/zarr-python/issues/2804))
- Add `zarr.testing.strategies.array_metadata` to generate ArrayV2Metadata and ArrayV3Metadata instances. ([#2813](https://github.com/zarr-developers/zarr-python/issues/2813))
- Add arbitrary `shards` to Hypothesis strategy for generating arrays. ([#2822](https://github.com/zarr-developers/zarr-python/issues/2822))

### Bugfixes

- Fixed bug with Zarr using device memory, instead of host memory, for storing metadata when using GPUs. ([#2751](https://github.com/zarr-developers/zarr-python/issues/2751))
- The array returned by `zarr.empty` and an empty `zarr.core.buffer.cpu.NDBuffer` will now be filled with the
  specified fill value, or with zeros if no fill value is provided.
  This fixes a bug where Zarr format 2 data with no fill value was written with un-predictable chunk sizes. ([#2755](https://github.com/zarr-developers/zarr-python/issues/2755))
- Fix zip-store path checking for stores with directories listed as files. ([#2758](https://github.com/zarr-developers/zarr-python/issues/2758))
- Use removeprefix rather than replace when removing filename prefixes in `FsspecStore.list` ([#2778](https://github.com/zarr-developers/zarr-python/issues/2778))
- Enable automatic removal of `needs release notes` with labeler action ([#2781](https://github.com/zarr-developers/zarr-python/issues/2781))
- Use the proper label config ([#2785](https://github.com/zarr-developers/zarr-python/issues/2785))
- Alters the behavior of `create_array` to ensure that any groups implied by the array's name are created if they do not already exist. Also simplifies the type signature for any function that takes an ArrayConfig-like object. ([#2795](https://github.com/zarr-developers/zarr-python/issues/2795))
- Enitialise empty chunks to the default fill value during writing and add default fill values for datetime, timedelta, structured, and other (void* fixed size) data types ([#2799](https://github.com/zarr-developers/zarr-python/issues/2799))
- Ensure utf8 compliant strings are used to construct numpy arrays in property-based tests ([#2801](https://github.com/zarr-developers/zarr-python/issues/2801))
- Fix pickling for ZipStore ([#2807](https://github.com/zarr-developers/zarr-python/issues/2807))
- Update numcodecs to not overwrite codec configuration ever. Closes [#2800](https://github.com/zarr-developers/zarr-python/issues/2800). ([#2811](https://github.com/zarr-developers/zarr-python/issues/2811))
- Fix fancy indexing (e.g. arr[5, [0, 1]]) with the sharding codec ([#2817](https://github.com/zarr-developers/zarr-python/issues/2817))

### Improved Documentation

- Added new user guide on GPU. ([#2751](https://github.com/zarr-developers/zarr-python/issues/2751))

## 3.0.2 (2025-01-31)

### Features

- Test `getsize()` and `getsize_prefix()` in `StoreTests`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Test that a `ValueError` is raised for invalid byte range syntax in `StoreTests`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Separate instantiating and opening a store in `StoreTests`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Add a test for using Stores as a context managers in `StoreTests`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Implemented `LogingStore.open()`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- `LoggingStore` is now a generic class. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Change StoreTest's `test_store_repr`, `test_store_supports_writes`,
  `test_store_supports_partial_writes`, and `test_store_supports_listing`
  to to be implemented using `@abstractmethod`, rather raising `NotImplementedError`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Test the error raised for invalid buffer arguments in `StoreTests`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Test that data can be written to a store that's not yet open using the store.set method in `StoreTests`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Adds a new function `init_array` for initializing an array in storage, and refactors `create_array`
  to use `init_array`. `create_array` takes two new parameters: `data`, an optional array-like object, and `write_data`, a bool which defaults to `True`.
  If `data` is given to `create_array`, then the `dtype` and `shape` attributes of `data` are used to define the
  corresponding attributes of the resulting Zarr array. Additionally, if `data` given and `write_data` is `True`,
  then the values in `data` will be written to the newly created array. ([#2761](https://github.com/zarr-developers/zarr-python/issues/2761))

### Bugfixes

- Wrap sync fsspec filesystems with `AsyncFileSystemWrapper`. ([#2533](https://github.com/zarr-developers/zarr-python/issues/2533))
- Added backwards compatibility for Zarr format 2 structured arrays. ([#2681](https://github.com/zarr-developers/zarr-python/issues/2681))
- Update equality for `LoggingStore` and `WrapperStore` such that 'other' must also be a `LoggingStore` or `WrapperStore` respectively, rather than only checking the types of the stores they wrap. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Ensure that `ZipStore` is open before getting or setting any values. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Use stdout rather than stderr as the default stream for `LoggingStore`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Match the errors raised by read only stores in `StoreTests`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Fixed `ZipStore` to make sure the correct attributes are saved when instances are pickled.
  This fixes a previous bug that prevent using `ZipStore` with a `ProcessPoolExecutor`. ([#2762](https://github.com/zarr-developers/zarr-python/issues/2762))
- Updated the optional test dependencies to include `botocore` and `fsspec`. ([#2768](https://github.com/zarr-developers/zarr-python/issues/2768))
- Fixed the fsspec tests to skip if `botocore` is not installed.
  Previously they would have failed with an import error. ([#2768](https://github.com/zarr-developers/zarr-python/issues/2768))
- Optimize full chunk writes. ([#2782](https://github.com/zarr-developers/zarr-python/issues/2782))

### Improved Documentation

- Changed the machinery for creating changelog entries.
  Now individual entries should be added as files to the `changes` directory in the `zarr-python` repository, instead of directly to the changelog file. ([#2736](https://github.com/zarr-developers/zarr-python/issues/2736))

### Other

- Created a type alias `ChunkKeyEncodingLike` to model the union of `ChunkKeyEncoding` instances and the dict form of the
  parameters of those instances. `ChunkKeyEncodingLike` should be used by high-level functions to provide a convenient
  way for creating `ChunkKeyEncoding` objects. ([#2763](https://github.com/zarr-developers/zarr-python/issues/2763))

## 3.0.1 (Jan. 17, 2025)

* Implement `zarr.from_array` using concurrent streaming ([#2622](https://github.com/zarr-developers/zarr-python/issues/2622)).

### Bug fixes

* Fixes `order` argument for Zarr format 2 arrays ([#2679](https://github.com/zarr-developers/zarr-python/issues/2679)).
* Fixes a bug that prevented reading Zarr format 2 data with consolidated
  metadata written using `zarr-python` version 2 ([#2694](https://github.com/zarr-developers/zarr-python/issues/2694)).
* Ensure that compressor=None results in no compression when writing Zarr
  format 2 data ([#2708](https://github.com/zarr-developers/zarr-python/issues/2708)).
* Fix for empty consolidated metadata dataset: backwards compatibility with
  Zarr-Python 2 ([#2695](https://github.com/zarr-developers/zarr-python/issues/2695)).

### Documentation

* Add v3.0.0 release announcement banner ([#2677](https://github.com/zarr-developers/zarr-python/issues/2677)).
* Quickstart guide alignment with V3 API ([#2697](https://github.com/zarr-developers/zarr-python/issues/2697)).
* Fix doctest failures related to numcodecs 0.15 ([#2727](https://github.com/zarr-developers/zarr-python/issues/2727)).

### Other

* Removed some unnecessary files from the source distribution
  to reduce its size. ([#2686](https://github.com/zarr-developers/zarr-python/issues/2686)).
* Enable codecov in GitHub actions ([#2682](https://github.com/zarr-developers/zarr-python/issues/2682)).
* Speed up hypothesis tests ([#2650](https://github.com/zarr-developers/zarr-python/issues/2650)).
* Remove multiple imports for an import name ([#2723](https://github.com/zarr-developers/zarr-python/issues/2723)).

## 3.0.0 (Jan. 9, 2025)

3.0.0 is a new major release of Zarr-Python, with many breaking changes.
See the [v3 migration guide](user-guide/v3_migration.md) for a listing of what's changed.

Normal release note service will resume with further releases in the 3.0.0
series.

Release notes for the zarr-python 2.x and 1.x releases can be found here:
https://zarr.readthedocs.io/en/support-v2/release.html
