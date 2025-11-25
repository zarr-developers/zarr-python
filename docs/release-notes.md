# Release notes

<!-- towncrier release notes start -->

# zarr 3.1.5 (2025-11-21)

## Bugfixes

- Fix formatting errors in the release notes section of the docs. ([#3594](https://github.com/zarr-developers/zarr-python/issues/3594))


## 3.1.4 (2025-11-20)

### Features

- The `Array` class can now also be parametrized in the same manner as the `AsyncArray` class, allowing Zarr format v2 and v3 `Array`s to be distinguished.
  New types have been added to `zarr.types` to help with this. ([#3304](https://github.com/zarr-developers/zarr-python/issues/3304))
- Adds `zarr.experimental.cache_store.CacheStore`, a `Store` that implements caching by combining two other `Store` instances. See the [docs page](https://zarr.readthedocs.io/en/latest/user-guide/experimental#cachestore) for more information about this feature. ([#3366](https://github.com/zarr-developers/zarr-python/issues/3366))
- Adds a `zarr.experimental` module for unstable user-facing features. ([#3490](https://github.com/zarr-developers/zarr-python/issues/3490))
- Add a `array.target_shard_size_bytes` to [`zarr.config`][] to allow users to set a maximum number of bytes per-shard when `shards="auto"` in, for example, [`zarr.create_array`][]. ([#3547](https://github.com/zarr-developers/zarr-python/issues/3547))
- Make `async_array` on the [`zarr.Array`][] class public (`_async_array` will remain untouched, but its stability is not guaranteed). ([#3556](https://github.com/zarr-developers/zarr-python/issues/3556))

### Bugfixes

- Fix a bug that prevented `PCodec` from being properly resolved when loading arrays using that compressor. ([#3483](https://github.com/zarr-developers/zarr-python/issues/3483))
- Fixed a bug that prevented Zarr Python from opening Zarr V3 array metadata documents that contained
   extra keys with permissible values (dicts with a `"must_understand"` key set to `"false"`). ([#3530](https://github.com/zarr-developers/zarr-python/issues/3530))
- Fixed a bug where the `"consolidated_metadata"` key was written to metadata documents even when
  consolidated metadata was not used, resulting in invalid metadata documents. ([#3535](https://github.com/zarr-developers/zarr-python/issues/3535))
- Improve write performance to large shards by up to 10x. ([#3560](https://github.com/zarr-developers/zarr-python/issues/3560))

### Improved Documentation

- Use mkdocs-material for Zarr-Python documentation ([#3118](https://github.com/zarr-developers/zarr-python/issues/3118))
- Document different values of StoreLike with examples in the user guide. ([#3303](https://github.com/zarr-developers/zarr-python/issues/3303))
- Reorganize the top-level `examples` directory to give each example its own sub-directory. Adds content to the docs for each example. ([#3502](https://github.com/zarr-developers/zarr-python/issues/3502))
- Updated 3.0 Migration Guide to include function signature change to zarr.Array.resize function. ([#3536](https://github.com/zarr-developers/zarr-python/issues/3536))

### Misc

- [#3515](https://github.com/zarr-developers/zarr-python/issues/3515), [#3532](https://github.com/zarr-developers/zarr-python/issues/3532), [#3533](https://github.com/zarr-developers/zarr-python/issues/3533), [#3553](https://github.com/zarr-developers/zarr-python/issues/3553)


## zarr 3.1.3 (2025-09-18)

### Features

- Add a command-line interface to migrate v2 Zarr metadata to v3. Corresponding functions are also provided under zarr.metadata. ([#1798](https://github.com/zarr-developers/zarr-python/issues/1798))
- Add obstore implementation of delete_dir. ([#3310](https://github.com/zarr-developers/zarr-python/issues/3310))
- Adds a registry for chunk key encodings for extensibility. This allows users to implement a custom `ChunkKeyEncoding`, which can be registered via `register_chunk_key_encoding` or as an entry point under `zarr.chunk_key_encoding`. ([#3436](https://github.com/zarr-developers/zarr-python/issues/3436))
- Trying to open a group at a path where an array already exists now raises a helpful error. ([#3444](https://github.com/zarr-developers/zarr-python/issues/3444))

### Bugfixes

- Prevents creation of groups (.create_group) or arrays (.create_array) as children of an existing array. ([#2582](https://github.com/zarr-developers/zarr-python/issues/2582))
- Fix a bug preventing `ones_like`, `full_like`, `empty_like`, `zeros_like` and `open_like` functions from accepting an explicit specification of array attributes like shape, dtype, chunks etc. The functions `full_like`, `empty_like`, and `open_like` now also more consistently infer a `fill_value` parameter from the provided array. ([#2992](https://github.com/zarr-developers/zarr-python/issues/2992))
- LocalStore now uses atomic writes, which should prevent some cases of corrupted data. ([#3411](https://github.com/zarr-developers/zarr-python/issues/3411))
- Fix a potential race condition when using `zarr.create_array` with the `data` parameter set to a NumPy array. Previously Zarr was iterating over the newly created array with a granularity that was too low. Now Zarr chooses a granularity that matches the size of the stored objects for that array. ([#3422](https://github.com/zarr-developers/zarr-python/issues/3422))
- Fix ChunkGrid definition (broken in 3.1.2) ([#3425](https://github.com/zarr-developers/zarr-python/issues/3425))
- Ensure syntax like `root['/subgroup']` works equivalently to `root['subgroup']` when using consolidated metadata. ([#3428](https://github.com/zarr-developers/zarr-python/issues/3428))
- Creating a new group with `zarr.group` no longer errors. This fixes a regression introduced in version 3.1.2. ([#3431](https://github.com/zarr-developers/zarr-python/issues/3431))
- Setting `fill_value` to a float like `0.0` when the data type of the array is an integer is a common mistake. This change lets Zarr Python read arrays with this erroneous metadata, although Zarr Python will not create such arrays. ([#3448](https://github.com/zarr-developers/zarr-python/issues/3448))

### Deprecations and Removals

- The `Store.set_partial_writes` method, which was not used by Zarr-Python, has been removed. `store.supports_partial_writes` is now always `False`. ([#2859](https://github.com/zarr-developers/zarr-python/issues/2859))

### Misc

- [#3376](https://github.com/zarr-developers/zarr-python/issues/3376), [#3390](https://github.com/zarr-developers/zarr-python/issues/3390), [#3403](https://github.com/zarr-developers/zarr-python/issues/3403), [#3449](https://github.com/zarr-developers/zarr-python/issues/3449)

## 3.1.2 (2025-08-25)

### Features

- Added support for async vectorized and orthogonal indexing. ([#3083](https://github.com/zarr-developers/zarr-python/issues/3083))
- Make config param optional in init_array ([#3391](https://github.com/zarr-developers/zarr-python/issues/3391))

### Bugfixes

- Ensure that -0.0 is not considered equal to 0.0 when checking if all the values in a chunk are equal to an array's fill value. ([#3144](https://github.com/zarr-developers/zarr-python/issues/3144))
- Fix a bug in `create_array` caused by iterating over chunk-aligned regions instead of shard-aligned regions when writing data. Additionally, the behavior of `nchunks_initialized` has been adjusted. This function consistently reports the number of chunks present in stored objects, even when the array uses the sharding codec. ([#3299](https://github.com/zarr-developers/zarr-python/issues/3299))
- Opening an array or group with `mode="r+"` will no longer create new arrays or groups. ([#3307](https://github.com/zarr-developers/zarr-python/issues/3307))
- Added `zarr.errors.ArrayNotFoundError`, which is raised when attempting to open a zarr array that does not exist, and `zarr.errors.NodeNotFoundError`, which is raised when failing to open an array or a group in a context where either an array or a group was expected. ([#3367](https://github.com/zarr-developers/zarr-python/issues/3367))
- Ensure passing `config` is handled properly when `open`ing an existing array. ([#3378](https://github.com/zarr-developers/zarr-python/issues/3378))
- Raise a Zarr-specific error class when a codec can't be found by name when deserializing the given codecs. This avoids hiding this error behind a "not part of a zarr hierarchy" warning. ([#3395](https://github.com/zarr-developers/zarr-python/issues/3395))

### Misc

- [#3098](https://github.com/zarr-developers/zarr-python/issues/3098), [#3288](https://github.com/zarr-developers/zarr-python/issues/3288), [#3318](https://github.com/zarr-developers/zarr-python/issues/3318), [#3368](https://github.com/zarr-developers/zarr-python/issues/3368), [#3371](https://github.com/zarr-developers/zarr-python/issues/3371), [#3372](https://github.com/zarr-developers/zarr-python/issues/3372), [#3374](https://github.com/zarr-developers/zarr-python/issues/3374)

## 3.1.1 (2025-07-28)

### Features

- Add lightweight implementations of `.getsize()` and `.getsize_prefix()` for ObjectStore. ([#3227](https://github.com/zarr-developers/zarr-python/issues/3227))

### Bugfixes

- Creating a Zarr format 2 array with the `order` keyword argument no longer raises a warning. ([#3112](https://github.com/zarr-developers/zarr-python/issues/3112))
- Fixed the error message when passing both `config` and `write_empty_chunks` arguments to reflect the current behaviour (`write_empty_chunks` takes precedence). ([#3112](https://github.com/zarr-developers/zarr-python/issues/3112))
- Creating a Zarr format 3 array with the `order` argument now consistently ignores this argument and raises a warning. ([#3112](https://github.com/zarr-developers/zarr-python/issues/3112))
- When using [`from_array`][zarr.api.asynchronous.from_array] to copy a Zarr format 2 array to a Zarr format 3 array, if the memory order of the input array is `"F"` a warning is raised and the order ignored. This is because Zarr format 3 arrays are always stored in "C" order. ([#3112](https://github.com/zarr-developers/zarr-python/issues/3112))
- The `config` argument to [`zarr.create`][zarr.create] (and functions that create arrays) is now used - previously it had no effect. ([#3112](https://github.com/zarr-developers/zarr-python/issues/3112))
- Ensure that all abstract methods of [`ZDType`][zarr.core.dtype.ZDType] raise a `NotImplementedError` when invoked. ([#3251](https://github.com/zarr-developers/zarr-python/issues/3251))
- Register 'gpu' marker with pytest for downstream StoreTests. ([#3258](https://github.com/zarr-developers/zarr-python/issues/3258))
- Expand the range of types accepted by `parse_data_type` to include strings and Sequences.
- Move the functionality of `zarr.core.dtype.parse_data_type` to a new function called `zarr.dtype.parse_dtype`. This change ensures that nomenclature is consistent across the codebase. `zarr.core.dtype.parse_data_type` remains, so this change is not breaking. ([#3264](https://github.com/zarr-developers/zarr-python/issues/3264))
- Fix a regression introduced in 3.1.0 that prevented `inf`, `-inf`, and `nan` values from being stored in `attributes`. ([#3280](https://github.com/zarr-developers/zarr-python/issues/3280))
- Fixes [`Group.nmembers()`][zarr.Group.nmembers] ignoring depth when using consolidated metadata. ([#3287](https://github.com/zarr-developers/zarr-python/issues/3287))

### Improved Documentation

- Expand the data type docs to include a demonstration of the `parse_data_type` function. Expand the docstring for the `parse_data_type` function. ([#3249](https://github.com/zarr-developers/zarr-python/issues/3249))
- Add a section on codecs to the migration guide. ([#3273](https://github.com/zarr-developers/zarr-python/issues/3273))

### Misc

- Remove warnings about vlen-utf8 and vlen-bytes codecs ([#3268](https://github.com/zarr-developers/zarr-python/issues/3268))

## 3.1.0 (2025-07-14)

### Features

- Ensure that invocations of `create_array` use consistent keyword arguments, with consistent defaults.

  [`zarr.api.synchronous.create_array`][] now takes a `write_data` keyword argument
  The `Group.create_array` method takes `data` and `write_data` keyword arguments.
  The functions [`zarr.api.asynchronous.create`][], [`zarr.api.asynchronous.create_array`]
  and the methods `Group.create_array`, `Group.array`, had the default
  `fill_value` changed from `0` to the `DEFAULT_FILL_VALUE` value, which instructs Zarr to
  use the default scalar value associated with the array's data type as the fill value. These are
  all functions or methods for array creation that mirror, wrap or are wrapped by, another function
  that already has a default `fill_value` set to `DEFAULT_FILL_VALUE`. This change is necessary
  to make these functions consistent across the entire codebase, but as this changes default values,
  new data might have a different fill value than expected after this change.

  For data types where 0 is meaningful, like integers or floats, the default scalar is 0, so this
  change should not be noticeable. For data types where 0 is ambiguous, like fixed-length unicode
  strings, the default fill value might be different after this change. Users who were relying on how
  Zarr interpreted `0` as a non-numeric scalar value should set their desired fill value explicitly
  after this change.

- Added public API for Buffer ABCs and implementations.

  Use `zarr.buffer` to access buffer implementations, and
  `zarr.abc.buffer` for the interface to implement new buffer types.

  Users previously importing buffer from `zarr.core.buffer` should update their
  imports to use `zarr.buffer`. As a reminder, all of `zarr.core` is
  considered a private API that's not covered by zarr-python's versioning policy. ([#2871](https://github.com/zarr-developers/zarr-python/issues/2871))

- Adds zarr-specific data type classes.

  This change adds a `ZDType` base class for Zarr V2 and Zarr V3 data types. Child classes are
  defined for each NumPy data type. Each child class defines routines for `JSON` serialization.
  New data types can be created and registered dynamically.

  Prior to this change, Zarr Python had two streams for handling data types. For Zarr V2 arrays,
  we used NumPy data type identifiers. For Zarr V3 arrays, we used a fixed set of string enums. Both
  of these systems proved hard to extend.

  This change is largely internal, but it does change the type of the `dtype` and `data_type`
  fields on the `ArrayV2Metadata` and `ArrayV3Metadata` classes. Previously, `ArrayV2Metadata.dtype`
  was a NumPy `dtype` object, and `ArrayV3Metadata.data_type` was an internally-defined `enum`.
  After this change, both `ArrayV2Metadata.dtype` and `ArrayV3Metadata.data_type` are instances of
  `ZDType`. A NumPy data type can be generated from a `ZDType` via the `ZDType.to_native_dtype()`
  method. The internally-defined Zarr V3 `enum` class is gone entirely, but the `ZDType.to_json(zarr_format=3)`
  method can be used to generate either a string, or dictionary that has a string `name` field, that
  represents the string value previously associated with that `enum`.

  For more on this new feature, see the [documentation](user-guide/data_types.md) ([#2874](https://github.com/zarr-developers/zarr-python/issues/2874))

- Added `NDBuffer.empty` method for faster ndbuffer initialization. ([#3191](https://github.com/zarr-developers/zarr-python/issues/3191))

- The minimum version of NumPy has increased to 1.26. ([#3226](https://github.com/zarr-developers/zarr-python/issues/3226))

- Add an alternate `from_array_metadata_and_store` constructor to `CodecPipeline`. ([#3233](https://github.com/zarr-developers/zarr-python/issues/3233))

### Bugfixes

- Fixes a variety of issues related to string data types.

  - Brings the `VariableLengthUTF8` data type Zarr V3 identifier in alignment with Zarr Python 3.0.8
  - Disallows creation of 0-length fixed-length data types
  - Adds a regression test for the `VariableLengthUTF8` data type that checks against version 3.0.8
  - Allows users to request the `VariableLengthUTF8` data type with `str`, `"str"`, or `"string"`. ([#3170](https://github.com/zarr-developers/zarr-python/issues/3170))

- Add human readable size for No. bytes stored to `info_complete` ([#3190](https://github.com/zarr-developers/zarr-python/issues/3190))

- Restores the ability to create a Zarr V2 array with a `null` fill value by introducing a new
  class `DefaultFillValue`, and setting the default value of the `fill_value` parameter in array
  creation routines to an instance of `DefaultFillValue`. For Zarr V3 arrays, `None` will act as an
  alias for a `DefaultFillValue` instance, thus preserving compatibility with existing code. ([#3198](https://github.com/zarr-developers/zarr-python/issues/3198))

- Fix the type of `ArrayV2Metadata.codec` to constrain it to `numcodecs.abc.Codec | None`.
  Previously the type was more permissive, allowing objects that can be parsed into Codecs (e.g., the codec name).
  The constructor of `ArrayV2Metadata` still allows the permissive input when creating new objects. ([#3232](https://github.com/zarr-developers/zarr-python/issues/3232))

### Improved Documentation

- Add a self-contained example of data type extension to the `examples` directory, and expanded
  the documentation for data types. ([#3157](https://github.com/zarr-developers/zarr-python/issues/3157))

- Add a description on how to create a RemoteStore of a specific filesystem to the `Remote Store` section in `docs/user-guide/storage.md`.
  State in the docstring of `FsspecStore.from_url` that the filesystem type is inferred from the URL scheme.

  It should help a user handling the case when the type of FsspecStore doesn't match the URL scheme. ([#3212](https://github.com/zarr-developers/zarr-python/issues/3212))

### Deprecations and Removals

- Removes default chunk encoding settings (filters, serializer, compressors) from the global
  configuration object.

  This removal is justified on the basis that storing chunk encoding settings in the config required
  a brittle, confusing, and inaccurate categorization of array data types, which was particularly
  unsuitable after the recent addition of new data types that didn't fit naturally into the
  pre-existing categories.

  The default chunk encoding is the same (Zstandard compression, and the required object codecs for
  variable length data types), but the chunk encoding is now generated by functions that cannot be
  reconfigured at runtime. Users who relied on setting the default chunk encoding via the global configuration object should
  instead specify the desired chunk encoding explicitly when creating an array.

  This change also adds an extra validation step to the creation of Zarr V2 arrays, which ensures that
  arrays with a `VariableLengthUTF8` or `VariableLengthBytes` data type cannot be created without the
  correct "object codec". ([#3228](https://github.com/zarr-developers/zarr-python/issues/3228))

- Removes support for passing keyword-only arguments positionally to the following functions and methods:
  `save_array`, `open`, `group`, `open_group`, `create`, `get_basic_selection`, `set_basic_selection`,
  `get_orthogonal_selection`,  `set_orthogonal_selection`, `get_mask_selection`, `set_mask_selection`,
  `get_coordinate_selection`, `set_coordinate_selection`, `get_block_selection`, `set_block_selection`,
  `Group.create_array`, `Group.empty`, `Group.zeroes`, `Group.ones`, `Group.empty_like`, `Group.full`,
  `Group.zeros_like`, `Group.ones_like`, `Group.full_like`, `Group.array`. Prior to this change,
  passing a keyword-only argument positionally to one of these functions or methods would raise a
  deprecation warning. That warning is now gone. Passing keyword-only arguments to these functions
  and methods positionally is now an error.

## 3.0.10 (2025-07-03)

### Bugfixes

- Removed an unnecessary check from `_fsspec._make_async` that would raise an exception when
  creating a read-only store backed by a local file system with `auto_mkdir` set  to `False`. ([#3193](https://github.com/zarr-developers/zarr-python/issues/3193))

- Add missing import for AsyncFileSystemWrapper for _make_async in _fsspec.py ([#3195](https://github.com/zarr-developers/zarr-python/issues/3195))

## 3.0.9 (2025-06-30)

### Features

- Add `zarr.storage.FsspecStore.from_mapper()` so that `zarr.open()` supports stores of type `fsspec.mapping.FSMap`. ([#2774](https://github.com/zarr-developers/zarr-python/issues/2774))

- Implemented `move` for `LocalStore` and `ZipStore`. This allows users to move the store to a different root path. ([#3021](https://github.com/zarr-developers/zarr-python/issues/3021))

- Added `zarr.errors.GroupNotFoundError`, which is raised when attempting to open a group that does not exist. ([#3066](https://github.com/zarr-developers/zarr-python/issues/3066))

- Adds `fill_value` to the list of attributes displayed in the output of the `AsyncArray.info()` method. ([#3081](https://github.com/zarr-developers/zarr-python/issues/3081))

- Use `numpy.zeros` instead of `np.full` for a performance speedup when creating a `zarr.core.buffer.NDBuffer` with `fill_value=0`. ([#3082](https://github.com/zarr-developers/zarr-python/issues/3082))

- Port more stateful testing actions from [Icechunk](https://icechunk.io). ([#3130](https://github.com/zarr-developers/zarr-python/issues/3130))

- Adds a `with_read_only` convenience method to the `Store` abstract base class (raises `NotImplementedError`) and implementations to the `MemoryStore`, `ObjectStore`, `LocalStore`, and `FsspecStore` classes. ([#3138](https://github.com/zarr-developers/zarr-python/issues/3138))

### Bugfixes

- Ignore stale child metadata when reconsolidating metadata. ([#2921](https://github.com/zarr-developers/zarr-python/issues/2921))

- For Zarr format 2, allow fixed-length string arrays to be created without automatically inserting a
  `Vlen-UT8` codec in the array of filters. Fixed-length string arrays do not need this codec. This
  change fixes a regression where fixed-length string arrays created with Zarr Python 3 could not be read with Zarr Python 2.18. ([#3100](https://github.com/zarr-developers/zarr-python/issues/3100))

- When creating arrays without explicitly specifying a chunk size using `zarr.create` and other
  array creation routines, the chunk size will now set automatically instead of defaulting to the data shape.
  For large arrays this will result in smaller default chunk sizes.
  To retain previous behaviour, explicitly set the chunk shape to the data shape.

  This fix matches the existing chunking behaviour of
  `zarr.save_array` and `zarr.api.asynchronous.AsyncArray.create`. ([#3103](https://github.com/zarr-developers/zarr-python/issues/3103))

- When `zarr.save` has an argument `path=some/path/` and multiple arrays in `args`, the path resulted in `some/path/some/path` due to using the `path`
  argument twice while building the array path. This is now fixed. ([#3127](https://github.com/zarr-developers/zarr-python/issues/3127))

- Fix `zarr.open` default for argument `mode` when `store` is `read_only` ([#3128](https://github.com/zarr-developers/zarr-python/issues/3128))

- Suppress `FileNotFoundError` when deleting non-existent keys in the `obstore` adapter.

  When writing empty chunks (i.e. chunks where all values are equal to the array's fill value) to a zarr array, zarr
  will delete those chunks from the underlying store. For zarr arrays backed by the `obstore` adapter, this will potentially
  raise a `FileNotFoundError` if the chunk doesn't already exist.
  Since whether or not a delete of a non-existing object raises an error depends on the behavior of the underlying store,
  suppressing the error in all cases results in consistent behavior across stores, and is also what `zarr` seems to expect
  from the store. ([#3140](https://github.com/zarr-developers/zarr-python/issues/3140))

- Trying to open a StorePath/Array with `mode='r'` when the store is not read-only creates a read-only copy of the store. ([#3156](https://github.com/zarr-developers/zarr-python/issues/3156))

## 3.0.8 (2025-05-19)

!!! warning

    In versions 3.0.0 to 3.0.7 opening arrays or groups with `mode='a'` (the default for many builtin functions) would cause any existing paths in the store to be deleted. This is fixed in 3.0.8, and we recommend all users upgrade to avoid this bug that could cause unintentional data loss.

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
