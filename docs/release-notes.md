# Release notes

<!-- markdownlint-disable MD012 -- towncrier intentionally emits a blank line between releases -->

<!-- towncrier release notes start -->

## 3.3.0 (2026-07-15)

### Features

- Optimizes reading multiple chunks from a shard. Serial calls to `Store.get()`
  in the sharding codec have been replaced with a single call to
  `Store.get_ranges()`, which coalesces nearby byte ranges and fetches them
  concurrently. ([#3004](https://github.com/zarr-developers/zarr-python/issues/3004))
- Added a `subchunk_write_order` option to `ShardingCodec` to control the physical order of subchunks within a shard. Supported values are `morton`, `unordered`, `lexicographic`, and `colexicographic`. `unordered` makes no guarantee about subchunk layout. This setting affects only on-disk layout, not the data read back, and is not persisted in array metadata: it applies per codec instance and is not recovered when reopening a sharded array. ([#3826](https://github.com/zarr-developers/zarr-python/issues/3826))
- Added `SyncByteGetter` and `SyncByteSetter` runtime-checkable protocols and a `get_ranges_sync` method on the `Store` ABC. These let custom byte getters/setters opt into the synchronous codec pipeline's fast path for in-memory IO, which the sharding codec uses for its inner chunks. ([#3885](https://github.com/zarr-developers/zarr-python/issues/3885))
- Added `FusedCodecPipeline`, an opt-in codec pipeline that runs codec compute synchronously and in bulk (avoiding the per-chunk async scheduling overhead of the default `BatchedCodecPipeline`), giving large speedups for sharded arrays (up to ~24x writes / ~14x reads on many-chunks-per-shard layouts, more with compression) and no regressions on compute-bound workloads. The default `BatchedCodecPipeline` is unchanged for standard configurations, so existing code keeps working unless you opt in; enable the new pipeline with `zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"})`. ([#3885](https://github.com/zarr-developers/zarr-python/issues/3885))
- Add `zarr.abc.store.Store.get_ranges` for concurrent, coalesced multi-range reads from a single key. The method is defined on the `Store` ABC with a default implementation built on `Store.get`, so every store inherits a working version; stores with native multi-range backends (e.g. `FsspecStore`) can override for efficiency. Coalescing knobs (`max_concurrency`, `max_gap_bytes`, `max_coalesced_bytes`) are passed as keyword arguments to `get_ranges`. Failures from underlying fetches surface as a `BaseExceptionGroup` (PEP 654); callers should use `except*` to filter for specific exception types such as `FileNotFoundError`. ([#3925](https://github.com/zarr-developers/zarr-python/issues/3925))
- Two new fields on `ArrayConfig` control how the sharding codec coalesces partial-shard reads: `sharding_coalesce_max_gap_bytes` (default 1 MiB) and `sharding_coalesce_max_bytes` (default 16 MiB). When reading multiple chunks from the same shard, nearby byte ranges are merged into a single request to the store if separated by no more than `sharding_coalesce_max_gap_bytes` and the merged read stays within `sharding_coalesce_max_bytes`. Defaults are seeded from the matching `array.sharding_coalesce_max_gap_bytes` / `array.sharding_coalesce_max_bytes` keys in [`zarr.config`][] at array-creation time, and can be overridden per array by passing `config={...}` to [`zarr.create_array`][]. ([#3987](https://github.com/zarr-developers/zarr-python/issues/3987))

### Bugfixes

- Stop emitting an `UnstableSpecificationWarning` when serializing the `struct` data type to Zarr V3 metadata. The `struct` data type now has a stable Zarr V3 specification. The legacy `structured` alias and the unspecified `null_terminated_bytes`, `raw_bytes`, and `variable_length_bytes` data types continue to warn. ([#202](https://github.com/zarr-developers/zarr-python/issues/202))
- Fix equality comparison of `ArrayV2Metadata` and `ArrayV3Metadata` objects with a
  `NaN` fill value. Such objects are now compared by their JSON-serialized form, so two
  otherwise-identical metadata objects with a `NaN` (or infinite) fill value compare equal. ([#2929](https://github.com/zarr-developers/zarr-python/issues/2929))
- Fixed `BytesCodec.from_dict` so that `BytesCodec` instances roundtrip to / from
  their dict representation. `BytesCodec.from_dict` now interprets a missing
  `endian` configuration as `endian=None` (matching what `BytesCodec.to_dict`
  emits), instead of falling back to the system's native byte order. ([#3417](https://github.com/zarr-developers/zarr-python/issues/3417))
- Fixed `save_array`, `Group.__setitem__`, and `load` for 0-dimensional arrays. ([#3469](https://github.com/zarr-developers/zarr-python/issues/3469))
- Fixed inner-codec spec evolution for sharded arrays. The sharding codec now threads the array spec through its inner codec chain when evolving codecs, so a codec that changes the dtype upstream of `BytesCodec` no longer leaves the inner chain evolved against the wrong spec (which previously failed at decode time). This runs on the default `BatchedCodecPipeline` as well. Standard inner chains (`[BytesCodec]`, `[BytesCodec, ZstdCodec]`, transpose + bytes) are byte-identical to before. Restores the behavior of #2179. ([#3885](https://github.com/zarr-developers/zarr-python/issues/3885))
- Fixed the opt-in `FusedCodecPipeline` for sharded arrays whose inner or index codec chain contains a codec implementing only the async codec interface (no `SupportsSyncCodec`). Such arrays previously raised `TypeError: All codecs must implement SupportsSyncCodec` on both read and write; the pipeline now declines its synchronous fast path for them and falls back to the async path, matching the behavior of the default `BatchedCodecPipeline`. Fully sync-capable codec chains keep the fast path unchanged. ([#4179](https://github.com/zarr-developers/zarr-python/issues/4179))
- Make chunk normalization properly handle `-1` as a compact representation of the
  length of an entire axis. Reject several previously-accepted but ill-defined
  chunk specifications: `chunks=True` (previously silently produced size-1 chunks),
  chunk tuples shorter than the array's number of dimensions (previously padded to
  the array's shape), and `None` as a per-dimension chunk size. These all now
  raise informative errors. Also fix chunk handling for 0-length array dimensions,
  and add explicit rejection of 0-length chunks. ([#3899](https://github.com/zarr-developers/zarr-python/issues/3899))
- Handle missing consolidated metadata in leaf Group nodes. ([#3954](https://github.com/zarr-developers/zarr-python/issues/3954))
- Corrected the JSON type definitions for the `numpy.datetime64` and
  `numpy.timedelta64` data types in Zarr V3 metadata: the `configuration` object
  (holding `unit` and `scale_factor`) is now required, matching the published
  specifications for these data types. Also updated the specification links in
  the docstrings to point to the zarr-extensions repository. ([#3955](https://github.com/zarr-developers/zarr-python/issues/3955))
- Fixed writing to 0-dimensional arrays that use the sharding codec. Previously
  assigning to a 0-dimensional sharded array raised an error. ([#3966](https://github.com/zarr-developers/zarr-python/issues/3966))
- Fix flaky stateful test bookkeeping when `delete_dir` matches string prefixes instead of true directory descendants. Previously a path such as `6/faNT…` could be incorrectly removed when deleting `6/f`. (See [issue #3977](https://github.com/zarr-developers/zarr-python/issues/3977).) ([#3977](https://github.com/zarr-developers/zarr-python/issues/3977))
- `FsspecStore.from_url()` and `from_mapper()` now close the async filesystem
  they create when `store.close()` is called. Previously the underlying aiohttp
  `ClientSession` was left open until garbage collection, producing
  `"Unclosed client session"` `ResourceWarning`s from aiohttp.

    The fix introduces `FsspecStore._owns_fs`, a boolean that is ``True`` only when
    `FsspecStore` itself created the filesystem (via `from_url` or `from_mapper`
    when a sync→async conversion was performed). When `_owns_fs` is ``True``,
    `store.close()` calls the new `_close_fs()` helper, which invokes
    `fs.set_session()` and closes the returned client. Callers who supply their own
    filesystem instance to `FsspecStore()` directly remain responsible for its
    lifecycle; `_owns_fs` is ``False`` for those stores.

    **Scope note**: This fix closes the S3 client session that is active at the time
    `store.close()` is called. Some S3-backed filesystem implementations (e.g.
    s3fs with ``cache_regions=True``) may internally refresh and replace their
    client during I/O operations, abandoning prior sessions before ``store.close()``
    is invoked. Those intermediate sessions are outside the scope of this fix and
    are an issue in the upstream filesystem library. ([#4003](https://github.com/zarr-developers/zarr-python/issues/4003))

- Fixed an invalid `zarr.create_array` example in the quick-start documentation (it passed an unsupported `mode` argument) and made the cloud-storage example execute against a mock S3 backend in CI. Added a test ensuring every Python code block in the documentation is either executed or explicitly opted out with a documented reason, so an invalid example can no longer go untested. ([#4016](https://github.com/zarr-developers/zarr-python/issues/4016))
- Fixed `ObjectStore.list_dir` for object-store listings that include a directory-marker object matching the requested non-root prefix. ([#4032](https://github.com/zarr-developers/zarr-python/issues/4032))
- Prevents mutation of the attributes dict provided by the user by copying them instead of keeping the reference ([#4059](https://github.com/zarr-developers/zarr-python/issues/4059))
- Fixed several storage and codec bugs:

    - Reading a value with a `SuffixByteRequest` larger than the value now correctly returns the whole value (matching HTTP `bytes=-N` suffix-range semantics), instead of silently returning incorrect data for `MemoryStore`.
    - `LoggingStore.get_partial_values` and `FsspecStore.get_partial_values` no longer return empty results when `key_ranges` is passed as a one-shot iterable (e.g. a generator).
    - `Store.getsize_prefix` no longer over-counts sibling keys that merely share a string prefix (e.g. `getsize_prefix("foo")` no longer includes keys under `foobar/`).
    - `ZipStore.close()` no longer raises `AttributeError` when the store was created but never opened (including when used as a context manager without any I/O).
    - `codecs_from_list` now raises a descriptive `TypeError` when a `BytesBytesCodec` immediately follows an `ArrayArrayCodec`, instead of a misleading "Required ArrayBytesCodec was not found" `ValueError`.

    ([#4074](https://github.com/zarr-developers/zarr-python/issues/4074))

- Fixed writing Fortran-ordered (F-contiguous) arrays through the variable-length string and bytes codecs and through numcodecs array-array filters such as `Delta`, `FixedScaleOffset` and `PackBits`. Chunks are now passed to numcodecs as C-contiguous arrays, so elements are no longer stored in transposed order. ([#4116](https://github.com/zarr-developers/zarr-python/issues/4116))
- Fix silent byte-order corruption for structured dtypes with the `bytes` codec: multi-byte fields are now byte-swapped to the codec's configured `endian` on write and decoded honoring it on read, so non-native-endian structured data (e.g. big-endian fields, as produced by virtual references to external data) round-trips correctly. ([#4141](https://github.com/zarr-developers/zarr-python/issues/4141))

### Improved Documentation

- Document the changes to `zarr.errors` in the 3.0 migration guide, including the removal of v2 exception classes and the introduction of `NodeNotFoundError`. ([#3009](https://github.com/zarr-developers/zarr-python/issues/3009))
- Clarify the difference between `zarr.load` and `zarr.open` in their docstrings.
  `load` eagerly reads data into an in-memory array, while `open` returns a
  lazy `Array` or `Group` backed by the store, with `See Also` cross-references
  linking the two. ([#3984](https://github.com/zarr-developers/zarr-python/issues/3984))
- Updated the custom dtype example in `examples/custom_dtype/custom_dtype.py` to
  use only the public API, eliminating all non-public imports, illustrating what
  users should do.

    To better support this, the following types and functions were made available
    from public modules:

    | Type/Function             | Non-public module        | Public module |
    | ------------------------- | ------------------------ | ------------- |
    | `DataTypeValidationError` | `zarr.core.dtype.common` | `zarr.errors` |
    | `JSON`                    | `zarr.core.common`       | `zarr.types`  |
    | `ZarrFormat`              | `zarr.core.common`       | `zarr.types`  |
    | `DTypeConfig_V2`          | `zarr.core.dtype.common` | `zarr.types`  |
    | `DTypeJSON`               | `zarr.core.dtype.common` | `zarr.types`  |
    | `DTypeSpec_V2`            | `zarr.core.dtype.common` | `zarr.dtype`  |
    | `check_dtype_spec_v2`     | `zarr.core.dtype.common` | `zarr.dtype`  |

    `DataTypeValidationError` was *moved* to `zarr.errors`. Importing it from
    `zarr.core.dtype.common` (its original location), `zarr.core.dtype`, or
    `zarr.dtype` still works but now raises a `ZarrDeprecationWarning`. The remaining
    types and functions are simply re-exported from the listed public module. ([#4052](https://github.com/zarr-developers/zarr-python/issues/4052))

- Document a self-merge policy in the contributor guide, describing when a core developer may merge their own pull request without a second reviewer and which changes warrant more caution. ([#4053](https://github.com/zarr-developers/zarr-python/issues/4053))
- Fixed many documentation errors found in a full review of the user guide, including
  prose contradicted by rendered example output on the performance page, invisible
  code blocks, an incorrect S3 example, stale "not yet implemented" claims in the
  v3 migration guide, and undocumented optional dependency groups. Also improved
  navigation order, cross-linking between pages, and coverage of group member
  enumeration, bulk attribute updates, and the `use_consolidated` keyword. ([#4132](https://github.com/zarr-developers/zarr-python/issues/4132))
- Fixed the documented default of ``max_age_seconds`` in the ``CacheStore`` docstring: the default is ``"infinity"`` (no expiration), not ``None``, which is rejected. Also noted that ``cache_store`` must support deletes. ([#4133](https://github.com/zarr-developers/zarr-python/issues/4133))

### Deprecations and Removals

- The ``BloscShuffle`` and ``BloscCname`` enums (``zarr.codecs.BloscShuffle``,
  ``zarr.codecs.BloscCname``) are now deprecated. Pass the equivalent literal
  string (e.g. ``"zstd"``, ``"bitshuffle"``) when constructing a ``BloscCodec``.
  The enum classes remain importable but emit ``DeprecationWarning`` on member
  access, and will be removed in a future release. ``BloscCodec.cname`` and
  ``BloscCodec.shuffle`` are now plain strings rather than enum members.

    Additional renames in ``zarr.codecs.blosc`` from the same change: the type
    aliases ``Shuffle`` and ``CName`` are now ``BloscShuffleLiteral`` and
    ``BloscCnameLiteral``, the constant ``SHUFFLE`` is now ``BLOSC_SHUFFLE``
    (with a new ``BLOSC_CNAME`` alongside it), and ``BloscShuffle.from_int``
    now returns a literal string rather than an enum member. ([#3963](https://github.com/zarr-developers/zarr-python/issues/3963))

- The ``Endian`` (``zarr.codecs.bytes.Endian``) and ``ShardingCodecIndexLocation``
  (``zarr.codecs.ShardingCodecIndexLocation``) enums are now deprecated. Pass the
  equivalent literal string instead (e.g. ``"little"`` / ``"big"``, ``"start"`` /
  ``"end"``). The enum classes remain importable but emit ``DeprecationWarning``
  on member access, and will be removed in a future release. ``BytesCodec.endian``
  and ``ShardingCodec.index_location`` are now plain strings rather than enum
  members.

    Two follow-on changes from this deprecation:

    - ``NDBuffer.byteorder`` now returns a literal string (``"little"`` or
      ``"big"``) rather than an ``Endian`` member. Subclasses overriding this
      property should update their return type.
    - The module-level binding ``zarr.codecs.bytes.default_system_endian`` was
      removed. ``BytesCodec()`` continues to default to ``sys.byteorder``;
      external callers that imported ``default_system_endian`` should use
      ``sys.byteorder`` directly.

    Additionally, the module-level function ``zarr.codecs.sharding.parse_index_location``
    was made private as part of this change.

    ([#3968](https://github.com/zarr-developers/zarr-python/issues/3968))

- Removed the NumPy 1.x implementation of the `VariableLengthUTF8` data type because NumPy 1.x is no longer supported under [SPEC0](https://scientific-python.org/specs/spec-0000/). ([#3973](https://github.com/zarr-developers/zarr-python/issues/3973))

### Misc

- [#214](https://github.com/zarr-developers/zarr-python/issues/214), [#215](https://github.com/zarr-developers/zarr-python/issues/215), [#3908](https://github.com/zarr-developers/zarr-python/issues/3908), [#3972](https://github.com/zarr-developers/zarr-python/issues/3972), [#3975](https://github.com/zarr-developers/zarr-python/issues/3975), [#3979](https://github.com/zarr-developers/zarr-python/issues/3979), [#3990](https://github.com/zarr-developers/zarr-python/issues/3990), [#3998](https://github.com/zarr-developers/zarr-python/issues/3998), [#4000](https://github.com/zarr-developers/zarr-python/issues/4000), [#4001](https://github.com/zarr-developers/zarr-python/issues/4001), [#4046](https://github.com/zarr-developers/zarr-python/issues/4046), [#4054](https://github.com/zarr-developers/zarr-python/issues/4054), [#4073](https://github.com/zarr-developers/zarr-python/issues/4073), [#4086](https://github.com/zarr-developers/zarr-python/issues/4086), [#4138](https://github.com/zarr-developers/zarr-python/issues/4138)


## 3.2.1 (2026-05-05)

### Bugfixes

- Fixed a `CastValue` validation bug where the "can we use an out-of-range mode" check
  inspected the source dtype instead of the target dtype. This meant arrays with a
  float source dtype and an integer target dtype incorrectly raised a `ValueError`
  when configured with a `wrap` out-of-range mode. ([#3938](https://github.com/zarr-developers/zarr-python/issues/3938))
- Fixed a bug where the codec pipeline evolved each codec against the original
  array spec instead of the spec produced by upstream array-to-array codecs. This
  caused failures whenever an upstream codec changed the dtype between codec
  boundaries — e.g. arrays using `CastValue` to convert a single-byte source dtype
  (`int8`) to a multi-byte target dtype (`int16`) raised a `ValueError` from
  `BytesCodec` about a missing `endian` configuration. ([#3941](https://github.com/zarr-developers/zarr-python/issues/3941))
- Fixed breakage in existing fsspec-dependent workflows caused by associating the "memory" URL scheme with
instances of `ManagedMemoryStore` instead of fsspec's memory-backed store. After this change, store URLs with a "memory" scheme are handled differently when `fsspec` is installed:
with `fsspec`, a `FsspecStore` backed by a `MemoryFileSystem` is used. Without `fsspec`,
a `ManagedMemoryStore` is used. ([#3944](https://github.com/zarr-developers/zarr-python/issues/3944))

## 3.2.0 (2026-04-30)

### Features

- Adds a new in-memory storage backend called `ManagedMemoryStore`. Instances of `ManagedMemoryStore`
  function similarly to `MemoryStore`, but instances of `ManagedMemoryStore` can be constructed from
  a URL like `memory://store`. ([#3679](https://github.com/zarr-developers/zarr-python/issues/3679))
- Added `array.read_missing_chunks` configuration option. When set to `False`, reading missing chunks raises a `ChunkNotFoundError` instead of filling them with the array's fill value. ([#3748](https://github.com/zarr-developers/zarr-python/issues/3748))
- Added `Struct` class (subclass of `Structured`) implementing the zarr-extensions `struct` dtype spec. Uses object-style field format and dict fill values. Legacy `Structured` remains available for backward compatibility. ([#3781](https://github.com/zarr-developers/zarr-python/issues/3781))
- Add support for rectilinear (variable-sized) chunk grids. This feature is experimental and
  must be explicitly enabled via `zarr.config.set({'array.rectilinear_chunks': True})`.

    Rectilinear chunks can be used through:

    - **Creating arrays**: Pass nested sequences (e.g., `[[10, 20, 30], [50, 50]]`) to `chunks`
      in `zarr.create_array`, `zarr.from_array`, `zarr.zeros`, `zarr.ones`, `zarr.full`,
      `zarr.open`, and related functions, or to `chunk_shape` in `zarr.create`.
    - **Opening existing arrays**: Arrays stored with the `rectilinear` chunk grid are read
      transparently via `zarr.open` and `zarr.open_array`.
    - **Rectilinear sharding**: Shard boundaries can be rectilinear while inner chunks remain regular.

    **Breaking change**: The `validate` method on `BaseCodec` and `CodecPipeline` now receives
    a `ChunkGridMetadata` instance instead of a `ChunkGrid` instance for the `chunk_grid`
    parameter. Third-party codecs that override `validate` and inspect the chunk grid will need to
    update their type annotations. No known downstream packages were using this parameter. ([#3802](https://github.com/zarr-developers/zarr-python/issues/3802))

- Add `cast_value` and `scale_offset` codecs. ([#3874](https://github.com/zarr-developers/zarr-python/issues/3874))

### Bugfixes

- Fix `SyncError` raised when assigning a `zarr.Array` as the value in a `__setitem__` call (e.g. `dst[:] = src` where `src` is a zarr array). The source array is now converted to a NumPy array before entering the async codec pipeline. ([#3611](https://github.com/zarr-developers/zarr-python/issues/3611))
- Fix an issue that prevents the correct parsing of special NumPy `uint32` dtypes resulting e.g.
  from bit wise operations on `uint32` arrays on Windows. ([#3797](https://github.com/zarr-developers/zarr-python/issues/3797))
- Fix `ZipStore.list()`, `list_dir()`, and `exists()` to auto-open the zip file when called before `open()`, consistent with the existing behavior of `get()` and `set()`. ([#3846](https://github.com/zarr-developers/zarr-python/issues/3846))
- Fix handling of `NaT` default fill values for `datetime64` and `timedelta64` data types. Equality checks now use `numpy.isnat` so that the default fill value compares correctly against `NaT`. ([#3863](https://github.com/zarr-developers/zarr-python/issues/3863))
- Use the unit associated with the `Datetime64` data type when creating the default `Nat` scalar value. ([#3920](https://github.com/zarr-developers/zarr-python/issues/3920))

### Improved Documentation

- Document removal of `zarr.storage.init_group` in v3 migration guide, with replacement using `zarr.open_group`/`zarr.create_group`. ([#2720](https://github.com/zarr-developers/zarr-python/issues/2720))
- Document the `threading.max_workers` configuration option in the performance guide. ([#3492](https://github.com/zarr-developers/zarr-python/issues/3492))
- Corrects the type annotation reported for the `batch_info` parameter in the `CodecPipeline.write`
  method docstring. ([#3836](https://github.com/zarr-developers/zarr-python/issues/3836))
- Remove result="ansi" from code blocks in the user guide that were causing empty output cells in the rendered documentation. ([#3845](https://github.com/zarr-developers/zarr-python/issues/3845))

### Deprecations and Removals

- Remove deprecated `zarr.convenience` and `zarr.creation` modules. ([#3900](https://github.com/zarr-developers/zarr-python/issues/3900))
- Remove the deprecated `zarr_version` parameter from several functions and methods. That parameter is replaced with `zarr_format`. ([#3901](https://github.com/zarr-developers/zarr-python/issues/3901))
- Remove deprecated `Group` methods `array`, `require_dataset`, and `create_dataset`. ([#3902](https://github.com/zarr-developers/zarr-python/issues/3902))
- Remove deprecated `AsyncArray.create` and `Array.create` methods. ([#3903](https://github.com/zarr-developers/zarr-python/issues/3903))

### Misc

- [#3546](https://github.com/zarr-developers/zarr-python/issues/3546), [#3793](https://github.com/zarr-developers/zarr-python/issues/3793), [#3800](https://github.com/zarr-developers/zarr-python/issues/3800), [#3828](https://github.com/zarr-developers/zarr-python/issues/3828), [#3830](https://github.com/zarr-developers/zarr-python/issues/3830), [#3833](https://github.com/zarr-developers/zarr-python/issues/3833), [#3837](https://github.com/zarr-developers/zarr-python/issues/3837), [#3897](https://github.com/zarr-developers/zarr-python/issues/3897)


## 3.1.6 (2026-03-19)

### Features

- Exposes the array runtime configuration as an attribute called `config` on the `Array` and
  `AsyncArray` classes. The previous `AsyncArray._config` attribute is now a deprecated alias for `AsyncArray.config`. ([#3668](https://github.com/zarr-developers/zarr-python/issues/3668))
- Adds a method for creating a new `Array` / `AsyncArray` instance with a new runtime configuration, and fixes inaccurate documentation about the `write_empty_chunks` configuration parameter. ([#3668](https://github.com/zarr-developers/zarr-python/issues/3668))
- Adds synchronous methods to stores that do not benefit from an async event loop. The shape of these methods is defined by protocol classes to support structural subtyping. ([#3725](https://github.com/zarr-developers/zarr-python/pull/3725))
- Fix near-miss penalty in `_morton_order` with hybrid ceiling+argsort strategy. ([#3718](https://github.com/zarr-developers/zarr-python/pull/3718))

### Bugfixes

- Correct the target bytes number for auto-chunking when auto-sharding. ([#3603](https://github.com/zarr-developers/zarr-python/issues/3603))
- Fixed a bug in the sharding codec that prevented nested shard reads in certain cases. ([#3655](https://github.com/zarr-developers/zarr-python/issues/3655))
- Fix obstore `_transform_list_dir` implementation to correctly relativize paths (removing `lstrip` usage). ([#3657](https://github.com/zarr-developers/zarr-python/issues/3657))
- Raise error when trying to encode `numpy.dtypes.StringDType` with `na_object` set. ([#3695](https://github.com/zarr-developers/zarr-python/issues/3695))
- `CacheStore`, `LoggingStore` and `LatencyStore` now support with_read_only. ([#3700](https://github.com/zarr-developers/zarr-python/issues/3700))
- Skip chunk coordinate enumeration in resize when the array is only growing, avoiding unbounded memory usage for large arrays. ([#3702](https://github.com/zarr-developers/zarr-python/issues/3702))
- Fix a performance bug in morton curve generation. ([#3705](https://github.com/zarr-developers/zarr-python/issues/3705))
- Add a dedicated in-memory cache for byte-range requests to the experimental `CacheStore`. ([#3710](https://github.com/zarr-developers/zarr-python/issues/3710))
- `BaseFloat._check_scalar` rejects invalid string values. ([#3586](https://github.com/zarr-developers/zarr-python/issues/3586))
- Apply drop_axes squeeze in partial decode path for sharding. ([#3763](https://github.com/zarr-developers/zarr-python/issues/3763))
- Set `copy=False` in reshape operation. ([#3649](https://github.com/zarr-developers/zarr-python/issues/3649))
- Validate that dask-style chunks have regular shapes. ([#3779](https://github.com/zarr-developers/zarr-python/issues/3779))

### Improved Documentation

- Add documentation example for creating uncompressed arrays in the Compression section of the user guide. ([#3464](https://github.com/zarr-developers/zarr-python/issues/3464))
- Add AI-assisted code policy to the contributing guide. ([#3769](https://github.com/zarr-developers/zarr-python/issues/3769))
- Added a glossary. ([#3767](https://github.com/zarr-developers/zarr-python/issues/3767))

### Misc

- [#3562](https://github.com/zarr-developers/zarr-python/issues/3562), [#3605](https://github.com/zarr-developers/zarr-python/issues/3605), [#3619](https://github.com/zarr-developers/zarr-python/issues/3619), [#3623](https://github.com/zarr-developers/zarr-python/issues/3623), [#3636](https://github.com/zarr-developers/zarr-python/issues/3636), [#3648](https://github.com/zarr-developers/zarr-python/issues/3648), [#3656](https://github.com/zarr-developers/zarr-python/issues/3656), [#3658](https://github.com/zarr-developers/zarr-python/issues/3658), [#3673](https://github.com/zarr-developers/zarr-python/issues/3673), [#3704](https://github.com/zarr-developers/zarr-python/issues/3704), [#3706](https://github.com/zarr-developers/zarr-python/issues/3706), [#3708](https://github.com/zarr-developers/zarr-python/issues/3708), [#3712](https://github.com/zarr-developers/zarr-python/issues/3712), [#3713](https://github.com/zarr-developers/zarr-python/issues/3713), [#3717](https://github.com/zarr-developers/zarr-python/issues/3717), [#3721](https://github.com/zarr-developers/zarr-python/issues/3721), [#3728](https://github.com/zarr-developers/zarr-python/issues/3728), [#3778](https://github.com/zarr-developers/zarr-python/issues/3778)


## 3.1.5 (2025-11-21)

### Bugfixes

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


## 3.1.3 (2025-09-18)

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
- Add a test for using Stores as context managers in `StoreTests`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Implemented `LoggingStore.open()`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- `LoggingStore` is now a generic class. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Change StoreTest's `test_store_repr`, `test_store_supports_writes`,
  `test_store_supports_partial_writes`, and `test_store_supports_listing`
  to be implemented using `@abstractmethod`, rather than raising `NotImplementedError`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Test the error raised for invalid buffer arguments in `StoreTests`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Test that data can be written to a store that's not yet open using the store.set method in `StoreTests`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Adds a new function `init_array` for initializing an array in storage, and refactors `create_array`
  to use `init_array`. `create_array` takes two new parameters: `data`, an optional array-like object, and `write_data`, a bool which defaults to `True`.
  If `data` is given to `create_array`, then the `dtype` and `shape` attributes of `data` are used to define the
  corresponding attributes of the resulting Zarr array. Additionally, if `data` is given and `write_data` is `True`,
  then the values in `data` will be written to the newly created array. ([#2761](https://github.com/zarr-developers/zarr-python/issues/2761))

### Bugfixes

- Wrap sync fsspec filesystems with `AsyncFileSystemWrapper`. ([#2533](https://github.com/zarr-developers/zarr-python/issues/2533))
- Added backwards compatibility for Zarr format 2 structured arrays. ([#2681](https://github.com/zarr-developers/zarr-python/issues/2681))
- Update equality for `LoggingStore` and `WrapperStore` such that 'other' must also be a `LoggingStore` or `WrapperStore` respectively, rather than only checking the types of the stores they wrap. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Ensure that `ZipStore` is open before getting or setting any values. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Use stdout rather than stderr as the default stream for `LoggingStore`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Match the errors raised by read only stores in `StoreTests`. ([#2693](https://github.com/zarr-developers/zarr-python/issues/2693))
- Fixed `ZipStore` to make sure the correct attributes are saved when instances are pickled.
  This fixes a previous bug that prevented using `ZipStore` with a `ProcessPoolExecutor`. ([#2762](https://github.com/zarr-developers/zarr-python/issues/2762))
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

## 3.0.1 (2025-01-17)

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

## 3.0.0 (2025-01-09)

3.0.0 is a new major release of Zarr-Python, with many breaking changes.
See the [v3 migration guide](user-guide/v3_migration.md) for a listing of what's changed.

Normal release note service will resume with further releases in the 3.0.0
series.

Release notes for the zarr-python 2.x and 1.x releases can be found here:
https://zarr.readthedocs.io/en/support-v2/release.html
