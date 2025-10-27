# 3.0 Migration Guide

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
open a [GitHub issue](https://github.com/zarr-developers/zarr-python/issues/new)
so we can improve this guide.

## Compatibility target

The goals described above necessitated some breaking changes to the API (hence the
major version update), but where possible we have maintained backwards compatibility
in the most widely used parts of the API. This in the [`zarr.Array`][] and
[`zarr.Group`][] classes and the "top-level API" (e.g. [`zarr.open_array`][] and
[`zarr.open_group`][]).

## Getting ready for 3.0

Before migrating to Zarr-Python 3, we suggest projects that depend on Zarr-Python take
the following actions in order:

1. Pin the supported Zarr-Python version to `zarr>=2,<3`. This is a best practice
   and will protect your users from any incompatibilities that may arise during the
   release of Zarr-Python 3. This pin can be removed after migrating to Zarr-Python 3.
2. Limit your imports from the Zarr-Python package. Most of the primary API `zarr.*`
   will be compatible in Zarr-Python 3. However, the following breaking API changes are
   planned:

   - `numcodecs.*` will no longer be available in `zarr.*`. To migrate, import codecs
     directly from `numcodecs`:

     ```python
     from numcodecs import Blosc
     # instead of:
     # from zarr import Blosc
     ```

   - The `zarr.v3_api_available` feature flag is being removed. In Zarr-Python 3
     the v3 API is always available, so you shouldn't need to use this flag.
   - The following internal modules are being removed or significantly changed. If
     your application relies on imports from any of the below modules, you will need
     to either a) modify your application to no longer rely on these imports or b)
     vendor the parts of the specific modules that you need.

     * `zarr.attrs` has gone, with no replacement
     * `zarr.codecs` has changed, see "Codecs" section below for more information
     * `zarr.context` has gone, with no replacement
     * `zarr.core` remains but should be considered private API
     * `zarr.hierarchy` has gone, with no replacement (use `zarr.Group` inplace of `zarr.hierarchy.Group`)
     * `zarr.indexing` has gone, with no replacement
     * `zarr.meta` has gone, with no replacement
     * `zarr.meta_v1` has gone, with no replacement
     * `zarr.sync` has gone, with no replacement
     * `zarr.types` has gone, with no replacement
     * `zarr.util` has gone, with no replacement
     * `zarr.n5` has gone, see below for an alternative N5 options

3. Test that your package works with version 3.
4. Update the pin to include `zarr>=3,<4`.

## Zarr-Python 2 support window

Zarr-Python 2.x is still available, though we recommend migrating to Zarr-Python 3 for
its performance improvements and new features. Security and bug fixes will be made to
the 2.x series for at least six months following the first Zarr-Python 3 release.
If you need to use the latest Zarr-Python 2 release, you can install it with:

```console
$ pip install "zarr==2.*"
```

!!! note
    Development and maintenance of the 2.x release series has moved to the
    [support/v2](https://github.com/zarr-developers/zarr-python/tree/support/v2) branch.
    Issues and pull requests related to this branch are tagged with the
    [V2](https://github.com/zarr-developers/zarr-python/labels/V2) label.

## Migrating to Zarr-Python 3

The following sections provide details on breaking changes in Zarr-Python 3.

### The Array class

1. Disallow direct construction - the signature for initializing the `Array` class has changed
   significantly. Please use [`zarr.create_array`][] or [`zarr.open_array`][] instead of
   directly constructing the [`zarr.Array`][] class.

2. Defaulting to `zarr_format=3` - newly created arrays will use the version 3 of the
   Zarr specification. To continue using version 2, set `zarr_format=2` when creating arrays
   or set `default_zarr_version=2` in Zarr's runtime configuration.

3. Function signature change to [`zarr.Array.resize`][] - the `resize` function now takes a
   `zarr.core.common.ShapeLike` input rather than separate arguments for each dimension.
   Use `resize((10,10))` in place of `resize(10,10)`.

### The Group class

1. Disallow direct construction - use [`zarr.open_group`][] or [`zarr.create_group`][]
   instead of directly constructing the `zarr.Group` class.
2. Most of the h5py compatibility methods are deprecated and will issue warnings if used.
   The following functions are drop in replacements that have the same signature and functionality:

   - Use [`zarr.Group.create_array`][] in place of `zarr.Group.create_dataset`
   - Use [`zarr.Group.require_array`][] in place of `zarr.Group.require_dataset`
3. Disallow "." syntax for getting group members. To get a member of a group named `foo`,
   use `group["foo"]` in place of `group.foo`.

### The Store class

The Store API has changed significant in Zarr-Python 3.

#### The base store class

The `MutableMapping` base class has been replaced in favor of a custom abstract base class ([`zarr.abc.store.Store`][]).
An asynchronous interface is used for all store methods that use I/O.
This change ensures that all store methods are non-blocking and are as performant as possible.

#### Store implementations

Store implementations have moved from the top-level module to `zarr.storage`:

```diff title="Store import changes from v2 to v3"
# Before (v2)
- from zarr import MemoryStore
+ from zarr.storage import MemoryStore
```

The following stores have been renamed or changed:

| v2                     | v3                                 |
|------------------------|------------------------------------|
| `DirectoryStore`   | [`zarr.storage.LocalStore`][]          |
| `FSStore`          | [`zarr.storage.FsspecStore`][]         |
| `TempStore`        | Use [`tempfile.TemporaryDirectory`][] with [`LocalStore`][zarr.storage.LocalStore]  |
| `zarr.


A number of deprecated stores were also removed.
See issue #1274 for more details on the removal of these stores.

- `N5Store` - see https://github.com/zarr-developers/n5py for an alternative interface to
  N5 formatted data.
- `ABSStore` - use the [`zarr.storage.FsspecStore`][] instead along with fsspec's
  [adlfs backend](https://github.com/fsspec/adlfs).
- `DBMStore`
- `LMDBStore`
- `SQLiteStore`
- `MongoDBStore`
- `RedisStore`

The latter five stores in this list do not have an equivalent in Zarr-Python 3.
If you are interested in developing a custom store that targets these backends, see
[developing custom stores](storage.md/#developing-custom-stores) or open an
[issue](https://github.com/zarr-developers/zarr-python/issues) to discuss your use case.

### Codecs

Codecs defined in ``numcodecs`` (and also imported into the ``zarr.codecs`` namespace in Zarr-Python 2)
should still be used when creating Zarr format 2 arrays.

Codecs for creating Zarr format 3 arrays are available in two locations:

- `zarr.codecs` contains Zarr format 3 codecs that are defined in the [codecs section of the Zarr format 3 specification](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/index.html).
- `numcodecs.zarr3` contains codecs from `numcodecs` that can be used to create Zarr format 3 arrays, but are not necessarily part of the Zarr format 3 specification.

### Dependencies

When installing using `pip`:

- The new `remote` dependency group can be used to install a supported version of
  `fsspec`, required for remote data access.
- The new `gpu` dependency group can be used to install a supported version of
  `cuda`, required for GPU functionality.
- The `jupyter` optional dependency group has been removed, since v3 contains no
  jupyter specific functionality.

### Miscellaneous

- The keyword argument `zarr_version` available in most creation functions in `zarr`
  (e.g. [`zarr.create`][], [`zarr.open`][], [`zarr.group`][], [`zarr.array`][]) has
  been deprecated in favor of `zarr_format`.

## 🚧 Work in Progress 🚧

Zarr-Python 3 is still under active development, and is not yet fully complete.
The following list summarizes areas of the codebase that we expect to build out
after the 3.0.0 release. If features listed below are important to your use case
of Zarr-Python, please open (or comment on) a
[GitHub issue](https://github.com/zarr-developers/zarr-python/issues/new).

- The following functions / methods have not been ported to Zarr-Python 3 yet:

  * `zarr.copy` ([issue #2407](https://github.com/zarr-developers/zarr-python/issues/2407))
  * `zarr.copy_all` ([issue #2407](https://github.com/zarr-developers/zarr-python/issues/2407))
  * `zarr.copy_store` ([issue #2407](https://github.com/zarr-developers/zarr-python/issues/2407))
  * `zarr.Group.move` ([issue #2108](https://github.com/zarr-developers/zarr-python/issues/2108))

- The following features (corresponding to function arguments to functions in
  `zarr`) have not been ported to Zarr-Python 3 yet. Using these features
  will raise a warning or a `NotImplementedError`:

  * `cache_attrs`
  * `cache_metadata`
  * `chunk_store` ([issue #2495](https://github.com/zarr-developers/zarr-python/issues/2495))
  * `meta_array`
  * `object_codec` ([issue #2617](https://github.com/zarr-developers/zarr-python/issues/2617))
  * `synchronizer` ([issue #1596](https://github.com/zarr-developers/zarr-python/issues/1596))
  * `dimension_separator`

- The following features that were supported by Zarr-Python 2 have not been ported
  to Zarr-Python 3 yet:

  * Structured arrays / dtypes ([issue #2134](https://github.com/zarr-developers/zarr-python/issues/2134))
  * Fixed-length string dtypes ([issue #2347](https://github.com/zarr-developers/zarr-python/issues/2347))
  * Datetime and timedelta dtypes ([issue #2616](https://github.com/zarr-developers/zarr-python/issues/2616))
  * Object dtypes ([issue #2616](https://github.com/zarr-developers/zarr-python/issues/2616))
  * Ragged arrays ([issue #2618](https://github.com/zarr-developers/zarr-python/issues/2618))
  * Groups and Arrays do not implement `__enter__` and `__exit__` protocols ([issue #2619](https://github.com/zarr-developers/zarr-python/issues/2619))
  * Default filters for object dtypes for Zarr format 2 arrays ([issue #2627](https://github.com/zarr-developers/zarr-python/issues/2627))
