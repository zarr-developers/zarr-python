# Storage guide

Zarr-Python supports multiple storage backends, including: local file systems,
Zip files, remote stores via [fsspec](https://filesystem-spec.readthedocs.io) (S3, HTTP, etc.), and in-memory stores. In
Zarr-Python 3, stores must implement the abstract store API from
[`zarr.abc.store.Store`][].

!!! note
    Unlike Zarr-Python 2 where the store interface was built around a generic `MutableMapping`
    API, Zarr-Python 3 utilizes a custom store API that utilizes Python's AsyncIO library.

## Implicit Store Creation

In most cases, it is not required to create a `Store` object explicitly. Passing a string
(or other [StoreLike value](#storelike)) to Zarr's top level API will result in the store
being created automatically:

```python exec="true" session="storage" source="above" result="ansi"
import zarr

# Implicitly create a writable LocalStore
group = zarr.create_group(store='data/foo/bar')
print(group)
```

```python exec="true" session="storage" source="above" result="ansi"
# Implicitly create a read-only FsspecStore
# Note: requires s3fs to be installed
group = zarr.open_group(
   store='s3://noaa-nwm-retro-v2-zarr-pds',
   mode='r',
   storage_options={'anon': True}
)
print(group)
```

```python exec="true" session="storage" source="above" result="ansi"
# Implicitly creates a MemoryStore
data = {}
group = zarr.create_group(store=data)
print(group)
```

[](){#user-guide-store-like}
### StoreLike

`StoreLike` values can be:

- a `Path` or string indicating a location on the local file system.
  This will create a [local store](#local-store):
   ```python exec="true" session="storage" source="above" result="ansi"
   group = zarr.open_group(store='data/foo/bar')
   print(group)
   ```
   ```python exec="true" session="storage" source="above" result="ansi"
   from pathlib import Path
   group = zarr.open_group(store=Path('data/foo/bar'))
   print(group)
   ```

- an FSSpec URI string, indicating a [remote store](#remote-store) location:
   ```python exec="true" session="storage" source="above" result="ansi"
   # Note: requires s3fs to be installed
   group = zarr.open_group(
      store='s3://noaa-nwm-retro-v2-zarr-pds',
      mode='r',
      storage_options={'anon': True}
   )
   print(group)
   ```

- an empty dictionary or None, which will create a new [memory store](#memory-store):
   ```python exec="true" session="storage" source="above" result="ansi"
   group = zarr.create_group(store={})
   print(group)
   ```
   ```python exec="true" session="storage" source="above" result="ansi"
   group = zarr.create_group(store=None)
   print(group)
   ```

- a dictionary of string to [`Buffer`][zarr.abc.buffer.Buffer] mappings. This will
  create a [memory store](#memory-store), using this dictionary as the
  [`store_dict` argument][zarr.storage.MemoryStore].

- an FSSpec [FSMap object](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.FSMap),
  which will create an [FsspecStore](#remote-store).

- a [`Store`][zarr.abc.store.Store] or [`StorePath`][zarr.storage.StorePath] -
  see explicit store creation below.

## Explicit Store Creation

In some cases, it may be helpful to create a store instance directly. Zarr-Python offers four
built-in store: [`zarr.storage.LocalStore`][], [`zarr.storage.FsspecStore`][],
[`zarr.storage.ZipStore`][], [`zarr.storage.MemoryStore`][], and [`zarr.storage.ObjectStore`][].

### Local Store

The [`zarr.storage.LocalStore`][] stores data in a nested set of directories on a local
filesystem:

```python exec="true" session="storage" source="above" result="ansi"
store = zarr.storage.LocalStore('data/foo/bar', read_only=True)
group = zarr.open_group(store=store, mode='r')
print(group)
```

### Zip Store

The [`zarr.storage.ZipStore`][] stores the contents of a Zarr hierarchy in a single
Zip file. The [Zip Store specification](https://github.com/zarr-developers/zarr-specs/pull/311) is currently in draft form:

```python exec="true" session="storage" source="above" result="ansi"
store = zarr.storage.ZipStore('data.zip', mode='w')
array = zarr.create_array(store=store, shape=(2,), dtype='float64')
print(array)
```

### Remote Store

The [`zarr.storage.FsspecStore`][] stores the contents of a Zarr hierarchy in following the same
logical layout as the [`LocalStore`][zarr.storage.LocalStore], except the store is assumed to be on a remote storage system
such as cloud object storage (e.g. AWS S3, Google Cloud Storage, Azure Blob Store). The
[`zarr.storage.FsspecStore`][] is backed by [fsspec](https://filesystem-spec.readthedocs.io) and can support any backend
that implements the [AbstractFileSystem](https://filesystem-spec.readthedocs.io/en/stable/api.html#fsspec.spec.AbstractFileSystem)
API. `storage_options` can be used to configure the fsspec backend:

```python exec="true" session="storage" source="above" result="ansi"
# Note: requires s3fs to be installed
store = zarr.storage.FsspecStore.from_url(
   's3://noaa-nwm-retro-v2-zarr-pds',
   read_only=True,
   storage_options={'anon': True}
)
group = zarr.open_group(store=store, mode='r')
print(group)
```

The type of filesystem (e.g. S3, https, etc..) is inferred from the scheme of the url (e.g. s3 for "**s3**://noaa-nwm-retro-v2-zarr-pds").
In case a specific filesystem is needed, one can explicitly create it. For example to create an S3 filesystem:

```python exec="true" session="storage" source="above" result="ansi"
# Note: requires s3fs to be installed
import fsspec
fs = fsspec.filesystem(
   's3', anon=True, asynchronous=True,
   client_kwargs={'endpoint_url': "https://noaa-nwm-retro-v2-zarr-pds.s3.amazonaws.com"}
)
store = zarr.storage.FsspecStore(fs)
print(store)
```


### Memory Store

The [`zarr.storage.MemoryStore`][] an in-memory store that allows for serialization of
Zarr data (metadata and chunks) to a dictionary:

```python exec="true" session="storage" source="above" result="ansi"
data = {}
store = zarr.storage.MemoryStore(data)
array = zarr.create_array(store=store, shape=(2,), dtype='float64')
print(array)
```

### Object Store

[`zarr.storage.ObjectStore`][] stores the contents of the Zarr hierarchy using any ObjectStore
[storage implementation](https://developmentseed.org/obstore/latest/api/store/), including AWS S3 ([`obstore.store.S3Store`][]), Google Cloud Storage ([`obstore.store.GCSStore`][]), and Azure Blob Storage ([`obstore.store.AzureStore`][]). This store is backed by [obstore](https://developmentseed.org/obstore/latest/), which
builds on the production quality Rust library [object_store](https://docs.rs/object_store/latest/object_store/).

```python exec="true" session="storage" source="above" result="ansi"
from zarr.storage import ObjectStore
from obstore.store import MemoryStore

store = ObjectStore(MemoryStore())
array = zarr.create_array(store=store, shape=(2,), dtype='float64')
print(array)
```

Here's an example of using ObjectStore for accessing remote data:

```python exec="true" session="storage" source="above" result="ansi"
from zarr.storage import ObjectStore
from obstore.store import S3Store

s3_store = S3Store('noaa-nwm-retro-v2-zarr-pds', skip_signature=True, region="us-west-2")
store = zarr.storage.ObjectStore(store=s3_store, read_only=True)
group = zarr.open_group(store=store, mode='r')
print(group.info)
```

!!! warning
    The [`zarr.storage.ObjectStore`][] class is experimental.

## Developing custom stores

Zarr-Python [`zarr.abc.store.Store`][] API is meant to be extended. The Store Abstract Base
Class includes all of the methods needed to be a fully operational store in Zarr Python.
Zarr also provides a test harness for custom stores: [`zarr.testing.store.StoreTests`][].
