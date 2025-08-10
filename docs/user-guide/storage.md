# Storage guide

Zarr-Python supports multiple storage backends, including: local file systems,
Zip files, remote stores via [fsspec](https://filesystem-spec.readthedocs.io) (S3, HTTP, etc.), and in-memory stores. In
Zarr-Python 3, stores must implement the abstract store API from
`zarr.abc.store.Store`.

!!! note
    Unlike Zarr-Python 2 where the store interface was built around a generic `MutableMapping`
    API, Zarr-Python 3 utilizes a custom store API that utilizes Python's AsyncIO library.

## Implicit Store Creation

In most cases, it is not required to create a `Store` object explicitly. Passing a string
to Zarr's top level API will result in the store being created automatically.:

```python
import zarr

# Implicitly create a writable LocalStore
zarr.create_group(store='data/foo/bar')
# <Group file://data/foo/bar>

# Implicitly create a read-only FsspecStore
zarr.open_group(
   store='s3://noaa-nwm-retro-v2-zarr-pds',
   mode='r',
   storage_options={'anon': True}
)
# <Group <FsspecStore(S3FileSystem, noaa-nwm-retro-v2-zarr-pds)>>

# Implicitly creates a MemoryStore
data = {}
zarr.create_group(store=data)
# <Group memory://...>
```

## Explicit Store Creation

In some cases, it may be helpful to create a store instance directly. Zarr-Python offers four
built-in store: `zarr.storage.LocalStore`, `zarr.storage.FsspecStore`,
`zarr.storage.ZipStore`, `zarr.storage.MemoryStore`, and `zarr.storage.ObjectStore`.

### Local Store

The `zarr.storage.LocalStore` stores data in a nested set of directories on a local
filesystem.:

```python
store = zarr.storage.LocalStore('data/foo/bar', read_only=True)
zarr.open_group(store=store, mode='r')
# <Group file://data/foo/bar>
```

### Zip Store

The `zarr.storage.ZipStore` stores the contents of a Zarr hierarchy in a single
Zip file. The [Zip Store specification](https://github.com/zarr-developers/zarr-specs/pull/311) is currently in draft form.:

```python
store = zarr.storage.ZipStore('data.zip', mode='w')
zarr.create_array(store=store, shape=(2,), dtype='float64')
# <Array zip://data.zip shape=(2,) dtype=float64>
```

### Remote Store

The `zarr.storage.FsspecStore` stores the contents of a Zarr hierarchy in following the same
logical layout as the `LocalStore`, except the store is assumed to be on a remote storage system
such as cloud object storage (e.g. AWS S3, Google Cloud Storage, Azure Blob Store). The
`zarr.storage.FsspecStore` is backed by [fsspec](https://filesystem-spec.readthedocs.io) and can support any backend
that implements the [AbstractFileSystem](https://filesystem-spec.readthedocs.io/en/stable/api.html#fsspec.spec.AbstractFileSystem)
API. `storage_options` can be used to configure the fsspec backend.:

```python
store = zarr.storage.FsspecStore.from_url(
   's3://noaa-nwm-retro-v2-zarr-pds',
   read_only=True,
   storage_options={'anon': True}
)
zarr.open_group(store=store, mode='r')
# <Group <FsspecStore(S3FileSystem, noaa-nwm-retro-v2-zarr-pds)>>
```

The type of filesystem (e.g. S3, https, etc..) is inferred from the scheme of the url (e.g. s3 for "**s3**://noaa-nwm-retro-v2-zarr-pds").
In case a specific filesystem is needed, one can explicitly create it. For example to create a S3 filesystem:

```python
import fsspec
fs = fsspec.filesystem(
   's3', anon=True, asynchronous=True,
   client_kwargs={'endpoint_url': "https://noaa-nwm-retro-v2-zarr-pds.s3.amazonaws.com"}
)
store = zarr.storage.FsspecStore(fs)
```


### Memory Store

The `zarr.storage.MemoryStore` a in-memory store that allows for serialization of
Zarr data (metadata and chunks) to a dictionary.:

```python
data = {}
store = zarr.storage.MemoryStore(data)
# TODO: replace with create_array after #2463
zarr.create_array(store=store, shape=(2,), dtype='float64')
# <Array memory://... shape=(2,) dtype=float64>
```

### Object Store

`zarr.storage.ObjectStore` stores the contents of the Zarr hierarchy using any ObjectStore
[storage implementation](https://developmentseed.org/obstore/latest/api/store/), including AWS S3 (`obstore.store.S3Store`), Google Cloud Storage (`obstore.store.GCSStore`), and Azure Blob Storage (`obstore.store.AzureStore`). This store is backed by [obstore](https://developmentseed.org/obstore/latest/), which
builds on the production quality Rust library [object_store](https://docs.rs/object_store/latest/object_store/).

```python
from zarr.storage import ObjectStore
from obstore.store import MemoryStore

store = ObjectStore(MemoryStore())
zarr.create_array(store=store, shape=(2,), dtype='float64')
# <Array object_store://... shape=(2,) dtype=float64>
```

Here's an example of using ObjectStore for accessing remote data:

```python
from zarr.storage import ObjectStore
from obstore.store import S3Store

s3_store = S3Store('noaa-nwm-retro-v2-zarr-pds', skip_signature=True, region="us-west-2")
store = zarr.storage.ObjectStore(store=s3_store, read_only=True)
group = zarr.open_group(store=store, mode='r')
group.info
# Name        :
# Type        : Group
# Zarr format : 2
# Read-only   : True
# Store type  : ObjectStore
# No. members : 12
# No. arrays  : 12
# No. groups  : 0
```

!!! warning
    The `zarr.storage.ObjectStore` class is experimental.

## Developing custom stores

Zarr-Python `zarr.abc.store.Store` API is meant to be extended. The Store Abstract Base
Class includes all of the methods needed to be a fully operational store in Zarr Python.
Zarr also provides a test harness for custom stores: `zarr.testing.store.StoreTests`.
