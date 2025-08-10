This section  will help you get up and running with
the Zarr library in Python to efficiently manage and analyze multi-dimensional arrays.

### Creating an Array

To get started, you can create a simple Zarr array:

```python exec="true" session="quickstart"
import shutil
shutil.rmtree('data', ignore_errors=True)
import numpy as np
from pprint import pprint
import io

np.random.seed(0)
```

```python exec="true" session="quickstart" source="above" result="ansi"
import zarr
import numpy as np

# Create a 2D Zarr array
z = zarr.create_array(
    store="data/example-1.zarr",
    shape=(100, 100),
    chunks=(10, 10),
    dtype="f4"
)

# Assign data to the array
z[:, :] = np.random.random((100, 100))
print(z.info)
```

Here, we created a 2D array of shape `(100, 100)`, chunked into blocks of
`(10, 10)`, and filled it with random floating-point data. This array was
written to a `LocalStore` in the `data/example-1.zarr` directory.

#### Compression and Filters

Zarr supports data compression and filters. For example, to use Blosc compression:


```python exec="true" session="quickstart" source="above" result="code"

# Create a 2D Zarr array with Blosc compression
z = zarr.create_array(
    store="data/example-2.zarr",
    shape=(100, 100),
    chunks=(10, 10),
    dtype="f4",
    compressors=zarr.codecs.BloscCodec(
        cname="zstd",
        clevel=3,
        shuffle=zarr.codecs.BloscShuffle.shuffle
    )
)

# Assign data to the array
z[:, :] = np.random.random((100, 100))
print(z.info)
```

This compresses the data using the Blosc codec with shuffle enabled for better compression.


### Hierarchical Groups

Zarr allows you to create hierarchical groups, similar to directories:

```python exec="true" session="quickstart" source="above" result="ansi"

# Create nested groups and add arrays
root = zarr.group("data/example-3.zarr")
foo = root.create_group(name="foo")
bar = root.create_array(
    name="bar", shape=(100, 10), chunks=(10, 10), dtype="f4"
)
spam = foo.create_array(name="spam", shape=(10,), dtype="i4")

# Assign values
bar[:, :] = np.random.random((100, 10))
spam[:] = np.arange(10)

# print the hierarchy
print(root.tree())
```

This creates a group with two datasets: `foo` and `bar`.

#### Batch Hierarchy Creation

Zarr provides tools for creating a collection of arrays and groups with a single function call.
Suppose we want to copy existing groups and arrays into a new storage backend:

```python exec="true" session="quickstart" source="above" result="html"

# Create nested groups and add arrays
root = zarr.group("data/example-4.zarr", attributes={'name': 'root'})
foo = root.create_group(name="foo")
bar = root.create_array(
    name="bar", shape=(100, 10), chunks=(10, 10), dtype="f4"
)
nodes = {'': root.metadata} | {k: v.metadata for k,v in root.members()}
# Report nodes
output = io.StringIO()
pprint(nodes, stream=output, width=60, depth=3)
result = output.getvalue()
print(result)
# Create new hierarchy from nodes
new_nodes = dict(zarr.create_hierarchy(store=zarr.storage.MemoryStore(), nodes=nodes))
new_root = new_nodes['']
assert new_root.attrs == root.attrs
```

Note that `zarr.create_hierarchy` will only initialize arrays and groups -- copying array data must
be done in a separate step.

### Persistent Storage

Zarr supports persistent storage to disk or cloud-compatible backends. While examples above
utilized a `zarr.storage.LocalStore`, a number of other storage options are available.

Zarr integrates seamlessly with cloud object storage such as Amazon S3 and Google Cloud Storage
using external libraries like [s3fs](https://s3fs.readthedocs.io) or
[gcsfs](https://gcsfs.readthedocs.io):

```python

import s3fs

z = zarr.create_array("s3://example-bucket/foo", mode="w", shape=(100, 100), chunks=(10, 10), dtype="f4")
z[:, :] = np.random.random((100, 100))
```

A single-file store can also be created using the `zarr.storage.ZipStore`:

```python exec="true" session="quickstart" source="above"

# Store the array in a ZIP file
store = zarr.storage.ZipStore("data/example-5.zip", mode="w")

z = zarr.create_array(
    store=store,
    shape=(100, 100),
    chunks=(10, 10),
    dtype="f4"
)

# write to the array
z[:, :] = np.random.random((100, 100))

# the ZipStore must be explicitly closed
store.close()
```

To open an existing array from a ZIP file:

```python exec="true" session="quickstart" source="above" result="code"

# Open the ZipStore in read-only mode
store = zarr.storage.ZipStore("data/example-5.zip", read_only=True)

z = zarr.open_array(store, mode='r')

# read the data as a NumPy Array
print(z[:])
```

Read more about Zarr's storage options in the [User Guide](user-guide/storage.md).
