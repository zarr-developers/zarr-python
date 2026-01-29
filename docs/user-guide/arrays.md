# Working with arrays

## Creating an array

Zarr has several functions for creating arrays. For example:

```python exec="true" session="arrays"
import shutil
shutil.rmtree('data', ignore_errors=True)
import numpy as np

np.random.seed(0)
```

```python exec="true" session="arrays" source="above" result="ansi"
import zarr
z = zarr.create_array(store="memory://arrays-demo", shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
print(z)
```

The code above creates a 2-dimensional array of 32-bit integers with 10000 rows
and 10000 columns, divided into chunks where each chunk has 1000 rows and 1000
columns (and so there will be 100 chunks in total). The data is written to an
in-memory store (see [`zarr.storage.MemoryStore`][] for more details). See
[Persistent arrays](#persistent-arrays) for details on storing arrays in other stores,
and see [Data types](data_types.md) for an in-depth look at the data types supported
by Zarr.

See the [creation API documentation](../api/zarr/create.md) for more detailed information about
creating arrays.

## Reading and writing data

Zarr arrays support a similar interface to [NumPy](https://numpy.org/doc/stable/)
arrays for reading and writing data. For example, the entire array can be filled
with a scalar value:

```python exec="true" session="arrays" source="above"
z[:] = 42
```

Regions of the array can also be written to, e.g.:

```python exec="true" session="arrays" source="above"
import numpy as np

z[0, :] = np.arange(10000)
z[:, 0] = np.arange(10000)
```

The contents of the array can be retrieved by slicing, which will load the
requested region into memory as a NumPy array, e.g.:

```python exec="true" session="arrays" source="above" result="ansi"
print(z[0, 0])
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z[-1, -1])
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z[0, :])
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z[:, 0])
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z[:])
```

Read more about NumPy-style indexing can be found in the
[NumPy documentation](https://numpy.org/doc/stable/user/basics.indexing.html).

## Persistent arrays

In the examples above, compressed data for each chunk of the array was stored in
main memory. Zarr arrays can also be stored on a file system, enabling
persistence of data between sessions. To do this, we can change the store
argument to point to a filesystem path:

```python exec="true" session="arrays" source="above"
z1 = zarr.create_array(store='data/example-1.zarr', shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
```

The array above will store its configuration metadata and all compressed chunk
data in a directory called `'data/example-1.zarr'` relative to the current working
directory. The [`zarr.create_array`][] function provides a convenient way
to create a new persistent array or continue working with an existing
array. Note, there is no need to close an array: data are automatically
flushed to disk, and files are automatically closed whenever an array is modified.

Persistent arrays support the same interface for reading and writing data,
e.g.:

```python exec="true" session="arrays" source="above"
z1[:] = 42
z1[0, :] = np.arange(10000)
z1[:, 0] = np.arange(10000)
```

Check that the data have been written and can be read again:

```python exec="true" session="arrays" source="above" result="ansi"
z2 = zarr.open_array('data/example-1.zarr', mode='r')
print(np.all(z1[:] == z2[:]))
```

If you are just looking for a fast and convenient way to save NumPy arrays to
disk then load back into memory later, the functions
[`zarr.save`][] and [`zarr.load`][] may be
useful. E.g.:

```python exec="true" session="arrays" source="above" result="ansi"
a = np.arange(10)
zarr.save('data/example-2.zarr', a)
print(zarr.load('data/example-2.zarr'))
```

Please note that there are a number of other options for persistent array
storage, see the [Storage Guide](storage.md) for more details.

## Resizing and appending

A Zarr array can be resized, which means that any of its dimensions can be
increased or decreased in length. For example:

```python exec="true" session="arrays" source="above" result="ansi"
z = zarr.create_array(store='data/example-3.zarr', shape=(10000, 10000), dtype='int32',chunks=(1000, 1000))
z[:] = 42
print(f"Original shape: {z.shape}")
z.resize((20000, 10000))
print(f"New shape: {z.shape}")
```

Note that when an array is resized, the underlying data are not rearranged in
any way. If one or more dimensions are shrunk, any chunks falling outside the
new array shape will be deleted from the underlying store.

[`zarr.Array.append`][] is provided as a convenience function, which can be
used to append data to any axis. E.g.:

```python exec="true" session="arrays" source="above" result="ansi"
a = np.arange(10000000, dtype='int32').reshape(10000, 1000)
z = zarr.create_array(store='data/example-4.zarr', shape=a.shape, dtype=a.dtype, chunks=(1000, 100))
z[:] = a
print(f"Original shape: {z.shape}")
z.append(a)
print(f"Shape after first append: {z.shape}")
z.append(np.vstack([a, a]), axis=1)
print(f"Shape after second append: {z.shape}")
```

## Compressors

A number of different compressors can be used with Zarr. Zarr includes Blosc,
Zstandard and Gzip compressors. Additional compressors are available through
a separate package called [NumCodecs](https://numcodecs.readthedocs.io/) which provides various
compressor libraries including LZ4, Zlib, BZ2 and LZMA.
Different compressors can be provided via the `compressors` keyword
argument accepted by all array creation functions. For example:

```python exec="true" session="arrays" source="above" result="ansi"
compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
data = np.arange(100000000, dtype='int32').reshape(10000, 10000)
z = zarr.create_array(store='data/example-5.zarr', shape=data.shape, dtype=data.dtype, chunks=(1000, 1000), compressors=compressors)
z[:] = data
print(z.compressors)
```

This array above will use Blosc as the primary compressor, using the Zstandard
algorithm (compression level 3) internally within Blosc, and with the
bit-shuffle filter applied.

When using a compressor, it can be useful to get some diagnostics on the
compression ratio. Zarr arrays provide the [`zarr.Array.info`][] property
which can be used to print useful diagnostics, e.g.:

```python exec="true" session="arrays" source="above" result="ansi"
print(z.info)
```

The [`zarr.Array.info_complete`][] method inspects the underlying store and
prints additional diagnostics, e.g.:

```python exec="true" session="arrays" source="above" result="ansi"
print(z.info_complete())
```

!!! note
    [`zarr.Array.info_complete`][] will inspect the underlying store and may
    be slow for large arrays. Use [`zarr.Array.info`][] if detailed storage
    statistics are not needed.

If you don't specify a compressor, by default Zarr uses the Zstandard
compressor.

In addition to Blosc and Zstandard, other compression libraries can also be used. For example,
here is an array using Gzip compression, level 1:

```python exec="true" session="arrays" source="above" result="ansi"
data = np.arange(100000000, dtype='int32').reshape(10000, 10000)
z = zarr.create_array(store='data/example-6.zarr', shape=data.shape, dtype=data.dtype, chunks=(1000, 1000), compressors=zarr.codecs.GzipCodec(level=1))
z[:] = data
print(f"Compressors: {z.compressors}")
```

Here is an example using LZMA from [NumCodecs](https://numcodecs.readthedocs.io/) with a custom filter pipeline including LZMA's
built-in delta filter:

```python exec="true" session="arrays" source="above" result="ansi"
import lzma
from zarr.codecs.numcodecs import LZMA

lzma_filters = [dict(id=lzma.FILTER_DELTA, dist=4), dict(id=lzma.FILTER_LZMA2, preset=1)]
compressors = LZMA(filters=lzma_filters)
data = np.arange(100000000, dtype='int32').reshape(10000, 10000)
z = zarr.create_array(store='data/example-7.zarr', shape=data.shape, dtype=data.dtype, chunks=(1000, 1000), compressors=compressors)
print(f"Compressors: {z.compressors}")
```

To disable compression, set `compressors=None` when creating an array, e.g.:

```python exec="true" session="arrays" source="above" result="ansi"
z = zarr.create_array(
    store='data/example-8.zarr',
    shape=(100000000,),
    chunks=(1000000,),
    dtype='int32',
    compressors=None
)
print(f"Compressors: {z.compressors}")
```

## Filters

In some cases, compression can be improved by transforming the data in some
way. For example, if nearby values tend to be correlated, then shuffling the
bytes within each numerical value or storing the difference between adjacent
values may increase compression ratio. Some compressors provide built-in filters
that apply transformations to the data prior to compression. For example, the
Blosc compressor has built-in implementations of byte- and bit-shuffle filters,
and the LZMA compressor has a built-in implementation of a delta
filter. However, to provide additional flexibility for implementing and using
filters in combination with different compressors, Zarr also provides a
mechanism for configuring filters outside of the primary compressor.

Here is an example using a delta filter with the Blosc compressor:

```python exec="true" session="arrays" source="above" result="ansi"
from zarr.codecs.numcodecs import Delta

filters = [Delta(dtype='int32')]
compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=1, shuffle=zarr.codecs.BloscShuffle.shuffle)
data = np.arange(100000000, dtype='int32').reshape(10000, 10000)
z = zarr.create_array(store='data/example-9.zarr', shape=data.shape, dtype=data.dtype, chunks=(1000, 1000), filters=filters, compressors=compressors)
print(z.info_complete())
```

For more information about available filter codecs, see the [Numcodecs](https://numcodecs.readthedocs.io/) documentation.

## Advanced indexing

Zarr arrays support several methods for advanced or "fancy"
indexing, which enable a subset of data items to be extracted or updated in an
array without loading the entire array into memory.

Note that although this functionality is similar to some of the advanced
indexing capabilities available on NumPy arrays and on h5py datasets, **the Zarr
API for advanced indexing is different from both NumPy and h5py**, so please
read this section carefully.  For a complete description of the indexing API,
see the documentation for the [`zarr.Array`][] class.

### Indexing with coordinate arrays

Items from a Zarr array can be extracted by providing an integer array of
coordinates. E.g.:

```python exec="true" session="arrays" source="above" result="ansi"
data = np.arange(10) ** 2
z = zarr.create_array(store='data/example-10.zarr', shape=data.shape, dtype=data.dtype)
z[:] = data
print(z[:])
print(z.get_coordinate_selection([2, 5]))
```

Coordinate arrays can also be used to update data, e.g.:

```python exec="true" session="arrays" source="above" result="ansi"
z.set_coordinate_selection([2, 5], [-1, -2])
print(z[:])
```

For multidimensional arrays, coordinates must be provided for each dimension,
e.g.:

```python exec="true" session="arrays" source="above" result="ansi"
data = np.arange(15).reshape(3, 5)
z = zarr.create_array(store='data/example-11.zarr', shape=data.shape, dtype=data.dtype)
z[:] = data
print(z[:])
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z.get_coordinate_selection(([0, 2], [1, 3])))
```

```python exec="true" session="arrays" source="above" result="ansi"
z.set_coordinate_selection(([0, 2], [1, 3]), [-1, -2])
print(z[:])
```

For convenience, coordinate indexing is also available via the `vindex`
property, as well as the square bracket operator, e.g.:

```python exec="true" session="arrays" source="above" result="ansi"
print(z.vindex[[0, 2], [1, 3]])
z.vindex[[0, 2], [1, 3]] = [-3, -4]
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z[:])
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z[[0, 2], [1, 3]])
```

When the indexing arrays have different shapes, they are broadcast together.
That is, the following two calls are equivalent:

```python exec="true" session="arrays" source="above" result="ansi"
print(z[1, [1, 3]])
print(z[[1, 1], [1, 3]])
```

### Indexing with a mask array

Items can also be extracted by providing a Boolean mask. E.g.:

```python exec="true" session="arrays" source="above" result="ansi"
data = np.arange(10) ** 2
z = zarr.create_array(store='data/example-12.zarr', shape=data.shape, dtype=data.dtype)
z[:] = data
print(z[:])
```

```python exec="true" session="arrays" source="above" result="ansi"
sel = np.zeros_like(z, dtype=bool)
sel[2] = True
sel[5] = True
print(z.get_mask_selection(sel))
```

```python exec="true" session="arrays" source="above" result="ansi"
z.set_mask_selection(sel, [-1, -2])
print(z[:])
```

Here's a multidimensional example:

```python exec="true" session="arrays" source="above" result="ansi"
data = np.arange(15).reshape(3, 5)
z = zarr.create_array(store='data/example-13.zarr', shape=data.shape, dtype=data.dtype)
z[:] = data
print(z[:])
```

```python exec="true" session="arrays" source="above" result="ansi"
sel = np.zeros_like(z, dtype=bool)
sel[0, 1] = True
sel[2, 3] = True
print(z.get_mask_selection(sel))
```

```python exec="true" session="arrays" source="above" result="ansi"
z.set_mask_selection(sel, [-1, -2])
print(z[:])
```

For convenience, mask indexing is also available via the `vindex` property,
e.g.:

```python exec="true" session="arrays" source="above" result="ansi"
print(z.vindex[sel])
```

```python exec="true" session="arrays" source="above" result="ansi"

z.vindex[sel] = [-3, -4]
print(z[:])
```

Mask indexing is conceptually the same as coordinate indexing, and is
implemented internally via the same machinery. Both styles of indexing allow
selecting arbitrary items from an array, also known as point selection.

### Orthogonal indexing

Zarr arrays also support methods for orthogonal indexing, which allows
selections to be made along each dimension of an array independently. For
example, this allows selecting a subset of rows and/or columns from a
2-dimensional array. E.g.:

```python exec="true" session="arrays" source="above" result="ansi"
data = np.arange(15).reshape(3, 5)
z = zarr.create_array(store='data/example-14.zarr', shape=data.shape, dtype=data.dtype)
z[:] = data
print(z[:])
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z.get_orthogonal_selection(([0, 2], slice(None))))  # select first and third rows
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z.get_orthogonal_selection((slice(None), [1, 3])))  # select second and fourth columns)
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z.get_orthogonal_selection(([0, 2], [1, 3])))  # select rows [0, 2] and columns [1, 4]
```

Data can also be modified, e.g.:

```python exec="true" session="arrays" source="above"
z.set_orthogonal_selection(([0, 2], [1, 3]), [[-1, -2], [-3, -4]])
```

For convenience, the orthogonal indexing functionality is also available via the
`oindex` property, e.g.:

```python exec="true" session="arrays" source="above" result="ansi"
data = np.arange(15).reshape(3, 5)
z = zarr.create_array(store='data/example-15.zarr', shape=data.shape, dtype=data.dtype)
z[:] = data
print(z.oindex[[0, 2], :])  # select first and third rows
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z.oindex[:, [1, 3]])  # select second and fourth columns
```

```python exec="true" session="arrays" source="above" result="ansi"
print(z.oindex[[0, 2], [1, 3]])  # select rows [0, 2] and columns [1, 4]
```

```python exec="true" session="arrays" source="above" result="ansi"
z.oindex[[0, 2], [1, 3]] = [[-1, -2], [-3, -4]]
print(z[:])
```

Any combination of integer, slice, 1D integer array and/or 1D Boolean array can
be used for orthogonal indexing.

If the index contains at most one iterable, and otherwise contains only slices and integers,
orthogonal indexing is also available directly on the array:

```python exec="true" session="arrays" source="above" result="ansi"
data = np.arange(15).reshape(3, 5)
z = zarr.create_array(store='data/example-16.zarr', shape=data.shape, dtype=data.dtype)
z[:] = data
print(np.all(z.oindex[[0, 2], :] == z[[0, 2], :]))
```

### Block Indexing

Zarr also support block indexing, which allows selections of whole chunks based on their
logical indices along each dimension of an array. For example, this allows selecting
a subset of chunk aligned rows and/or columns from a 2-dimensional array. E.g.:

```python exec="true" session="arrays" source="above"
data = np.arange(100).reshape(10, 10)
z = zarr.create_array(store='data/example-17.zarr', shape=data.shape, dtype=data.dtype, chunks=(3, 3))
z[:] = data
```

Retrieve items by specifying their block coordinates:

```python exec="true" session="arrays" source="above" result="ansi"
print(z.get_block_selection(1))
```

Equivalent slicing:

```python exec="true" session="arrays" source="above" result="ansi"
print(z[3:6])
```

For convenience, the block selection functionality is also available via the
`blocks` property, e.g.:

```python exec="true" session="arrays" source="above" result="ansi"
print(z.blocks[1])
```

Block index arrays may be multidimensional to index multidimensional arrays.
For example:

```python exec="true" session="arrays" source="above" result="ansi"
print(z.blocks[0, 1:3])
```

Data can also be modified. Let's start by a simple 2D array:

```python exec="true" session="arrays" source="above"
z = zarr.create_array(store='data/example-18.zarr', shape=(6, 6), dtype=int, chunks=(2, 2))
```

Set data for a selection of items:

```python exec="true" session="arrays" source="above" result="ansi"
z.set_block_selection((1, 0), 1)
print(z[...])
```

For convenience, this functionality is also available via the `blocks` property.
E.g.:

```python exec="true" session="arrays" source="above" result="ansi"
z.blocks[:, 2] = 7
print(z[...])
```

Any combination of integer and slice can be used for block indexing:

```python exec="true" session="arrays" source="above" result="ansi"
print(z.blocks[2, 1:3])
```

```python exec="true" session="arrays" source="above" result="ansi"
root = zarr.create_group('data/example-19.zarr')
foo = root.create_array(name='foo', shape=(1000, 100), chunks=(10, 10), dtype='float32')
bar = root.create_array(name='bar', shape=(100,), dtype='int32')
foo[:, :] = np.random.random((1000, 100))
bar[:] = np.arange(100)
print(root.tree())
```

## Sharding

Using small chunk shapes in very large arrays can lead to a very large number of chunks.
This can become a performance issue for file systems and object storage.
With Zarr format 3, a new sharding feature has been added to address this issue.

With sharding, multiple chunks can be stored in a single storage object (e.g. a file).
Within a shard, chunks are compressed and serialized separately.
This allows individual chunks to be read independently.
However, when writing data, a full shard must be written in one go for optimal
performance and to avoid concurrency issues.
That means that shards are the units of writing and chunks are the units of reading.
Users need to configure the chunk and shard shapes accordingly.

Sharded arrays can be created by providing the `shards` parameter to [`zarr.create_array`][].

```python exec="true" session="arrays" source="above" result="ansi"
a = zarr.create_array('data/example-20.zarr', shape=(10000, 10000), shards=(1000, 1000), chunks=(100, 100), dtype='uint8')
a[:] = (np.arange(10000 * 10000) % 256).astype('uint8').reshape(10000, 10000)
print(a.info_complete())
```

In this example a shard shape of (1000, 1000) and a chunk shape of (100, 100) is used.
This means that `10*10` chunks are stored in each shard, and there are `10*10` shards in total.
Without the `shards` argument, there would be 10,000 chunks stored as individual files.

## Missing features in 3.0

The following features have not been ported to 3.0 yet.

### Copying and migrating data

See the Zarr-Python 2 documentation on [Copying and migrating data](https://zarr.readthedocs.io/en/support-v2/tutorial.html#copying-migrating-data) for more details.
