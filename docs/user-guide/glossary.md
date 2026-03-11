# Glossary

This page defines key terms used throughout the zarr-python documentation and API.

## Array Structure

### Array

An N-dimensional typed array stored in a Zarr [store](#store). An array's
[metadata](#metadata) defines its shape, data type, chunk layout, and codecs.

### Chunk

The fundamental unit of data in a Zarr array. An array is divided into chunks
along each dimension according to the [chunk grid](#chunk-grid), which is currently
part of Zarr's private API. Each chunk is independently compressed and encoded
through the array's [codec](#codec) pipeline.

When [sharding](#shard) is used, "chunk" refers to the inner chunks within each
shard, because those are the compressible units. The chunks are the smallest units
that can be read independently.

**API**: [`Array.chunks`][zarr.Array.chunks] returns the chunk shape. When
sharding is used, this is the inner chunk shape.

### Chunk Grid

The partitioning of an array's elements into [chunks](#chunk). In Zarr V3, the
chunk grid is defined in the array [metadata](#metadata) and determines the
boundaries of each storage object.

When sharding is used, the chunk grid defines the [shard](#shard) boundaries,
not the inner chunk boundaries. The inner chunk shape is defined within the
[sharding codec](#shard).

**API**: The `chunk_grid` field in array metadata contains the storage-level
grid.

### Shard

A storage object that contains one or more [chunks](#chunk). Sharding reduces the
number of objects in a [store](#store) by grouping chunks together, which
improves performance on file systems and object storage.

Within each shard, chunks are compressed independently and can be read
individually. However, writing requires updating the full shard for consistency,
making shards the unit of writing and chunks the unit of reading.

Sharding is implemented as a [codec](#codec) (the sharding indexed codec).
When sharding is used:

- The [chunk grid](#chunk-grid) in metadata defines the shard boundaries
- The sharding codec's `chunk_shape` defines the inner chunk size
- Each shard contains `shard_shape / chunk_shape` chunks per dimension

**API**: [`Array.shards`][zarr.Array.shards] returns the shard shape, or `None`
if sharding is not used. [`Array.chunks`][zarr.Array.chunks] returns the inner
chunk shape.

## Storage

### Store

A key-value storage backend that holds Zarr data and metadata. Stores implement
the [`zarr.abc.store.Store`][] interface. Examples include local file systems,
cloud object storage (S3, GCS, Azure), zip files, and in-memory dictionaries.

Each [chunk](#chunk) or [shard](#shard) is stored as a single value (object or
file) in the store, addressed by a key derived from its grid coordinates.

### Metadata

The JSON document (`zarr.json`) that describes an [array](#array) or group. For
arrays, metadata includes the shape, data type, [chunk grid](#chunk-grid), fill
value, and [codec](#codec) pipeline. Metadata is stored alongside the data in
the [store](#store). Zarr-Python does not yet expose its internal metadata
representation as part of its public API.

## Codecs

### Codec

A transformation applied to array data during reading and writing. Codecs are
chained into a pipeline and come in three types:

- **Array-to-array**: Transforms like transpose that rearrange array elements
- **Array-to-bytes**: Serialization that converts an array to a byte sequence
  (exactly one required)
- **Bytes-to-bytes**: Compression or checksums applied to the serialized bytes

The [sharding indexed codec](#shard) is a special array-to-bytes codec that
groups multiple [chunks](#chunk) into a single storage object.

## API Properties

The following properties are available on [`zarr.Array`][]:

| Property | Description |
|----------|-------------|
| `.chunks` | Chunk shape — the inner chunk shape when sharding is used |
| `.shards` | Shard shape, or `None` if no sharding |
| `.nchunks` | Total number of independently compressible units across the array |
| `.cdata_shape` | Number of independently compressible units per dimension |
