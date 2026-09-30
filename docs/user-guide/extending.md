# Extending Zarr

Zarr-Python 3 was designed to be extensible. This means that you can extend
the library by writing custom classes and plugins. Currently, Zarr can be extended
in the following ways:

## Custom codecs

!!! note
    This section explains how custom codecs can be created for Zarr format 3 arrays. For Zarr
    format 2, codecs should subclass the
    [numcodecs.abc.Codec](https://numcodecs.readthedocs.io/en/stable/abc.html#numcodecs.abc.Codec)
    base class and register through
    [numcodecs.registry.register_codec](https://numcodecs.readthedocs.io/en/stable/registry.html#numcodecs.registry.register_codec).

There are three types of codecs in Zarr:

- array-to-array
- array-to-bytes
- bytes-to-bytes

Array-to-array codecs are used to transform the array data before serializing
to bytes. Examples include delta encoding or scaling codecs. Array-to-bytes codecs are used
for serializing the array data to bytes. In Zarr, the main codec to use for numeric arrays
is the [`zarr.codecs.BytesCodec`][]. Bytes-to-bytes codecs transform the serialized bytestreams
of the array data. Examples include compression codecs, such as
[`zarr.codecs.GzipCodec`][], [`zarr.codecs.BloscCodec`][] or
[`zarr.codecs.ZstdCodec`][], and codecs that add a checksum to the bytestream, such as
[`zarr.codecs.Crc32cCodec`][].

Custom codecs for Zarr are implemented by subclassing the relevant base class, see
[`zarr.abc.codec.ArrayArrayCodec`][], [`zarr.abc.codec.ArrayBytesCodec`][] and
[`zarr.abc.codec.BytesBytesCodec`][]. Most custom codecs should implement the
`_encode_single` and `_decode_single` methods. These methods operate on single chunks
of the array data. Alternatively, custom codecs can implement the `encode` and `decode`
methods, which operate on batches of chunks, in case the codec is intended to implement
its own batch processing.

Custom codecs should also implement the following methods:

- `compute_encoded_size`, which returns the byte size of the encoded data given the byte
  size of the original data. It should raise `NotImplementedError` for codecs with
  variable-sized outputs, such as compression codecs.
- `validate` (optional), which can be used to check that the codec metadata is compatible with the
  array metadata. It should raise errors if not.
- `resolve_metadata` (optional), which is important for codecs that change the shape,
  dtype or fill value of a chunk.
- `resolve_chunk_grid` (optional, but recommended for any codec that overrides
  `resolve_metadata`), which describes how the codec changes the chunk grid as a whole.
  See [Chunk geometry and validation](#chunk-geometry-and-validation).
- `evolve_from_array_spec` (optional), which can be useful for automatically filling in
  codec configuration metadata from the array metadata.

### Chunk geometry and validation

When array metadata is created, Zarr validates every codec against the chunks it will
actually receive, which are the chunks produced by the codecs before it. For example, in
the chain `transpose` → `sharding_indexed`, the shard's inner chunk shape must divide the
*transposed* chunks.

The geometry is carried through the chain as a whole chunk grid rather than chunk by
chunk. A rectilinear grid with `n` distinct edge lengths along each of `d` axes has `n**d`
distinct chunk shapes, which quickly becomes too many to check one at a time (a 3-d grid
with 1000 distinct edges per axis has a billion). Each codec therefore reports how it
changes the grid through `resolve_chunk_grid(shape=..., chunk_grid=...)`:

- **Return `(shape, chunk_grid)` unchanged** if the codec never changes the chunk
  shape, even if it changes the data type or fill value. This is the default for codecs
  that do not override `resolve_metadata`. If your codec overrides `resolve_metadata`
  only to change the data type or fill value, override `resolve_chunk_grid` too, as
  `CastValue`, `ScaleOffset` and the numcodecs `Delta` do.
- **Return the mapped shape and grid** if the codec changes chunk shapes in a way a grid
  can express. `TransposeCodec`, for example, permutes the axes of both.
- **Return `None`** if the chunks after the codec cannot be described by a single grid
  computed from the input grid. This is the default for codecs that override
  `resolve_metadata`.

Returning `None` has no cost on a regular chunk grid, because every chunk has the same
shape and `resolve_metadata` describes it exactly. On a rectilinear chunk grid it makes
the rest of the chain **chunk-local**. From then on, codecs are validated against a single
representative chunk: the one with the largest edge along each axis. This has the
following consequences:

- **Rejections are always correct.** The representative is a real chunk of the array, so
  a codec that rejects it would fail on that chunk.
- **Acceptance is incomplete.** A chain that is invalid only for some *other* chunk
  shape is accepted when the array is created, and the error surfaces when a chunk of
  that shape is first encoded or decoded. Suppose a sharding codec follows an undeclared
  filter on a grid with chunk edges `[4, 6]`. An inner chunk size of `3` divides the
  representative edge `6`, so the array is created, but writing to a chunk with edge `4`
  raises an error.
- **Size-sensitive codecs must check at run time.** Any codec whose correctness depends on
  the chunk shape (for example, requiring it to divide evenly) must repeat that check
  when encoding and decoding, because metadata validation may have seen only the
  representative chunk. `ShardingCodec` does this. A codec that skips the check can
  silently corrupt data on the chunks that validation never saw.

Zarr uses a declared grid only when it can trust it. A declaration is ignored, and the
codec treated as returning `None`, if a subclass overrides `resolve_metadata` without
also overriding `resolve_chunk_grid`. It is also ignored if the declared grid disagrees
with what `resolve_metadata` returns for the representative chunk.

This design follows [zarrs](https://github.com/zarrs/zarrs), whose array-to-array codecs
map chunk grids through `encoded_chunk_grid` and return a "chunk-local" result when no
whole-array grid exists.

To use custom codecs in Zarr, they need to be registered using the
[entrypoint mechanism](https://packaging.python.org/en/latest/specifications/entry-points/).
Commonly, entrypoints are declared in the `pyproject.toml` of your package under the
`[project.entry-points."zarr.codecs"]` section. Zarr will automatically discover
all codecs registered via the entrypoint mechanism in installed packages.

```toml
[project.entry-points."zarr.codecs"]
"custompackage.fancy_codec" = "custompackage:FancyCodec"
```

New codecs need to have their own unique identifier. To avoid naming collisions, it is
strongly recommended to prefix the codec identifier with a unique name. For example,
the codecs from `numcodecs` are prefixed with `numcodecs.`, e.g. `numcodecs.delta`.

If someone opens an array that uses your codec without your package installed, Zarr raises
[`zarr.errors.UnknownCodecError`][] explaining how to register an implementation. Zarr also
keeps a small table of codec names and the published packages that provide them, and names
those packages in that error. Once your package is on PyPI, please open a pull request adding
it to the codec-package tables in `src/zarr/registry.py`, so that users get a message telling them
exactly what to install.

!!! note
    Note that the extension mechanism for the Zarr format 3 is still under development.
    Requirements for custom codecs including the choice of codec identifiers might
    change in the future.

It is also possible to register codecs as replacements for existing codecs. This might be
useful for providing specialized implementations, such as GPU-based codecs. In case of
multiple codecs, the [`zarr.config`][] mechanism can be used to select the preferred
implementation.

## Custom stores

Custom stores can be created by implementing the [`zarr.abc.store.Store`][] interface.
See [developing custom stores](storage.md#developing-custom-stores) for more information.

## Custom array buffers

Zarr-python provides control over where and how arrays are stored in memory through
[`zarr.abc.buffer.Buffer`][]. Currently both CPU (the default) and GPU implementations are
provided (see [Using GPUs with Zarr](gpu.md) for more information). You can implement your own buffer
classes by implementing the interface defined in [`zarr.abc.buffer.BufferPrototype`][].
Like codecs, custom buffer implementations can be registered via entrypoints, using the
`zarr.buffer` and `zarr.ndbuffer` entrypoint groups.

## Custom data types

Zarr supports user-defined data types. See the
[data types documentation](data_types.md) for an explanation of how Zarr Python
models data types and how to write your own, and the
[custom data type example](examples/custom_dtype.md) for a complete worked example.

## Other extensions

In the future, Zarr will support writing custom chunk grids.
