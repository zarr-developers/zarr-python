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
is the `zarr.codecs.BytesCodec`. Bytes-to-bytes codecs transform the serialized bytestreams
of the array data. Examples include compression codecs, such as
`zarr.codecs.GzipCodec`, `zarr.codecs.BloscCodec` or
`zarr.codecs.ZstdCodec`, and codecs that add a checksum to the bytestream, such as
`zarr.codecs.Crc32cCodec`.

Custom codecs for Zarr are implemented by subclassing the relevant base class, see
`zarr.abc.codec.ArrayArrayCodec`, `zarr.abc.codec.ArrayBytesCodec` and
`zarr.abc.codec.BytesBytesCodec`. Most custom codecs should implemented the
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
- `evolve_from_array_spec` (optional), which can be useful for automatically filling in
  codec configuration metadata from the array metadata.

To use custom codecs in Zarr, they need to be registered using the
[entrypoint mechanism](https://packaging.python.org/en/latest/specifications/entry-points/).
Commonly, entrypoints are declared in the `pyproject.toml` of your package under the
`[project.entry-points."zarr.codecs"]` section. Zarr will automatically discover and
load all codecs registered with the entrypoint mechanism from imported modules.

```toml
[project.entry-points."zarr.codecs"]
"custompackage.fancy_codec" = "custompackage:FancyCodec"
```

New codecs need to have their own unique identifier. To avoid naming collisions, it is
strongly recommended to prefix the codec identifier with a unique name. For example,
the codecs from `numcodecs` are prefixed with `numcodecs.`, e.g. `numcodecs.delta`.

!!! note
    Note that the extension mechanism for the Zarr format 3 is still under development.
    Requirements for custom codecs including the choice of codec identifiers might
    change in the future.

It is also possible to register codecs as replacements for existing codecs. This might be
useful for providing specialized implementations, such as GPU-based codecs. In case of
multiple codecs, the `zarr.core.config` mechanism can be used to select the preferred
implementation.

## Custom stores

Coming soon.

## Custom array buffers

Coming soon.

## Other extensions

In the future, Zarr will support writing custom custom data types and chunk grids.
