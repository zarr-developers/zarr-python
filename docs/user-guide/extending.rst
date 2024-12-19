
Extending Zarr
==============

Zarr-Python 3 was designed to be extensible. This means that you can extend
the library by writing custom classes and plugins. Currently, Zarr can be extended
in the following ways:

Writing custom stores
---------------------


Writing custom codecs
---------------------

There are three types of codecs in Zarr: array-to-array, array-to-bytes, and bytes-to-bytes. 
Array-to-array codecs are used to transform the n-dimensional array data before serializing 
to bytes. Examples include delta encoding or scaling codecs. Array-to-bytes codecs are used
for serializing the array data to bytes. In Zarr, the main codec to use for numeric arrays 
is the :class:`zarr.codecs.BytesCodec`. Bytes-to-bytes transform the serialized bytestreams 
of the array data. Examples include compression codecs, such as 
:class:`zarr.codecs.GzipCodec`, :class:`zarr.codecs.BloscCodec` or 
:class:`zarr.codecs.ZstdCodec`, and codecs that add a checksum to the bytestream, such as 
:class:`zarr.codecs.Crc32cCodec`.

Custom codecs for Zarr are implemented as classes that inherit from the relevant base class, 
see :class:`zarr.abc.codecs.ArrayArrayCodec`, :class:`zarr.abc.codecs.ArrayBytesCodec` and 
:class:`zarr.abc.codecs.BytesBytesCodec`. Most custom codecs should implemented the 
``_encode_single`` and ``_decode_single`` methods. These methods operate on single chunks 
of the array data. Custom codecs can also implement the ``encode`` and ``decode`` methods, 
which operate on batches of chunks, in case the codec is intended to implement its own 
batch processing.

Custom codecs should also implement these methods:
- ``compute_encoded_size``, which returns the byte size of the encoded data given the byte 
  size of the original data. It should raise ``NotImplementedError`` for codecs with 
  variable-sized outputs, such as compression codecs.
- ``validate``, which can be used to check that the codec metadata is compatible with the 
  array metadata. It should raise errors if not.
- ``resolve_metadata`` (optional), which is important for codecs that change the shape, 
  dtype or fill value of a chunk.
- ``evolve_from_array_spec`` (optional), which can be useful for automatically filling in 
  codec configuration metadata from the array metadata.

To use custom codecs in Zarr, they need to be registered using the 
`entrypoint mechanism <https://packaging.python.org/en/latest/specifications/entry-points/>_`.
Commonly, entrypoints are declared in the ``pyproject.toml`` of your package under the 
``[project.entry-points]`` section. Zarr will automatically discover and load all codecs 
registered with the entrypoint mechanism from imported modules.

    [project.entry-points."zarr.codecs"]
    "custompackage.fancy_codec" = "custompackage:FancyCodec"

New codecs need to have their own unique identifier. To avoid naming collisions, it is 
strongly recommended to prefix the codec identifier with a unique name. For example, 
the codecs from ``numcodecs`` are prefixed with ``numcodecs.``, e.g. ``numcodecs.delta``.

.. note::
    Note that the extension mechanism for the Zarr specification version 3 is still 
    under development. Requirements for custom codecs including the choice of codec 
    identifiers might change in the future.

It is also possible to register codecs as replacements for existing codecs. This might be 
useful for providing specialized implementations, such as GPU-based codecs. In case of 
multiple codecs, the :mod:`zarr.core.config` mechanism can be used to select the preferred
implementation. 

TODO: Link to documentation of :mod:`zarr.core.config`

.. note::
    This sections explains how custom codecs can be created for Zarr version 3. For Zarr
    version 2, codecs should implement the 
    ```numcodecs.abc.Codec`` <https://numcodecs.readthedocs.io/en/stable/abc.html>_` 
    base class.


In the future, Zarr will support writing custom custom data types and chunk grids.

TODO: Expand this doc page with more detail.
