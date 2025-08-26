
Extending Zarr
==============

Zarr-Python 3 was designed to be extensible. This means that you can extend
the library by writing custom classes and plugins. Currently, Zarr can be extended
in the following ways:

Custom codecs
-------------

.. note::
    This section explains how custom codecs can be created for Zarr format 3 arrays. For Zarr
    format 2, codecs should subclass the
    `numcodecs.abc.Codec <https://numcodecs.readthedocs.io/en/stable/abc.html#numcodecs.abc.Codec>`_
    base class and register through
    `numcodecs.registry.register_codec <https://numcodecs.readthedocs.io/en/stable/registry.html#numcodecs.registry.register_codec>`_.

There are three types of codecs in Zarr:
- array-to-array
- array-to-bytes
- bytes-to-bytes

Array-to-array codecs are used to transform the array data before serializing
to bytes. Examples include delta encoding or scaling codecs. Array-to-bytes codecs are used
for serializing the array data to bytes. In Zarr, the main codec to use for numeric arrays
is the :class:`zarr.codecs.BytesCodec`. Bytes-to-bytes codecs transform the serialized bytestreams
of the array data. Examples include compression codecs, such as
:class:`zarr.codecs.GzipCodec`, :class:`zarr.codecs.BloscCodec` or
:class:`zarr.codecs.ZstdCodec`, and codecs that add a checksum to the bytestream, such as
:class:`zarr.codecs.Crc32cCodec`.

Custom codecs for Zarr are implemented by subclassing the relevant base class, see
:class:`zarr.abc.codec.ArrayArrayCodec`, :class:`zarr.abc.codec.ArrayBytesCodec` and
:class:`zarr.abc.codec.BytesBytesCodec`. Most custom codecs should implemented the
``_encode_single`` and ``_decode_single`` methods. These methods operate on single chunks
of the array data. Alternatively, custom codecs can implement the ``encode`` and ``decode``
methods, which operate on batches of chunks, in case the codec is intended to implement
its own batch processing.

Custom codecs should also implement the following methods:

- ``compute_encoded_size``, which returns the byte size of the encoded data given the byte
  size of the original data. It should raise ``NotImplementedError`` for codecs with
  variable-sized outputs, such as compression codecs.
- ``validate`` (optional), which can be used to check that the codec metadata is compatible with the
  array metadata. It should raise errors if not.
- ``resolve_metadata`` (optional), which is important for codecs that change the shape,
  dtype or fill value of a chunk.
- ``evolve_from_array_spec`` (optional), which can be useful for automatically filling in
  codec configuration metadata from the array metadata.

To use custom codecs in Zarr, they need to be registered using the
`entrypoint mechanism <https://packaging.python.org/en/latest/specifications/entry-points/>`_.
Commonly, entrypoints are declared in the ``pyproject.toml`` of your package under the
``[project.entry-points."zarr.codecs"]`` section. Zarr will automatically discover and
load all codecs registered with the entrypoint mechanism from imported modules.

.. code-block:: toml

    [project.entry-points."zarr.codecs"]
    "custompackage.fancy_codec" = "custompackage:FancyCodec"

New codecs need to have their own unique identifier. To avoid naming collisions, it is
strongly recommended to prefix the codec identifier with a unique name. For example,
the codecs from ``numcodecs`` are prefixed with ``numcodecs.``, e.g. ``numcodecs.delta``.

.. note::
    Note that the extension mechanism for the Zarr format 3 is still under development.
    Requirements for custom codecs including the choice of codec identifiers might
    change in the future.

It is also possible to register codecs as replacements for existing codecs. This might be
useful for providing specialized implementations, such as GPU-based codecs. In case of
multiple codecs, the :mod:`zarr.core.config` mechanism can be used to select the preferred
implementation.

Custom stores
-------------

Zarr-Python supports two levels of store customization: custom store implementations and custom store adapters for ZEP 8 URL syntax.

Custom Store Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom stores can be created by subclassing :class:`zarr.abc.store.Store`. The Store Abstract Base Class includes all of the methods needed to be a fully operational store in Zarr Python. Zarr also provides a test harness for custom stores: :class:`zarr.testing.store.StoreTests`.

See the :ref:`user-guide-custom-stores` section in the storage guide for more details.

Custom Store Adapters
~~~~~~~~~~~~~~~~~~~~~~

Store adapters enable custom storage backends to work with ZEP 8 URL syntax. This allows users to access your storage backend using simple URL strings instead of explicitly creating store objects.

Store adapters are implemented by subclassing :class:`zarr.abc.store_adapter.StoreAdapter` and registering them via entry points:

.. code-block:: python

    from zarr.abc.store_adapter import StoreAdapter
    from zarr.abc.store import Store

    class FooStoreAdapter(StoreAdapter):
        adapter_name = "foo"  # Used in URLs like "file:data|foo"

        @classmethod
        async def from_url_segment(cls, segment, preceding_url=None, **kwargs):
            # Create and return a Store instance based on the URL segment
            # segment.path contains the path from the URL
            # preceding_url contains the URL from previous adapters

            store = FooStore(segment.path, **kwargs)
            await store._open()
            return store

Register the adapter in your ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."zarr.stores"]
    "foo" = "mypackage:FooStoreAdapter"

Once registered, your adapter can be used in ZEP 8 URLs:

.. code-block:: python

    # Users can now use your custom adapter
    zarr.open_array("file:data.foo|foo", mode='r')

    # Or chain with other adapters
    zarr.open_array("s3://bucket/data.custom|foo|zip", mode='r')

Store Adapter Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

When implementing custom store adapters:

1. **Choose unique names**: Use descriptive, unique adapter names to avoid conflicts
2. **Handle errors gracefully**: Provide clear error messages, particularly for invalid URLs or missing dependencies
3. **Document URL syntax**: Clearly document the expected URL format for your adapter

Custom array buffers
--------------------

Zarr-python provides control over where and how arrays stored in memory through
:mod:`zarr.buffer`. Currently both CPU (the default) and GPU implementations are
provided (see :ref:`user-guide-gpu` for more). You can implement your own buffer
classes by implementing the interface defined in :mod:`zarr.abc.buffer`.

Other extensions
----------------

In the future, Zarr will support writing custom custom data types and chunk grids.
