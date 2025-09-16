.. _user-guide-config:

Runtime configuration
=====================

``zarr.config`` is responsible for managing the configuration of zarr and
is based on the `donfig <https://github.com/pytroll/donfig>`_ Python library.

Configuration values can be set using code like the following::

   >>> import zarr
   >>>
   >>> zarr.config.set({'array.order': 'F'})
   <donfig.config_obj.ConfigSet object at ...>
   >>>
   >>> # revert this change so it doesn't impact the rest of the docs
   >>> zarr.config.set({'array.order': 'C'})
   <donfig.config_obj.ConfigSet object at ...>

Alternatively, configuration values can be set using environment variables, e.g.
``ZARR_ARRAY__ORDER=F``.

The configuration can also be read from a YAML file in standard locations.
For more information, see the
`donfig documentation <https://donfig.readthedocs.io/en/latest/>`_.

Configuration options include the following:

- ``default_zarr_version``  
  Sets the default Zarr format version. The options are ``2`` or ``3``.

  - **Default:** ``3`` 

- **array.order**  
  Defines the default memory layout for arrays.  

  - ``'C'`` → Row-major (The default value used for NumPy arrays)  
  - ``'F'`` → Column-major (like Fortran)  
  - **Default:** ``'C'``  
 

- Default filters, serializers and compressors, e.g. ``array.v3_default_filters``, ``array.v3_default_serializer``, ``array.v3_default_compressors``, ``array.v2_default_filters`` and ``array.v2_default_compressor``

- ``array.write_empty_chunks``
  Determines whether empty chunks (filled with default values) are written to storage. Setting this to ``False`` can reduce the number of write operations and objects created when writing arrays with large empty regions.

  - ``False`` → Empty chunks are **not written**.  
  - ``True`` → Empty chunks are explicitly stored.  
  - **Default:** ``False``  

- ``async.concurrency``
  Sets the number of concurrent async operations.

  - **Default:** ``10``  

- ``threading.max_workers``
  Defines the maximum number of worker threads for parallel execution.

  - **Default**: ``None`` (uses system default)  

- ``codecs`` / ``codec_pipeline`` / ``buffer``
  Allows selection of custom implementations for codecs, encoding pipelines, and data buffers.  

- GPU Support : Enabling GPU support with ``zarr.config.enable_gpu()``. See :ref:`user-guide-gpu` for more. 


For selecting custom implementations of codecs, pipelines, buffers and ndbuffers,
first register the implementations in the registry and then select them in the config.
For example, an implementation of the bytes codec in a class ``'custompackage.NewBytesCodec'``,
requires the value of ``codecs.bytes.name`` to be ``'custompackage.NewBytesCodec'``.

This is the current default configuration::

   >>> zarr.config.pprint()
   {'array': {'order': 'C', 'write_empty_chunks': False},
    'async': {'concurrency': 10, 'timeout': None},
    'buffer': 'zarr.buffer.cpu.Buffer',
    'codec_pipeline': {'batch_size': 1,
                       'path': 'zarr.core.codec_pipeline.BatchedCodecPipeline'},
    'codecs': {'blosc': 'zarr.codecs.blosc.BloscCodec',
               'bytes': 'zarr.codecs.bytes.BytesCodec',
               'crc32c': 'zarr.codecs.crc32c_.Crc32cCodec',
               'endian': 'zarr.codecs.bytes.BytesCodec',
               'gzip': 'zarr.codecs.gzip.GzipCodec',
               'numcodecs.adler32': 'zarr.codecs.numcodecs.Adler32',
               'numcodecs.astype': 'zarr.codecs.numcodecs.AsType',
               'numcodecs.bitround': 'zarr.codecs.numcodecs.BitRound',
               'numcodecs.blosc': 'zarr.codecs.numcodecs.Blosc',
               'numcodecs.bz2': 'zarr.codecs.numcodecs.BZ2',
               'numcodecs.crc32': 'zarr.codecs.numcodecs.CRC32',
               'numcodecs.crc32c': 'zarr.codecs.numcodecs.CRC32C',
               'numcodecs.delta': 'zarr.codecs.numcodecs.Delta',
               'numcodecs.fixedscaleoffset': 'zarr.codecs.numcodecs.FixedScaleOffset',
               'numcodecs.fletcher32': 'zarr.codecs.numcodecs.Fletcher32',
               'numcodecs.gzip': 'zarr.codecs.numcodecs.GZip',
               'numcodecs.jenkins_lookup3': 'zarr.codecs.numcodecs.JenkinsLookup3',
               'numcodecs.lz4': 'zarr.codecs.numcodecs.LZ4',
               'numcodecs.lzma': 'zarr.codecs.numcodecs.LZMA',
               'numcodecs.packbits': 'zarr.codecs.numcodecs.PackBits',
               'numcodecs.pcodec': 'zarr.codecs.numcodecs.PCodec',
               'numcodecs.quantize': 'zarr.codecs.numcodecs.Quantize',
               'numcodecs.shuffle': 'zarr.codecs.numcodecs.Shuffle',
               'numcodecs.zfpy': 'zarr.codecs.numcodecs.ZFPY',
               'numcodecs.zlib': 'zarr.codecs.numcodecs.Zlib',
               'numcodecs.zstd': 'zarr.codecs.numcodecs.Zstd',
               'sharding_indexed': 'zarr.codecs.sharding.ShardingCodec',
               'transpose': 'zarr.codecs.transpose.TransposeCodec',
               'vlen-bytes': 'zarr.codecs.vlen_utf8.VLenBytesCodec',
               'vlen-utf8': 'zarr.codecs.vlen_utf8.VLenUTF8Codec',
               'zstd': 'zarr.codecs.zstd.ZstdCodec'},
    'default_zarr_format': 3,
    'json_indent': 2,
    'ndbuffer': 'zarr.buffer.cpu.NDBuffer',
    'threading': {'max_workers': None}}
