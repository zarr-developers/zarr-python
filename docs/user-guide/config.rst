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

- Default Zarr format ``default_zarr_version``
- Default array order in memory ``array.order``
- Default filters, serializers and compressors, e.g. ``array.v3_default_filters``, ``array.v3_default_serializer``, ``array.v3_default_compressors``, ``array.v2_default_filters`` and ``array.v2_default_compressor``
- Whether empty chunks are written to storage ``array.write_empty_chunks``
- Async and threading options, e.g. ``async.concurrency`` and ``threading.max_workers``
- Selections of implementations of codecs, codec pipelines and buffers
- Enabling GPU support with ``zarr.config.enable_gpu()``. See :ref:`user-guide-gpu` for more.

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
    'codecs': {'adler32': 'zarr.codecs.numcodecs.Adler32',
               'astype': 'zarr.codecs.numcodecs.AsType',
               'bitround': 'zarr.codecs.numcodecs.BitRound',
               'blosc': 'zarr.codecs.blosc.BloscCodec',
               'bytes': 'zarr.codecs.bytes.BytesCodec',
               'bz2': 'zarr.codecs.numcodecs.BZ2',
               'crc32': 'zarr.codecs.numcodecs.CRC32',
               'crc32c': 'zarr.codecs.crc32c_.Crc32cCodec',
               'delta': 'zarr.codecs.numcodecs.Delta',
               'endian': 'zarr.codecs.bytes.BytesCodec',
               'fixedscaleoffset': 'zarr.codecs.numcodecs.FixedScaleOffset',
               'fletcher32': 'zarr.codecs.numcodecs.Fletcher32',
               'gzip': 'zarr.codecs.gzip.GzipCodec',
               'jenkins_lookup3': 'zarr.codecs.numcodecs.JenkinsLookup3',
               'lz4': 'zarr.codecs.numcodecs.LZ4',
               'lzma': 'zarr.codecs.numcodecs.LZMA',
               'numcodecs.adler32': 'zarr.codecs.numcodecs.Adler32',
               'numcodecs.astype': 'zarr.codecs.numcodecs.AsType',
               'numcodecs.bitround': 'zarr.codecs.numcodecs.BitRound',
               'numcodecs.blosc': 'zarr.codecs.blosc.BloscCodec',
               'numcodecs.bz2': 'zarr.codecs.numcodecs.BZ2',
               'numcodecs.crc32': 'zarr.codecs.numcodecs.CRC32',
               'numcodecs.crc32c': 'zarr.codecs.crc32c_.Crc32cCodec',
               'numcodecs.delta': 'zarr.codecs.numcodecs.Delta',
               'numcodecs.fixedscaleoffset': 'zarr.codecs.numcodecs.FixedScaleOffset',
               'numcodecs.fletcher32': 'zarr.codecs.numcodecs.Fletcher32',
               'numcodecs.gzip': 'zarr.codecs.gzip.GzipCodec',
               'numcodecs.jenkins_lookup3': 'zarr.codecs.numcodecs.JenkinsLookup3',
               'numcodecs.lz4': 'zarr.codecs.numcodecs.LZ4',
               'numcodecs.lzma': 'zarr.codecs.numcodecs.LZMA',
               'numcodecs.packbits': 'zarr.codecs.numcodecs.PackBits',
               'numcodecs.pcodec': 'zarr.codecs.numcodecs.PCodec',
               'numcodecs.quantize': 'zarr.codecs.numcodecs.Quantize',
               'numcodecs.shuffle': 'zarr.codecs.numcodecs.Shuffle',
               'numcodecs.zfpy': 'zarr.codecs.numcodecs.ZFPY',
               'numcodecs.zlib': 'zarr.codecs.numcodecs.Zlib',
               'numcodecs.zstd': 'zarr.codecs.zstd.ZstdCodec',
               'packbits': 'zarr.codecs.numcodecs.PackBits',
               'pcodec': 'zarr.codecs.numcodecs.PCodec',
               'quantize': 'zarr.codecs.numcodecs.Quantize',
               'sharding_indexed': 'zarr.codecs.sharding.ShardingCodec',
               'shuffle': 'zarr.codecs.numcodecs.Shuffle',
               'transpose': 'zarr.codecs.transpose.TransposeCodec',
               'vlen-bytes': 'zarr.codecs.vlen_utf8.VLenBytesCodec',
               'vlen-utf8': 'zarr.codecs.vlen_utf8.VLenUTF8Codec',
               'zfpy': 'zarr.codecs.numcodecs.ZFPY',
               'zlib': 'zarr.codecs.numcodecs.Zlib',
               'zstd': 'zarr.codecs.zstd.ZstdCodec'},
    'default_zarr_format': 3,
    'json_indent': 2,
    'ndbuffer': 'zarr.buffer.cpu.NDBuffer',
    'threading': {'max_workers': None}}
