.. _user-guide-config:

Runtime configuration
=====================

The :mod:`zarr.core.config` module is responsible for managing the configuration of zarr
and is based on the `donfig <https://github.com/pytroll/donfig>`_ Python library.

Configuration values can be set using code like the following::

   >>> import zarr
   >>>
   >>> zarr.config.set({"array.order": "F"})
   <donfig.config_obj.ConfigSet object at ...>
   >>>
   >>> # revert this change so it doesn't impact the rest of the docs
   >>> zarr.config.set({"array.order": "C"})
   <donfig.config_obj.ConfigSet object at ...>

Alternatively, configuration values can be set using environment variables, e.g.
``ZARR_ARRAY__ORDER=F``.

The configuration can also be read from a YAML file in standard locations.
For more information, see the
`donfig documentation <https://donfig.readthedocs.io/en/latest/>`_.

Configuration options include the following:

- Default Zarr format ``default_zarr_version``
- Default array order in memory ``array.order``
- Default codecs ``array.v3_default_codecs`` and ``array.v2_default_compressor``
- Whether empty chunks are written to storage ``array.write_empty_chunks``
- Async and threading options, e.g. ``async.concurrency`` and ``threading.max_workers``
- Selections of implementations of codecs, codec pipelines and buffers

For selecting custom implementations of codecs, pipelines, buffers and ndbuffers,
first register the implementations in the registry and then select them in the config.
For example, an implementation of the bytes codec in a class "custompackage.NewBytesCodec",
requires the value of ``codecs.bytes.name`` to be "custompackage.NewBytesCodec".

This is the current default configuration::

   >>> zarr.config.pprint()
   {'array': {'order': 'C',
              'v2_default_compressor': {'bytes': 'vlen-bytes',
                                        'numeric': 'zstd',
                                        'string': 'vlen-utf8'},
              'v3_default_codecs': {'bytes': ['vlen-bytes'],
                                    'numeric': ['bytes', 'zstd'],
                                    'string': ['vlen-utf8']},
              'write_empty_chunks': False},
    'async': {'concurrency': 10, 'timeout': None},
    'buffer': 'zarr.core.buffer.cpu.Buffer',
    'codec_pipeline': {'batch_size': 1,
                       'path': 'zarr.core.codec_pipeline.BatchedCodecPipeline'},
    'codecs': {'blosc': 'zarr.codecs.blosc.BloscCodec',
               'bytes': 'zarr.codecs.bytes.BytesCodec',
               'crc32c': 'zarr.codecs.crc32c_.Crc32cCodec',
               'endian': 'zarr.codecs.bytes.BytesCodec',
               'gzip': 'zarr.codecs.gzip.GzipCodec',
               'sharding_indexed': 'zarr.codecs.sharding.ShardingCodec',
               'transpose': 'zarr.codecs.transpose.TransposeCodec',
               'vlen-bytes': 'zarr.codecs.vlen_utf8.VLenBytesCodec',
               'vlen-utf8': 'zarr.codecs.vlen_utf8.VLenUTF8Codec',
               'zstd': 'zarr.codecs.zstd.ZstdCodec'},
    'default_zarr_version': 3,
    'json_indent': 2,
    'ndbuffer': 'zarr.core.buffer.cpu.NDBuffer',
    'threading': {'max_workers': None}}
