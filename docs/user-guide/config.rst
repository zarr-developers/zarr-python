Runtime configuration
=====================

The :mod:`zarr.core.config` module is responsible for managing the configuration of zarr
and is based on the `donfig <https://github.com/pytroll/donfig>`_ Python library.

Configuration values can be set using code like the following:

.. code-block:: python

    import zarr
    zarr.config.set({"array.order": "F"})

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

This is the current default configuration:

.. ipython:: python

    import zarr

    zarr.config.pprint()