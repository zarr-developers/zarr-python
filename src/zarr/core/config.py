"""
The config module is responsible for managing the configuration of zarr
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
- Async and threading options, e.g. ``async.concurrency`` and ``threading.max_workers``
- Selections of implementations of codecs, codec pipelines and buffers

For selecting custom implementations of codecs, pipelines, buffers and ndbuffers,
first register the implementations in the registry and then select them in the config.
For example, an implementation of the bytes codec in a class "custompackage.NewBytesCodec",
requires the value of ``codecs.bytes.name`` to be "custompackage.NewBytesCodec".

This is the current default configuration:

.. code-block:: python

    {
        "default_zarr_version": 3,
        "array": {"order": "C"},
        "async": {"concurrency": 10, "timeout": None},
        "threading": {"max_workers": None},
        "json_indent": 2,
        "codec_pipeline": {
            "path": "zarr.core.codec_pipeline.BatchedCodecPipeline",
            "batch_size": 1,
        },
        "codecs": {
            "blosc": "zarr.codecs.blosc.BloscCodec",
            "gzip": "zarr.codecs.gzip.GzipCodec",
            "zstd": "zarr.codecs.zstd.ZstdCodec",
            "bytes": "zarr.codecs.bytes.BytesCodec",
            "endian": "zarr.codecs.bytes.BytesCodec",
            "crc32c": "zarr.codecs.crc32c_.Crc32cCodec",
            "sharding_indexed": "zarr.codecs.sharding.ShardingCodec",
            "transpose": "zarr.codecs.transpose.TransposeCodec",
            "vlen-utf8": "zarr.codecs.vlen_utf8.VLenUTF8Codec",
            "vlen-bytes": "zarr.codecs.vlen_utf8.VLenBytesCodec",
        },
        "buffer": "zarr.core.buffer.cpu.Buffer",
        "ndbuffer": "zarr.core.buffer.cpu.NDBuffer",
    }
"""

from __future__ import annotations

from typing import Any, Literal, cast

from donfig import Config as DConfig


class BadConfigError(ValueError):
    _msg = "bad Config: %r"


class Config(DConfig):  # type: ignore[misc]
    """Will collect configuration from config files and environment variables

    Example environment variables:
    Grabs environment variables of the form "ZARR_FOO__BAR_BAZ=123" and
    turns these into config variables of the form ``{"foo": {"bar-baz": 123}}``
    It transforms the key and value in the following way:

    -  Lower-cases the key text
    -  Treats ``__`` (double-underscore) as nested access
    -  Calls ``ast.literal_eval`` on the value

    """

    def reset(self) -> None:
        self.clear()
        self.refresh()


# The config module is responsible for managing the configuration of zarr and  is based on the Donfig python library.
# For selecting custom implementations of codecs, pipelines, buffers and ndbuffers, first register the implementations
# in the registry and then select them in the config.
# e.g. an implementation of the bytes codec in a class "NewBytesCodec", requires the value of codecs.bytes.name to be
# "NewBytesCodec".
# Donfig can be configured programmatically, by environment variables, or from YAML files in standard locations
# e.g. export ZARR_CODECS__BYTES__NAME="NewBytesCodec"
# (for more information see github.com/pytroll/donfig)
# Default values below point to the standard implementations of zarr-python
config = Config(
    "zarr",
    defaults=[
        {
            "default_zarr_version": 3,
            "array": {"order": "C"},
            "async": {"concurrency": 10, "timeout": None},
            "threading": {"max_workers": None},
            "json_indent": 2,
            "codec_pipeline": {
                "path": "zarr.core.codec_pipeline.BatchedCodecPipeline",
                "batch_size": 1,
            },
            "codecs": {
                "blosc": "zarr.codecs.blosc.BloscCodec",
                "gzip": "zarr.codecs.gzip.GzipCodec",
                "zstd": "zarr.codecs.zstd.ZstdCodec",
                "bytes": "zarr.codecs.bytes.BytesCodec",
                "endian": "zarr.codecs.bytes.BytesCodec",  # compatibility with earlier versions of ZEP1
                "crc32c": "zarr.codecs.crc32c_.Crc32cCodec",
                "sharding_indexed": "zarr.codecs.sharding.ShardingCodec",
                "transpose": "zarr.codecs.transpose.TransposeCodec",
                "vlen-utf8": "zarr.codecs.vlen_utf8.VLenUTF8Codec",
                "vlen-bytes": "zarr.codecs.vlen_utf8.VLenBytesCodec",
            },
            "buffer": "zarr.core.buffer.cpu.Buffer",
            "ndbuffer": "zarr.core.buffer.cpu.NDBuffer",
        }
    ],
)


def parse_indexing_order(data: Any) -> Literal["C", "F"]:
    if data in ("C", "F"):
        return cast(Literal["C", "F"], data)
    msg = f"Expected one of ('C', 'F'), got {data} instead."
    raise ValueError(msg)
