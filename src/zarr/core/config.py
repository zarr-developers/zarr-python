"""
The config module is responsible for managing the configuration of zarr and is based on the Donfig python library.
For selecting custom implementations of codecs, pipelines, buffers and ndbuffers, first register the implementations
in the registry and then select them in the config.

Example:
    An implementation of the bytes codec in a class `your.module.NewBytesCodec` requires the value of `codecs.bytes`
    to be `your.module.NewBytesCodec`.

    ```python
    from your.module import NewBytesCodec
    from zarr.core.config import register_codec, config

    register_codec("bytes", NewBytesCodec)
    config.set({"codecs.bytes": "your.module.NewBytesCodec"})
    ```

Donfig can be configured programmatically, by environment variables, or from YAML files in standard locations.
For example, to set the bytes codec via an environment variable:

    export ZARR_CODECS__BYTES="your.module.NewBytesCodec"

For more information, see the Donfig documentation at https://github.com/pytroll/donfig.

Default values below point to the standard implementations of zarr-python.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from donfig import Config as DConfig


class BadConfigError(ValueError):
    _msg = "bad Config: %r"


class Config(DConfig):  # type: ignore[misc]
    """The Config will collect configuration from config files and environment variables

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


config = Config(
    "zarr",
    defaults=[
        {
            "default_zarr_version": 3,
            "array": {
                "order": "C",
                "v2_default_compressor": {
                    "numeric": "zstd",
                    "string": "vlen-utf8",
                    "bytes": "vlen-bytes",
                },
                "v3_default_codecs": {
                    "numeric": ["bytes", "zstd"],
                    "string": ["vlen-utf8"],
                    "bytes": ["vlen-bytes"],
                },
            },
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
