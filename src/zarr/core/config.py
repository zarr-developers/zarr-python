"""
The config module is responsible for managing the configuration of zarr and is based on the Donfig python library.
For selecting custom implementations of codecs, pipelines, buffers and ndbuffers, first register the implementations
in the registry and then select them in the config.

Example:
    An implementation of the bytes codec in a class ``your.module.NewBytesCodec`` requires the value of ``codecs.bytes``
    to be ``your.module.NewBytesCodec``. Donfig can be configured programmatically, by environment variables, or from
    YAML files in standard locations.

    ```python
    from your.module import NewBytesCodec
    from zarr.core.config import register_codec, config

    register_codec("bytes", NewBytesCodec)
    config.set({"codecs.bytes": "your.module.NewBytesCodec"})
    ```

    Instead of setting the value programmatically with ``config.set``, you can also set the value with an environment
    variable. The environment variable ``ZARR_CODECS__BYTES`` can be set to ``your.module.NewBytesCodec``. The double
    underscore ``__`` is used to indicate nested access.

    ```bash
    export ZARR_CODECS__BYTES="your.module.NewBytesCodec"
    ```

For more information, see the Donfig documentation at https://github.com/pytroll/donfig.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from donfig import Config as DConfig

if TYPE_CHECKING:
    from donfig.config_obj import ConfigSet


DEFAULT_CODECS: dict[str, str] = {
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
    "numcodecs.bz2": "zarr.codecs.numcodecs.BZ2",
    "numcodecs.crc32": "zarr.codecs.numcodecs.CRC32",
    "numcodecs.crc32c": "zarr.codecs.numcodecs.CRC32C",
    "numcodecs.lz4": "zarr.codecs.numcodecs.LZ4",
    "numcodecs.lzma": "zarr.codecs.numcodecs.LZMA",
    "numcodecs.zfpy": "zarr.codecs.numcodecs.ZFPY",
    "numcodecs.adler32": "zarr.codecs.numcodecs.Adler32",
    "numcodecs.astype": "zarr.codecs.numcodecs.AsType",
    "numcodecs.bitround": "zarr.codecs.numcodecs.BitRound",
    "numcodecs.blosc": "zarr.codecs.numcodecs.Blosc",
    "numcodecs.delta": "zarr.codecs.numcodecs.Delta",
    "numcodecs.fixedscaleoffset": "zarr.codecs.numcodecs.FixedScaleOffset",
    "numcodecs.fletcher32": "zarr.codecs.numcodecs.Fletcher32",
    "numcodecs.gzip": "zarr.codecs.numcodecs.GZip",
    "numcodecs.jenkins_lookup3": "zarr.codecs.numcodecs.JenkinsLookup3",
    "numcodecs.pcodec": "zarr.codecs.numcodecs.PCodec",
    "numcodecs.packbits": "zarr.codecs.numcodecs.PackBits",
    "numcodecs.shuffle": "zarr.codecs.numcodecs.Shuffle",
    "numcodecs.quantize": "zarr.codecs.numcodecs.Quantize",
    "numcodecs.zlib": "zarr.codecs.numcodecs.Zlib",
    "numcodecs.zstd": "zarr.codecs.numcodecs.Zstd",
}

# Map serialized dotted-key segments to Python field names where they differ
# (Python keywords cannot be used as identifiers).
_FIELD_ALIASES: dict[str, str] = {"async": "async_"}
_SERIALIZED_NAMES: dict[str, str] = {v: k for k, v in _FIELD_ALIASES.items()}


@dataclass(frozen=True, slots=True)
class ArraySettings:
    order: Literal["C", "F"] = "C"
    write_empty_chunks: bool = False
    read_missing_chunks: bool = True
    target_shard_size_bytes: int | None = None
    rectilinear_chunks: bool = False
    sharding_coalesce_max_gap_bytes: int = 1 << 20
    sharding_coalesce_max_bytes: int = 16 << 20


@dataclass(frozen=True, slots=True)
class AsyncSettings:
    concurrency: int = 10
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class ThreadingSettings:
    max_workers: int | None = None


@dataclass(frozen=True, slots=True)
class CodecPipelineSettings:
    path: str = "zarr.core.codec_pipeline.BatchedCodecPipeline"
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class ZarrConfig:
    default_zarr_format: Literal[2, 3] = 3
    array: ArraySettings = field(default_factory=ArraySettings)
    async_: AsyncSettings = field(default_factory=AsyncSettings)
    threading: ThreadingSettings = field(default_factory=ThreadingSettings)
    json_indent: int = 2
    codec_pipeline: CodecPipelineSettings = field(default_factory=CodecPipelineSettings)
    codecs: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_CODECS))
    buffer: str = "zarr.buffer.cpu.Buffer"
    ndbuffer: str = "zarr.buffer.cpu.NDBuffer"


def make_default_config() -> ZarrConfig:
    """Return a fresh `ZarrConfig` populated with the built-in defaults."""
    return ZarrConfig()


def _resolve_field(obj: Any, segment: str) -> str:
    """Translate a serialized key segment to the dataclass field name."""
    return _FIELD_ALIASES.get(segment, segment)


def get_path(cfg: ZarrConfig, key: str) -> Any:
    """Read a dotted-string key from a `ZarrConfig` snapshot.

    Raises
    ------
    KeyError
        If the key does not resolve to a value.
    """
    obj: Any = cfg
    segments = key.split(".")
    for i, segment in enumerate(segments):
        if isinstance(obj, Mapping):
            # remaining segments index into an open mapping (e.g. codecs.*)
            remainder = ".".join(segments[i:])
            try:
                return obj[remainder]
            except KeyError:
                raise KeyError(key) from None
        field_name = _resolve_field(obj, segment)
        if not hasattr(obj, field_name):
            raise KeyError(key)
        obj = getattr(obj, field_name)
    return obj


def replace_path(cfg: ZarrConfig, key: str, value: Any) -> ZarrConfig:
    """Return a new `ZarrConfig` with the dotted-string key set to ``value``."""
    segments = key.split(".")
    return cast(ZarrConfig, _replace_recursive(cfg, segments, value, key))


def _replace_recursive(obj: Any, segments: list[str], value: Any, key: str) -> Any:
    segment = segments[0]
    if isinstance(obj, Mapping):
        remainder = ".".join(segments)
        return {**obj, remainder: value}
    field_name = _resolve_field(obj, segment)
    if not hasattr(obj, field_name):
        raise KeyError(key)
    if len(segments) == 1:
        return replace(obj, **{field_name: value})
    child = getattr(obj, field_name)
    new_child = _replace_recursive(child, segments[1:], value, key)
    return replace(obj, **{field_name: new_child})


def to_nested_dict(cfg: ZarrConfig) -> dict[str, Any]:
    """Convert a `ZarrConfig` to a donfig-style nested dict (serialized keys)."""

    def convert(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return dict(obj)
        if hasattr(type(obj), "__dataclass_fields__"):
            out: dict[str, Any] = {}
            for f in fields(obj):
                serialized = _SERIALIZED_NAMES.get(f.name, f.name)
                out[serialized] = convert(getattr(obj, f.name))
            return out
        return obj

    return convert(cfg)  # type: ignore[no-any-return]


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

    def enable_gpu(self) -> ConfigSet:
        """
        Configure Zarr to use GPUs where possible.
        """
        return self.set(
            {"buffer": "zarr.buffer.gpu.Buffer", "ndbuffer": "zarr.buffer.gpu.NDBuffer"}
        )


# these keys were removed from the config as part of the 3.1.0 release.
# these deprecations should be removed in 3.1.1 or thereabouts.
deprecations = {
    "array.v2_default_compressor.numeric": None,
    "array.v2_default_compressor.string": None,
    "array.v2_default_compressor.bytes": None,
    "array.v2_default_filters.string": None,
    "array.v2_default_filters.bytes": None,
    "array.v3_default_filters.numeric": None,
    "array.v3_default_filters.raw": None,
    "array.v3_default_filters.bytes": None,
    "array.v3_default_serializer.numeric": None,
    "array.v3_default_serializer.string": None,
    "array.v3_default_serializer.bytes": None,
    "array.v3_default_compressors.string": None,
    "array.v3_default_compressors.bytes": None,
    "array.v3_default_compressors": None,
}

# The default configuration for zarr
config = Config(
    "zarr",
    defaults=[
        {
            "default_zarr_format": 3,
            "array": {
                "order": "C",
                "write_empty_chunks": False,
                "read_missing_chunks": True,
                "target_shard_size_bytes": None,
                "rectilinear_chunks": False,
                "sharding_coalesce_max_gap_bytes": 1 << 20,  # 1 MiB
                "sharding_coalesce_max_bytes": 16 << 20,  # 16 MiB
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
                "numcodecs.bz2": "zarr.codecs.numcodecs.BZ2",
                "numcodecs.crc32": "zarr.codecs.numcodecs.CRC32",
                "numcodecs.crc32c": "zarr.codecs.numcodecs.CRC32C",
                "numcodecs.lz4": "zarr.codecs.numcodecs.LZ4",
                "numcodecs.lzma": "zarr.codecs.numcodecs.LZMA",
                "numcodecs.zfpy": "zarr.codecs.numcodecs.ZFPY",
                "numcodecs.adler32": "zarr.codecs.numcodecs.Adler32",
                "numcodecs.astype": "zarr.codecs.numcodecs.AsType",
                "numcodecs.bitround": "zarr.codecs.numcodecs.BitRound",
                "numcodecs.blosc": "zarr.codecs.numcodecs.Blosc",
                "numcodecs.delta": "zarr.codecs.numcodecs.Delta",
                "numcodecs.fixedscaleoffset": "zarr.codecs.numcodecs.FixedScaleOffset",
                "numcodecs.fletcher32": "zarr.codecs.numcodecs.Fletcher32",
                "numcodecs.gzip": "zarr.codecs.numcodecs.GZip",
                "numcodecs.jenkins_lookup3": "zarr.codecs.numcodecs.JenkinsLookup3",
                "numcodecs.pcodec": "zarr.codecs.numcodecs.PCodec",
                "numcodecs.packbits": "zarr.codecs.numcodecs.PackBits",
                "numcodecs.shuffle": "zarr.codecs.numcodecs.Shuffle",
                "numcodecs.quantize": "zarr.codecs.numcodecs.Quantize",
                "numcodecs.zlib": "zarr.codecs.numcodecs.Zlib",
                "numcodecs.zstd": "zarr.codecs.numcodecs.Zstd",
            },
            "buffer": "zarr.buffer.cpu.Buffer",
            "ndbuffer": "zarr.buffer.cpu.NDBuffer",
        }
    ],
    deprecations=deprecations,
)


def parse_indexing_order(data: Any) -> Literal["C", "F"]:
    if data in ("C", "F"):
        return cast("Literal['C', 'F']", data)
    msg = f"Expected one of ('C', 'F'), got {data} instead."
    raise ValueError(msg)
