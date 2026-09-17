from __future__ import annotations

import threading
import warnings
from collections import defaultdict
from collections.abc import Mapping
from importlib.metadata import entry_points as get_entry_points
from typing import TYPE_CHECKING, Any

from zarr.core.config import BadConfigError, config
from zarr.core.dtype import data_type_registry
from zarr.errors import UnknownCodecError, URLPipelineError, ZarrUserWarning

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from zarr.abc.codec import (
        ArrayArrayCodec,
        ArrayBytesCodec,
        BytesBytesCodec,
        Codec,
        CodecJSON_V2,
        CodecPipeline,
    )
    from zarr.abc.numcodec import Numcodec
    from zarr.abc.url_pipeline import URLPipelineAdapter
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.chunk_key_encodings import ChunkKeyEncoding
    from zarr.core.common import JSON, ZarrFormat

__all__ = [
    "Registry",
    "get_buffer_class",
    "get_chunk_key_encoding_class",
    "get_codec_class",
    "get_ndbuffer_class",
    "get_pipeline_class",
    "get_url_adapter",
    "list_url_adapter_schemes",
    "register_buffer",
    "register_chunk_key_encoding",
    "register_codec",
    "register_ndbuffer",
    "register_pipeline",
    "register_url_adapter",
]

_ZARR_CODEC_DOCS_URL = "https://zarr.readthedocs.io/en/stable/user-guide/extending/#custom-codecs"
_NUMCODECS_CODEC_DOCS_URL = (
    "https://numcodecs.readthedocs.io/en/stable/registry.html#numcodecs.registry.register_codec"
)

# Codecs zarr-python does not implement, mapped to the names of Python packages that do.
# These tables exist purely to make the "no implementation for this codec" error actionable;
# nothing here affects which codecs zarr can actually read or write. Values are what you would
# pass to `pip install`. Only add an entry you have verified against the package's declared
# entry points, and only for a package that is actually published.
#
# The two Zarr formats resolve codecs through different registries, so they get different
# tables: a name can mean one thing as a Zarr format 3 codec name and another as a Zarr
# format 2 codec id. For example, `imagecodecs_*` names are registered by
# `imagecodecs-zarr` and `virtual-tiff` under `zarr.codecs`, and by
# `imagecodecs-numcodecs` under `numcodecs.codecs`.

# Zarr format 3 codec names (entry point group "zarr.codecs").
_CODEC_PACKAGES: dict[str, tuple[str, ...]] = {
    "gribberish": ("gribberish",),
    # Verified against imagecodecs-zarr 2026.8.16's published zarr.codecs entry points:
    # https://pypi.org/project/imagecodecs-zarr/2026.8.16/
    # virtual-tiff also provides 13 of these names, plus jpeg8 and jetraw.
    # Keep exact names: neither package provides every possible imagecodecs_* name.
    "imagecodecs_aec": ("imagecodecs-zarr",),
    "imagecodecs_apng": ("imagecodecs-zarr",),
    "imagecodecs_avif": ("imagecodecs-zarr",),
    "imagecodecs_b2nd": ("imagecodecs-zarr",),
    "imagecodecs_bfloat16": ("imagecodecs-zarr",),
    "imagecodecs_bitorder": ("imagecodecs-zarr",),
    "imagecodecs_bitshuffle": ("imagecodecs-zarr",),
    "imagecodecs_blosc": ("imagecodecs-zarr",),
    "imagecodecs_blosc2": ("imagecodecs-zarr",),
    "imagecodecs_bmp": ("imagecodecs-zarr",),
    "imagecodecs_brotli": ("imagecodecs-zarr",),
    "imagecodecs_byteshuffle": ("imagecodecs-zarr",),
    "imagecodecs_bz2": ("imagecodecs-zarr",),
    "imagecodecs_ccittfax3": ("imagecodecs-zarr",),
    "imagecodecs_ccittfax4": ("imagecodecs-zarr",),
    "imagecodecs_ccittrle": ("imagecodecs-zarr",),
    "imagecodecs_checksum": ("imagecodecs-zarr",),
    "imagecodecs_chunked": ("imagecodecs-zarr",),
    "imagecodecs_cms": ("imagecodecs-zarr",),
    "imagecodecs_dds": ("imagecodecs-zarr",),
    "imagecodecs_deflate": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_delta": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_dicomrle": ("imagecodecs-zarr",),
    "imagecodecs_eer": ("imagecodecs-zarr",),
    "imagecodecs_exr": ("imagecodecs-zarr",),
    "imagecodecs_float24": ("imagecodecs-zarr",),
    "imagecodecs_floatpred": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_gif": ("imagecodecs-zarr",),
    "imagecodecs_hcomp": ("imagecodecs-zarr",),
    "imagecodecs_heif": ("imagecodecs-zarr",),
    "imagecodecs_htj2k": ("imagecodecs-zarr",),
    "imagecodecs_isal": ("imagecodecs-zarr",),
    "imagecodecs_jetraw": ("virtual-tiff",),
    "imagecodecs_jpeg": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_jpeg2k": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_jpeg8": ("virtual-tiff",),
    "imagecodecs_jpegls": ("imagecodecs-zarr",),
    "imagecodecs_jpegxl": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_jpegxr": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_jpegxs": ("imagecodecs-zarr",),
    "imagecodecs_lerc": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_ljpeg": ("imagecodecs-zarr",),
    "imagecodecs_lz4": ("imagecodecs-zarr",),
    "imagecodecs_lz4f": ("imagecodecs-zarr",),
    "imagecodecs_lz4h5": ("imagecodecs-zarr",),
    "imagecodecs_lzf": ("imagecodecs-zarr",),
    "imagecodecs_lzfse": ("imagecodecs-zarr",),
    "imagecodecs_lzham": ("imagecodecs-zarr",),
    "imagecodecs_lzma": ("imagecodecs-zarr",),
    "imagecodecs_lzo": ("imagecodecs-zarr",),
    "imagecodecs_lzw": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_meshopt": ("imagecodecs-zarr",),
    "imagecodecs_openzl": ("imagecodecs-zarr",),
    "imagecodecs_packbits": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_packints": ("imagecodecs-zarr",),
    "imagecodecs_pcodec": ("imagecodecs-zarr",),
    "imagecodecs_pcx": ("imagecodecs-zarr",),
    "imagecodecs_pglz": ("imagecodecs-zarr",),
    "imagecodecs_pixarlog": ("imagecodecs-zarr",),
    "imagecodecs_plio": ("imagecodecs-zarr",),
    "imagecodecs_png": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_qoi": ("imagecodecs-zarr",),
    "imagecodecs_quantize": ("imagecodecs-zarr",),
    "imagecodecs_rcomp": ("imagecodecs-zarr",),
    "imagecodecs_rgbe": ("imagecodecs-zarr",),
    "imagecodecs_snappy": ("imagecodecs-zarr",),
    "imagecodecs_sperr": ("imagecodecs-zarr",),
    "imagecodecs_spng": ("imagecodecs-zarr",),
    "imagecodecs_sz3": ("imagecodecs-zarr",),
    "imagecodecs_szip": ("imagecodecs-zarr",),
    "imagecodecs_tga": ("imagecodecs-zarr",),
    "imagecodecs_tiff": ("imagecodecs-zarr",),
    "imagecodecs_ultrahdr": ("imagecodecs-zarr",),
    "imagecodecs_wavpack": ("imagecodecs-zarr",),
    "imagecodecs_webp": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_wic": ("imagecodecs-zarr",),
    "imagecodecs_xor": ("imagecodecs-zarr",),
    "imagecodecs_zfp": ("imagecodecs-zarr",),
    "imagecodecs_zlib": ("imagecodecs-zarr",),
    "imagecodecs_zlibng": ("imagecodecs-zarr",),
    "imagecodecs_zopfli": ("imagecodecs-zarr",),
    "imagecodecs_zstd": ("imagecodecs-zarr", "virtual-tiff"),
    "imagecodecs_zstd1": ("imagecodecs-zarr",),
    "n5_default": ("zarr-n5",),
}

# As `_CODEC_PACKAGES`, but each key is matched against the start of the codec name. Packages
# that provide many codecs namespace them behind a shared prefix, so one entry covers them all.
_CODEC_PACKAGE_PREFIXES: dict[str, tuple[str, ...]] = {
    "any-numcodecs.": ("zarr-any-numcodecs",),
    "omfiles.": ("omfiles",),
    "virtual_tiff.": ("virtual-tiff",),
}

# Zarr format 2 codec ids (entry point group "numcodecs.codecs"). `numcodecs` itself gates
# several of its own codecs behind optional dependencies, so the package to install for those
# is an extra of numcodecs rather than a third-party distribution.
_NUMCODEC_PACKAGES: dict[str, tuple[str, ...]] = {
    "FITSAscii": ("kerchunk",),
    "FITSVarBintable": ("kerchunk",),
    "crc32c": ("numcodecs[crc32c]",),
    "fill_hdf_strings": ("kerchunk",),
    "grib": ("kerchunk",),
    "msgpack2": ("numcodecs[msgpack]",),
    "pcodec": ("numcodecs[pcodec]",),
    "rawgrib": ("gribscan",),
    "record_member": ("kerchunk",),
    "vc-delta3d": ("vc-delta3d",),
    "wavpack": ("wavpack-numcodecs",),
    "zfpy": ("numcodecs[zfpy]",),
}

# As `_NUMCODEC_PACKAGES`, but matched against the start of the codec id.
_NUMCODEC_PACKAGE_PREFIXES: dict[str, tuple[str, ...]] = {
    "gribscan.": ("gribscan",),
    "imagecodecs_": ("imagecodecs-numcodecs",),
}


def _packages_for_codec(name: str, *, zarr_format: ZarrFormat) -> tuple[str, ...]:
    """
    Names of Python packages known to provide an implementation of the codec ``name``.

    Returns an empty tuple if we don't know of any.

    Parameters
    ----------
    name : str
        The codec name (Zarr format 3) or codec id (Zarr format 2) we failed to resolve.
    zarr_format : ZarrFormat
        Which registry the codec was looked up in.
    """
    if zarr_format == 2:
        exact, prefixes = _NUMCODEC_PACKAGES, _NUMCODEC_PACKAGE_PREFIXES
    else:
        exact, prefixes = _CODEC_PACKAGES, _CODEC_PACKAGE_PREFIXES
    if name in exact:
        return exact[name]
    for prefix, packages in prefixes.items():
        if name.startswith(prefix):
            return packages
    return ()


def _missing_codec_message(name: str, *, zarr_format: ZarrFormat) -> str:
    """
    Build the error message raised when no implementation of the codec ``name`` is available.

    Parameters
    ----------
    name : str
        The codec name (Zarr format 3) or codec id (Zarr format 2) we failed to resolve.
    zarr_format : ZarrFormat
        Which registry the codec was looked up in. Zarr format 2 codecs are resolved through
        numcodecs, so that case points at the numcodecs registry rather than at zarr's.
    """
    if zarr_format == 2:
        docs_url, registry = _NUMCODECS_CODEC_DOCS_URL, "numcodecs"
    else:
        docs_url, registry = _ZARR_CODEC_DOCS_URL, "zarr"
    msg = (
        f"An implementation for codec {name!r} is not available. Register one explicitly "
        f"using the codec registry (see {docs_url}), or install a Python package that "
        f"registers a codec implementation with {registry}."
    )
    packages = _packages_for_codec(name, zarr_format=zarr_format)
    if packages:
        msg += f" Known packages supporting this codec: {', '.join(packages)}."
    return msg


class Registry[T](dict[str, type[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.lazy_load_list: list[EntryPoint] = []

    def lazy_load(self, use_entrypoint_name: bool = False) -> None:
        for e in self.lazy_load_list:
            self.register(e.load(), qualname=e.name if use_entrypoint_name else None)

        self.lazy_load_list.clear()

    def register(self, cls: type[T], qualname: str | None = None) -> None:
        if qualname is None:
            qualname = fully_qualified_name(cls)
        self[qualname] = cls


_codec_registries: dict[str, Registry[Codec]] = defaultdict(Registry)
_pipeline_registry: Registry[CodecPipeline] = Registry()
_buffer_registry: Registry[Buffer] = Registry()
_ndbuffer_registry: Registry[NDBuffer] = Registry()
_chunk_key_encoding_registry: Registry[ChunkKeyEncoding] = Registry()
_url_adapter_registry: Registry[URLPipelineAdapter] = Registry()

"""
The registry module is responsible for managing implementations of codecs,
pipelines, buffers, ndbuffers, and chunk key encodings and collecting them from entrypoints.
The implementation used is determined by the config.

The registry module is also responsible for managing dtypes.
"""


def _collect_entrypoints() -> list[Registry[Any]]:
    """
    Collects codecs, pipelines, dtypes, buffers and ndbuffers from entrypoints.
    Entry points can either be single items or groups of items.
    Allowed syntax for entry_points.txt is e.g.

        [zarr.codecs]
        gzip = package:EntrypointGzipCodec1
        [zarr.codecs.gzip]
        some_name = package:EntrypointGzipCodec2
        another = package:EntrypointGzipCodec3

        [zarr]
        buffer = package:TestBuffer1
        [zarr.buffer]
        xyz = package:TestBuffer2
        abc = package:TestBuffer3
        ...
    """
    entry_points = get_entry_points()

    _buffer_registry.lazy_load_list.extend(entry_points.select(group="zarr.buffer"))
    _buffer_registry.lazy_load_list.extend(entry_points.select(group="zarr", name="buffer"))
    _ndbuffer_registry.lazy_load_list.extend(entry_points.select(group="zarr.ndbuffer"))
    _ndbuffer_registry.lazy_load_list.extend(entry_points.select(group="zarr", name="ndbuffer"))

    data_type_registry._lazy_load_list.extend(entry_points.select(group="zarr.data_type"))
    data_type_registry._lazy_load_list.extend(entry_points.select(group="zarr", name="data_type"))

    _chunk_key_encoding_registry.lazy_load_list.extend(
        entry_points.select(group="zarr.chunk_key_encoding")
    )
    _chunk_key_encoding_registry.lazy_load_list.extend(
        entry_points.select(group="zarr", name="chunk_key_encoding")
    )

    _url_adapter_registry.lazy_load_list.extend(entry_points.select(group="zarr.url_adapters"))

    _pipeline_registry.lazy_load_list.extend(entry_points.select(group="zarr.codec_pipeline"))
    _pipeline_registry.lazy_load_list.extend(
        entry_points.select(group="zarr", name="codec_pipeline")
    )
    for e in entry_points.select(group="zarr.codecs"):
        _codec_registries[e.name].lazy_load_list.append(e)
    for group in entry_points.groups:
        if group.startswith("zarr.codecs."):
            codec_name = group.split(".")[2]
            _codec_registries[codec_name].lazy_load_list.extend(entry_points.select(group=group))
    return [
        *_codec_registries.values(),
        _pipeline_registry,
        _buffer_registry,
        _ndbuffer_registry,
        _chunk_key_encoding_registry,
        _url_adapter_registry,
    ]


def _reload_config() -> None:
    config.refresh()


def fully_qualified_name(cls: type) -> str:
    module = cls.__module__
    return f"{module}.{cls.__qualname__}"


def register_codec(key: str, codec_cls: type[Codec], *, qualname: str | None = None) -> None:
    if key not in _codec_registries:
        _codec_registries[key] = Registry()
    _codec_registries[key].register(codec_cls, qualname=qualname)


def register_pipeline(pipe_cls: type[CodecPipeline]) -> None:
    _pipeline_registry.register(pipe_cls)


def register_ndbuffer(cls: type[NDBuffer], qualname: str | None = None) -> None:
    _ndbuffer_registry.register(cls, qualname)


def register_buffer(cls: type[Buffer], qualname: str | None = None) -> None:
    _buffer_registry.register(cls, qualname)


def register_chunk_key_encoding(key: str, cls: type) -> None:
    _chunk_key_encoding_registry.register(cls, key)


def get_codec_class(key: str, reload_config: bool = False) -> type[Codec]:
    if reload_config:
        _reload_config()

    if key in _codec_registries:
        # logger.debug("Auto loading codec '%s' from entrypoint", codec_id)
        _codec_registries[key].lazy_load()

    codec_classes = _codec_registries[key]
    if not codec_classes:
        raise UnknownCodecError(_missing_codec_message(key, zarr_format=3))
    config_entry = config.get("codecs", {}).get(key)
    if config_entry is None:
        if len(codec_classes) == 1:
            return next(iter(codec_classes.values()))
        warnings.warn(
            f"Codec '{key}' not configured in config. Selecting any implementation.",
            stacklevel=2,
            category=ZarrUserWarning,
        )
        return list(codec_classes.values())[-1]
    selected_codec_cls = codec_classes.get(config_entry)
    if selected_codec_cls is None:
        # Not UnknownCodecError: the codec is known, the implementation named in the config is
        # not registered. That is a configuration problem, which is what the sibling getters in
        # this module raise BadConfigError for.
        raise BadConfigError(
            f"Codec {key!r} is configured to use the implementation {config_entry!r}, which is "
            f"not registered. Registered implementations of this codec: "
            f"{sorted(codec_classes)}."
        )
    return selected_codec_cls


def _resolve_codec(data: dict[str, JSON]) -> Codec:
    """
    Get a codec instance from a dict representation of that codec.
    """
    # TODO: narrow the type of the input to only those dicts that map on to codec class instances.
    return get_codec_class(data["name"]).from_dict(data)  # type: ignore[arg-type]


def _parse_bytes_bytes_codec(data: dict[str, JSON] | Codec) -> BytesBytesCodec:
    """
    Normalize the input to a `BytesBytesCodec` instance.
    If the input is already a `BytesBytesCodec`, it is returned as is. If the input is a dict, it
    is converted to a `BytesBytesCodec` instance via the `_resolve_codec` function.
    """
    from zarr.abc.codec import BytesBytesCodec

    if isinstance(data, dict):
        result = _resolve_codec(data)
        if not isinstance(result, BytesBytesCodec):
            msg = f"Expected a dict representation of a BytesBytesCodec; got a dict representation of a {type(result)} instead."
            raise TypeError(msg)
    else:
        if not isinstance(data, BytesBytesCodec):
            raise TypeError(f"Expected a BytesBytesCodec. Got {type(data)} instead.")
        result = data
    return result


def _parse_array_bytes_codec(data: dict[str, JSON] | Codec) -> ArrayBytesCodec:
    """
    Normalize the input to a `ArrayBytesCodec` instance.
    If the input is already a `ArrayBytesCodec`, it is returned as is. If the input is a dict, it
    is converted to a `ArrayBytesCodec` instance via the `_resolve_codec` function.
    """
    from zarr.abc.codec import ArrayBytesCodec

    if isinstance(data, dict):
        result = _resolve_codec(data)
        if not isinstance(result, ArrayBytesCodec):
            msg = f"Expected a dict representation of an ArrayBytesCodec; got a dict representation of a {type(result)} instead."
            raise TypeError(msg)
    else:
        if not isinstance(data, ArrayBytesCodec):
            raise TypeError(f"Expected an ArrayBytesCodec. Got {type(data)} instead.")
        result = data
    return result


def _parse_array_array_codec(data: dict[str, JSON] | Codec) -> ArrayArrayCodec:
    """
    Normalize the input to a `ArrayArrayCodec` instance.
    If the input is already a `ArrayArrayCodec`, it is returned as is. If the input is a dict, it
    is converted to a `ArrayArrayCodec` instance via the `_resolve_codec` function.
    """
    from zarr.abc.codec import ArrayArrayCodec

    if isinstance(data, dict):
        result = _resolve_codec(data)
        if not isinstance(result, ArrayArrayCodec):
            msg = f"Expected a dict representation of an ArrayArrayCodec; got a dict representation of a {type(result)} instead."
            raise TypeError(msg)
    else:
        if not isinstance(data, ArrayArrayCodec):
            raise TypeError(f"Expected an ArrayArrayCodec. Got {type(data)} instead.")
        result = data
    return result


def get_pipeline_class(reload_config: bool = False) -> type[CodecPipeline]:
    if reload_config:
        _reload_config()
    _pipeline_registry.lazy_load()
    path = config.get("codec_pipeline.path")
    pipeline_class = _pipeline_registry.get(path)
    if pipeline_class:
        return pipeline_class
    raise BadConfigError(
        f"Pipeline class '{path}' not found in registered pipelines: {list(_pipeline_registry)}."
    )


def get_buffer_class(reload_config: bool = False) -> type[Buffer]:
    if reload_config:
        _reload_config()
    _buffer_registry.lazy_load()

    path = config.get("buffer")
    buffer_class = _buffer_registry.get(path)
    if buffer_class:
        return buffer_class
    raise BadConfigError(
        f"Buffer class '{path}' not found in registered buffers: {list(_buffer_registry)}."
    )


def get_ndbuffer_class(reload_config: bool = False) -> type[NDBuffer]:
    if reload_config:
        _reload_config()
    _ndbuffer_registry.lazy_load()
    path = config.get("ndbuffer")
    ndbuffer_class = _ndbuffer_registry.get(path)
    if ndbuffer_class:
        return ndbuffer_class
    raise BadConfigError(
        f"NDBuffer class '{path}' not found in registered buffers: {list(_ndbuffer_registry)}."
    )


def get_chunk_key_encoding_class(key: str) -> type[ChunkKeyEncoding]:
    _chunk_key_encoding_registry.lazy_load(use_entrypoint_name=True)
    if key not in _chunk_key_encoding_registry:
        raise KeyError(
            f"Chunk key encoding '{key}' not found in registered chunk key encodings: {list(_chunk_key_encoding_registry)}."
        )
    return _chunk_key_encoding_registry[key]


def register_url_adapter(scheme: str, cls: type[URLPipelineAdapter]) -> None:
    """
    Register a [`URLPipelineAdapter`][zarr.abc.url_pipeline.URLPipelineAdapter]
    class for a URL scheme.

    Registering a scheme that already has an adapter replaces it and emits a
    [`ZarrUserWarning`][zarr.errors.ZarrUserWarning].
    """
    key = scheme.lower()
    previous = _url_adapter_registry.get(key)
    if previous is not None and previous is not cls:
        warnings.warn(
            f"URL pipeline adapter for scheme {scheme!r} is being replaced: "
            f"{fully_qualified_name(previous)} -> {fully_qualified_name(cls)}",
            category=ZarrUserWarning,
            stacklevel=2,
        )
    _url_adapter_registry.register(cls, key)


def list_url_adapter_schemes() -> set[str]:
    """
    The set of URL schemes with a registered URL pipeline adapter.

    Includes adapters advertised via not-yet-loaded `zarr.url_adapters`
    entry points; consulting this does not import any adapter code.
    Schemes are case-insensitive and reported lowercased.
    """
    return set(_url_adapter_registry) | {
        e.name.lower() for e in _url_adapter_registry.lazy_load_list
    }


_url_adapter_lock = threading.Lock()


def get_url_adapter(scheme: str) -> type[URLPipelineAdapter]:
    """
    Get the URL pipeline adapter class registered for `scheme`.

    Loads pending `zarr.url_adapters` entry points for this scheme only, so
    resolving one scheme never imports other providers' packages.
    """
    key = scheme.lower()
    # The lock keeps concurrent first-time resolutions of different schemes
    # from clobbering each other's rebuild of the pending entry-point list.
    with _url_adapter_lock:
        if key not in _url_adapter_registry:
            remaining = []
            for entry_point in _url_adapter_registry.lazy_load_list:
                if entry_point.name.lower() == key:
                    _url_adapter_registry.register(entry_point.load(), qualname=key)
                else:
                    remaining.append(entry_point)
            _url_adapter_registry.lazy_load_list[:] = remaining
    try:
        return _url_adapter_registry[key]
    except KeyError:
        registered = sorted(list_url_adapter_schemes())
        raise URLPipelineError(
            f"no URL pipeline adapter is registered for scheme {scheme!r}. "
            f"Registered schemes: {registered}. Adapters are provided by "
            "packages via the 'zarr.url_adapters' entry-point group."
        ) from None


_collect_entrypoints()


def get_numcodec(data: CodecJSON_V2[str]) -> Numcodec:
    """
    Resolve a numcodec codec from the numcodecs registry.

    This requires the Numcodecs package to be installed.

    Parameters
    ----------
    data : CodecJSON_V2
        The JSON metadata for the codec.

    Returns
    -------
    codec : Numcodec

    Raises
    ------
    UnknownCodecError
        If ``data`` carries a string ``"id"`` that is not registered with numcodecs. Any other
        failure, including a registered codec rejecting its configuration and a ``data`` that is
        not a mapping, propagates from numcodecs unchanged.

    Examples
    --------
    ```python
    from zarr.registry import get_numcodec
    codec = get_numcodec({'id': 'zlib', 'level': 1})
    codec
    # Zlib(level=1)
    ```
    """

    from numcodecs.registry import codec_registry, entries, get_codec

    # Check whether numcodecs can resolve the id *before* handing off, rather than catching what
    # `get_codec` raises. Catching cannot tell "this id is unregistered" from "a registered codec
    # rejected its configuration" or from "a wrapper codec failed to resolve an inner codec", and
    # relabelling either of those with this id would attach a package hint that is simply wrong.
    # This mirrors the two lookups `get_codec` performs (it then tests the result for
    # truthiness rather than membership, which only differs for a falsy registry value).
    # Widened to `object` deliberately: `data` is annotated as a TypedDict, but this is a public
    # function and callers pass whatever they like. numcodecs coerces with `dict(config)` and
    # raises for anything that is not a mapping, which is the behaviour to preserve.
    raw: object = data
    codec_id = raw.get("id") if isinstance(raw, Mapping) else None
    if isinstance(codec_id, str) and codec_id not in codec_registry and codec_id not in entries:
        raise UnknownCodecError(_missing_codec_message(codec_id, zarr_format=2))
    return get_codec(data)  # type: ignore[no-any-return]
