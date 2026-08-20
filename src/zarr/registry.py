from __future__ import annotations

import warnings
from collections import defaultdict
from importlib.metadata import entry_points as get_entry_points
from typing import TYPE_CHECKING, Any

from zarr.core.config import BadConfigError, config
from zarr.core.dtype import data_type_registry
from zarr.errors import UnknownCodecError, ZarrUserWarning

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
    "register_buffer",
    "register_chunk_key_encoding",
    "register_codec",
    "register_ndbuffer",
    "register_pipeline",
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
# format 2 codec id. `imagecodecs_*` is exactly that -- `virtual-tiff` declares those names
# under `zarr.codecs`, while `imagecodecs-numcodecs` declares them under `numcodecs.codecs`.

# Zarr format 3 codec names (entry point group "zarr.codecs").
_CODEC_PACKAGES: dict[str, tuple[str, ...]] = {
    "gribberish": ("gribberish",),
    "n5_default": ("zarr-n5",),
}

# As `_CODEC_PACKAGES`, but each key is matched against the start of the codec name. Packages
# that provide many codecs namespace them behind a shared prefix, so one entry covers them all.
_CODEC_PACKAGE_PREFIXES: dict[str, tuple[str, ...]] = {
    "any-numcodecs.": ("zarr-any-numcodecs",),
    "imagecodecs_": ("virtual-tiff",),
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


def _is_missing_numcodec_error(exc: ValueError) -> bool:
    """
    Whether ``exc`` from ``numcodecs.registry.get_codec`` means "this codec isn't registered".

    numcodecs 0.15.1 added ``numcodecs.errors.UnknownCodecError`` for this. Older versions,
    down to the 0.14 floor zarr supports, raise a plain ``ValueError`` instead, so fall back to
    matching the message numcodecs has used throughout.

    Parameters
    ----------
    exc : ValueError
        The exception ``numcodecs.registry.get_codec`` raised.
    """
    try:
        from numcodecs.errors import UnknownCodecError as NumcodecsUnknownCodecError
    except ImportError:  # numcodecs < 0.15.1
        return str(exc).startswith("codec not available")
    return isinstance(exc, NumcodecsUnknownCodecError)


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
        If no implementation of the codec id in ``data`` is registered with numcodecs. When
        ``data`` carries no string ``"id"`` there is nothing to look up, and numcodecs' own
        error propagates unchanged instead.

    Examples
    --------
    ```python
    from zarr.registry import get_numcodec
    codec = get_numcodec({'id': 'zlib', 'level': 1})
    codec
    # Zlib(level=1)
    ```
    """

    from numcodecs.registry import get_codec

    try:
        return get_codec(data)  # type: ignore[no-any-return]
    except ValueError as e:
        # Read the codec id from the input rather than from the exception: numcodecs sets its
        # `codec_id` attribute to the repr of the id ("'wavpack'") rather than the id itself,
        # and older numcodecs versions raise a plain ValueError that carries no id at all.
        codec_id = data.get("id")
        if not _is_missing_numcodec_error(e) or not isinstance(codec_id, str):
            raise
        raise UnknownCodecError(_missing_codec_message(codec_id, zarr_format=2)) from e
