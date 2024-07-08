from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.abc.codec import Codec, CodecPipeline
    from zarr.buffer import Buffer, NDBuffer

from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as get_entry_points

from zarr.config import BadConfigError, config

__codec_registry: dict[str, dict[str, type[Codec]]] = {}
__lazy_load_codecs: dict[str, EntryPoint] = {}
__pipeline_registry: dict[str, type[CodecPipeline]] = {}
__lazy_load_pipelines: list[EntryPoint] = []
__buffer_registry: dict[str, type[Buffer]] = {}
__lazy_load_buffer: list[EntryPoint] = []
__ndbuffer_registry: dict[str, type[NDBuffer]] = {}
__lazy_load_ndbuffer: list[EntryPoint] = []

"""
The registry module is responsible for managing implementations of codecs, pipelines, buffers and ndbuffers and 
collecting them from entrypoints.
The implementation used is determined by the config
"""


def _collect_entrypoints() -> (
    tuple[dict[str, EntryPoint], list[EntryPoint], list[EntryPoint], list[EntryPoint]]
):
    """Collects codecs, pipelines, buffers and ndbuffers from entrypoints"""
    entry_points = get_entry_points()
    for e in entry_points.select(group="zarr.codecs"):
        __lazy_load_codecs[e.name] = e
    for e in entry_points.select(group="zarr"):
        if e.name == "codec_pipeline":
            __lazy_load_pipelines.append(e)
        if e.name == "buffer":
            __lazy_load_buffer.append(e)
        if e.name == "ndbuffer":
            __lazy_load_ndbuffer.append(e)

    return __lazy_load_codecs, __lazy_load_pipelines, __lazy_load_buffer, __lazy_load_ndbuffer


def _reload_config() -> None:
    config.refresh()


def fully_qualified_name(cls: type) -> str:
    module = cls.__module__
    return module + "." + cls.__qualname__


def register_codec(key: str, codec_cls: type[Codec]) -> None:
    registered_codecs = __codec_registry.get(key, {})
    registered_codecs[fully_qualified_name(codec_cls)] = codec_cls
    __codec_registry[key] = registered_codecs


def register_pipeline(pipe_cls: type[CodecPipeline]) -> None:
    __pipeline_registry[fully_qualified_name(pipe_cls)] = pipe_cls


def register_ndbuffer(cls: type[NDBuffer]) -> None:
    __ndbuffer_registry[fully_qualified_name(cls)] = cls


def register_buffer(cls: type[Buffer]) -> None:
    __buffer_registry[fully_qualified_name(cls)] = cls


def get_codec_class(key: str, reload_config: bool = False) -> type[Codec]:
    if reload_config:
        _reload_config()

    if key in __lazy_load_codecs:
        # logger.debug("Auto loading codec '%s' from entrypoint", codec_id)
        cls = __lazy_load_codecs[key].load()
        register_codec(key, cls)

    codec_classes = __codec_registry[key]
    if not codec_classes:
        raise KeyError(key)

    config_entry = config.get("codecs", {}).get(key)
    if config_entry is None:
        warnings.warn(
            f"Codec '{key}' not configured in config. Selecting any implementation.", stacklevel=2
        )
        return list(codec_classes.values())[-1]
    print(f"{codec_classes=}")
    print(f"{config_entry=}")
    selected_codec_cls = codec_classes[config_entry]

    if selected_codec_cls:
        return selected_codec_cls
    raise KeyError(key)


def get_pipeline_class(reload_config: bool = False) -> type[CodecPipeline]:
    if reload_config:
        _reload_config()
    for e in __lazy_load_pipelines:
        __lazy_load_pipelines.remove(e)
        register_pipeline(e.load())
    path = config.get("codec_pipeline.path")
    pipeline_class = __pipeline_registry.get(path)
    if pipeline_class:
        return pipeline_class
    raise BadConfigError(
        f"Pipeline class '{path}' not found in registered pipelines: {list(__pipeline_registry.keys())}."
    )


def get_buffer_class(reload_config: bool = False) -> type[Buffer]:
    if reload_config:
        _reload_config()
    for e in __lazy_load_buffer:
        __lazy_load_buffer.remove(e)
        register_buffer(e.load())
    path = config.get("buffer")
    buffer_class = __buffer_registry.get(path)
    if buffer_class:
        return buffer_class
    raise BadConfigError(
        f"Buffer class '{path}' not found in registered buffers: {list(__buffer_registry.keys())}."
    )


def get_ndbuffer_class(reload_config: bool = False) -> type[NDBuffer]:
    if reload_config:
        _reload_config()
    for e in __lazy_load_ndbuffer:
        __lazy_load_ndbuffer.remove(e)
        register_ndbuffer(e.load())
    path = config.get("ndbuffer")
    ndbuffer_class = __ndbuffer_registry.get(path)
    if ndbuffer_class:
        return ndbuffer_class
    raise BadConfigError(
        f"NDBuffer class '{path}' not found in registered buffers: {list(__ndbuffer_registry.keys())}."
    )


_collect_entrypoints()
