from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from zarr.abc.codec import Codec, CodecPipeline
    from zarr.buffer import Buffer, NDBuffer

from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as get_entry_points

from zarr.config import BadConfigError, config

T = TypeVar("T")


class Registry(Generic[T], dict[str, type[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.lazy_load_list: list[EntryPoint] = []


__codec_registries: dict[str, Registry[Codec]] = defaultdict(Registry)
__pipeline_registry: Registry[CodecPipeline] = Registry()
__buffer_registry: Registry[Buffer] = Registry()
__ndbuffer_registry: Registry[NDBuffer] = Registry()

"""
The registry module is responsible for managing implementations of codecs, pipelines, buffers and ndbuffers and 
collecting them from entrypoints.
The implementation used is determined by the config
"""


def _collect_entrypoints() -> list[Registry[Any]]:
    """Collects codecs, pipelines, buffers and ndbuffers from entrypoints"""
    entry_points = get_entry_points()
    for e in entry_points:
        if e.matches(group="zarr", name="codec_pipeline") or e.matches(group="zarr.codec_pipeline"):
            __pipeline_registry.lazy_load_list.append(e)
        if e.matches(group="zarr", name="buffer") or e.matches(group="zarr.buffer"):
            __buffer_registry.lazy_load_list.append(e)
        if e.matches(group="zarr", name="ndbuffer") or e.matches(group="zarr.ndbuffer"):
            __ndbuffer_registry.lazy_load_list.append(e)
        if e.matches(group="zarr.codecs"):
            __codec_registries[e.name].lazy_load_list.append(e)
        if e.group.startswith("zarr.codecs."):
            codec_name = e.group.split(".")[2]
            __codec_registries[codec_name].lazy_load_list.append(e)
    return [
        *__codec_registries.values(),
        __pipeline_registry,
        __buffer_registry,
        __ndbuffer_registry,
    ]


def _reload_config() -> None:
    config.refresh()


def fully_qualified_name(cls: type) -> str:
    module = cls.__module__
    return module + "." + cls.__qualname__


def register_codec(key: str, codec_cls: type[Codec]) -> None:
    registry = __codec_registries.get(key, Registry())
    registry[fully_qualified_name(codec_cls)] = codec_cls
    __codec_registries[key] = registry


def register_pipeline(pipe_cls: type[CodecPipeline]) -> None:
    __pipeline_registry[fully_qualified_name(pipe_cls)] = pipe_cls


def register_ndbuffer(cls: type[NDBuffer]) -> None:
    __ndbuffer_registry[fully_qualified_name(cls)] = cls


def register_buffer(cls: type[Buffer]) -> None:
    __buffer_registry[fully_qualified_name(cls)] = cls


def get_codec_class(key: str, reload_config: bool = False) -> type[Codec]:
    if reload_config:
        _reload_config()

    if key in __codec_registries:
        # logger.debug("Auto loading codec '%s' from entrypoint", codec_id)
        for lazy_load_item in __codec_registries[key].lazy_load_list:
            cls = lazy_load_item.load()
            register_codec(key, cls)

    codec_classes = __codec_registries[key]
    if not codec_classes:
        raise KeyError(key)

    config_entry = config.get("codecs", {}).get(key)
    if config_entry is None:
        warnings.warn(
            f"Codec '{key}' not configured in config. Selecting any implementation.", stacklevel=2
        )
        return list(codec_classes.values())[-1]
    selected_codec_cls = codec_classes[config_entry]

    if selected_codec_cls:
        return selected_codec_cls
    raise KeyError(key)


def get_pipeline_class(reload_config: bool = False) -> type[CodecPipeline]:
    if reload_config:
        _reload_config()
    for e in __pipeline_registry.lazy_load_list:
        __pipeline_registry.lazy_load_list.remove(e)
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
    for e in __buffer_registry.lazy_load_list:
        __buffer_registry.lazy_load_list.remove(e)
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
    for e in __ndbuffer_registry.lazy_load_list:
        __ndbuffer_registry.lazy_load_list.remove(e)
        register_ndbuffer(e.load())
    path = config.get("ndbuffer")
    ndbuffer_class = __ndbuffer_registry.get(path)
    if ndbuffer_class:
        return ndbuffer_class
    raise BadConfigError(
        f"NDBuffer class '{path}' not found in registered buffers: {list(__ndbuffer_registry.keys())}."
    )


_collect_entrypoints()
