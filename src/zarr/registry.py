from __future__ import annotations

import warnings
from collections import defaultdict
from importlib.metadata import entry_points as get_entry_points
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from zarr.core.config import BadConfigError, config

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from zarr.abc.codec import Codec, CodecPipeline
    from zarr.core.buffer import Buffer, NDBuffer

__all__ = [
    "Registry",
    "register_codec",
    "register_pipeline",
    "register_buffer",
    "register_ndbuffer",
    "get_codec_class",
    "get_pipeline_class",
    "get_buffer_class",
    "get_ndbuffer_class",
]

T = TypeVar("T")


class Registry(Generic[T], dict[str, type[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.lazy_load_list: list[EntryPoint] = []

    def lazy_load(self) -> None:
        for e in self.lazy_load_list:
            self.register(e.load())
        self.lazy_load_list.clear()

    def register(self, cls: type[T]) -> None:
        self[fully_qualified_name(cls)] = cls


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
    """
    Collects codecs, pipelines, buffers and ndbuffers from entrypoints.
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

    __buffer_registry.lazy_load_list.extend(entry_points.select(group="zarr.buffer"))
    __buffer_registry.lazy_load_list.extend(entry_points.select(group="zarr", name="buffer"))
    __ndbuffer_registry.lazy_load_list.extend(entry_points.select(group="zarr.ndbuffer"))
    __ndbuffer_registry.lazy_load_list.extend(entry_points.select(group="zarr", name="ndbuffer"))
    __pipeline_registry.lazy_load_list.extend(entry_points.select(group="zarr.codec_pipeline"))
    __pipeline_registry.lazy_load_list.extend(
        entry_points.select(group="zarr", name="codec_pipeline")
    )
    for e in entry_points.select(group="zarr.codecs"):
        __codec_registries[e.name].lazy_load_list.append(e)
    for group in entry_points.groups:
        if group.startswith("zarr.codecs."):
            codec_name = group.split(".")[2]
            __codec_registries[codec_name].lazy_load_list.extend(entry_points.select(group=group))
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
    if key not in __codec_registries.keys():
        __codec_registries[key] = Registry()
    __codec_registries[key].register(codec_cls)


def register_pipeline(pipe_cls: type[CodecPipeline]) -> None:
    __pipeline_registry.register(pipe_cls)


def register_ndbuffer(cls: type[NDBuffer]) -> None:
    __ndbuffer_registry.register(cls)


def register_buffer(cls: type[Buffer]) -> None:
    __buffer_registry.register(cls)


def get_codec_class(key: str, reload_config: bool = False) -> type[Codec]:
    if reload_config:
        _reload_config()

    if key in __codec_registries:
        # logger.debug("Auto loading codec '%s' from entrypoint", codec_id)
        __codec_registries[key].lazy_load()

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
    __pipeline_registry.lazy_load()
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
    __buffer_registry.lazy_load()

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
    __ndbuffer_registry.lazy_load()
    path = config.get("ndbuffer")
    ndbuffer_class = __ndbuffer_registry.get(path)
    if ndbuffer_class:
        return ndbuffer_class
    raise BadConfigError(
        f"NDBuffer class '{path}' not found in registered buffers: {list(__ndbuffer_registry.keys())}."
    )


_collect_entrypoints()
