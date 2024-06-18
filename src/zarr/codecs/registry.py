from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.abc.codec import Codec, CodecPipeline

from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as get_entry_points

from zarr.config import BadConfigError, config

__codec_registry: dict[str, dict[str, type[Codec]]] = {}
__lazy_load_codecs: dict[str, EntryPoint] = {}
__pipeline_registry: dict[str, type[CodecPipeline]] = {}
__lazy_load_pipelines: list[EntryPoint] = []


def _collect_entrypoints() -> tuple[dict[str, EntryPoint], list[EntryPoint]]:
    entry_points = get_entry_points()
    for e in entry_points.select(group="zarr.codecs"):
        __lazy_load_codecs[e.name] = e
    for e in entry_points.select(group="zarr"):
        if e.name == "codec_pipeline":
            __lazy_load_pipelines.append(e)
    return __lazy_load_codecs, __lazy_load_pipelines


def _reload_config() -> None:
    config.refresh()


def register_codec(key: str, codec_cls: type[Codec]) -> None:
    registered_codecs = __codec_registry.get(key, {})
    registered_codecs[codec_cls.__name__] = codec_cls
    __codec_registry[key] = registered_codecs


def register_pipeline(pipe_cls: type[CodecPipeline]) -> None:
    __pipeline_registry[pipe_cls.__name__] = pipe_cls


def get_pipeline_class(reload_config: bool = False) -> type[CodecPipeline]:
    if reload_config:
        _reload_config()
    for e in __lazy_load_pipelines:
        __lazy_load_pipelines.remove(e)
        register_pipeline(e.load())

    name = config.get("codec_pipeline.name")
    pipeline_class = __pipeline_registry.get(name)
    if pipeline_class:
        return pipeline_class
    raise BadConfigError(
        f"Pipeline class '{name}' not found in registered pipelines: {list(__pipeline_registry.keys())}."
    )


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

    name = config_entry.get("name")
    selected_codec_cls = codec_classes[name]

    if selected_codec_cls:
        return selected_codec_cls
    raise KeyError(key)


_collect_entrypoints()
