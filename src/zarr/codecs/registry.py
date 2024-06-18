from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.abc.codec import Codec

from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as get_entry_points

from zarr.config import config

__codec_registry: dict[str, type[Codec]] = {}
__lazy_load_codecs: dict[str, EntryPoint] = {}

all_codecs = {}


def _collect_entrypoints() -> dict[str, EntryPoint]:
    entry_points = get_entry_points()
    for e in entry_points.select(group="zarr.codecs"):
        __lazy_load_codecs[e.name] = e
    return __lazy_load_codecs


def _reload_config() -> None:
    config.refresh()
    for codec_cls, key in all_codecs.items():
        register_codec(key, codec_cls)


def register_codec(key: str, codec_cls: type[Codec]) -> None:
    all_codecs[codec_cls] = key

    selected_codec = config.get("codecs", {}).get(key)
    if selected_codec is None:
        raise ValueError(f"Codec '{key}' not found in config.")
    name = selected_codec.get("name")
    if codec_cls.__name__ == name:
        __codec_registry[key] = codec_cls


def get_codec_class(key: str, reload_config=False) -> type[Codec]:
    if reload_config:
        _reload_config()
    item = __codec_registry.get(key)
    if item is None:
        if key in __lazy_load_codecs:
            # logger.debug("Auto loading codec '%s' from entrypoint", codec_id)
            cls = __lazy_load_codecs[key].load()
            register_codec(key, cls)
            item = __codec_registry.get(key)
    if item:
        return item
    raise KeyError(key)


_collect_entrypoints()
