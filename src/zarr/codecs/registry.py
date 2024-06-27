from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.abc.codec import Codec

from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as get_entry_points

__codec_registry: dict[str, type[Codec]] = {}
__lazy_load_codecs: dict[str, EntryPoint] = {}


def _collect_entrypoints() -> dict[str, EntryPoint]:
    entry_points = get_entry_points()
    for e in entry_points.select(group="zarr.codecs"):
        __lazy_load_codecs[e.name] = e
    return __lazy_load_codecs


def register_codec(key: str, codec_cls: type[Codec]) -> None:
    __codec_registry[key] = codec_cls


def get_codec_class(key: str) -> type[Codec]:
    item = __codec_registry.get(key)
    if item is None and key in __lazy_load_codecs:
        # logger.debug("Auto loading codec '%s' from entrypoint", codec_id)
        cls = __lazy_load_codecs[key].load()
        register_codec(key, cls)
        item = __codec_registry.get(key)
    if item:
        return item
    raise KeyError(key)


_collect_entrypoints()
