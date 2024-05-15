from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Type
    from zarr.abc.codec import Codec

from importlib.metadata import EntryPoint, entry_points as get_entry_points

__codec_registry: Dict[str, Type[Codec]] = {}
__lazy_load_codecs: Dict[str, EntryPoint] = {}


def _collect_entrypoints() -> Dict[str, EntryPoint]:
    entry_points = get_entry_points()
    for e in entry_points.select(group="zarr.codecs"):
        __lazy_load_codecs[e.name] = e
    return __lazy_load_codecs


def register_codec(key: str, codec_cls: Type[Codec]) -> None:
    __codec_registry[key] = codec_cls


def get_codec_class(key: str) -> Type[Codec]:
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
