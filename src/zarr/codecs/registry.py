from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Type
    from zarr.abc.codec import Codec

from importlib.metadata import EntryPoint, entry_points as get_entry_points


__codec_registry: Dict[str, Type[Codec]] = {}
__lazy_load_codecs: Dict[str, EntryPoint] = {}


def _collect_entrypoints() -> None:
    entry_points = get_entry_points()
    if hasattr(entry_points, "select"):
        # If entry_points() has a select method, use that. Python 3.10+
        for e in entry_points.select(group="zarr.codecs"):
            __lazy_load_codecs[e.name] = e
    else:
        # Otherwise, fallback to using get
        for e in entry_points.get("zarr.codecs", []):
            __lazy_load_codecs[e.name] = e


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
