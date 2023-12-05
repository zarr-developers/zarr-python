from __future__ import annotations

from typing import Dict, NamedTuple, Type
from importlib.metadata import EntryPoint, entry_points as get_entry_points

from zarr.v3.abc.codec import Codec
from zarr.v3.metadata import CodecMetadata


class CodecRegistryItem(NamedTuple):
    codec_cls: Type[Codec]
    codec_metadata_cls: Type[CodecMetadata]


__codec_registry: Dict[str, CodecRegistryItem] = {}
__lazy_load_codecs: Dict[str, EntryPoint] = {}


def _collect_entrypoints() -> None:
    entry_points = get_entry_points()
    print(entry_points.keys())
    if hasattr(entry_points, "select"):
        # If entry_points() has a select method, use that. Python 3.10+
        for e in entry_points.select(group="zarr.codecs"):
            __lazy_load_codecs[e.name] = e
    else:
        # Otherwise, fallback to using get
        for e in entry_points.get("zarr.codecs", []):
            __lazy_load_codecs[e.name] = e


def register_codec(key: str, codec_cls: Type[Codec]) -> None:
    __codec_registry[key] = CodecRegistryItem(codec_cls, codec_cls.get_metadata_class())


def _get_codec_item(key: str) -> CodecRegistryItem:
    item = __codec_registry.get(key)
    if item is None:
        if key in __lazy_load_codecs:
            # logger.debug("Auto loading codec '%s' from entrypoint", codec_id)
            cls = __lazy_load_codecs[key].load()
            register_codec(key, cls)
            item = __codec_registry.get(key)
    if item:
        return item
    raise KeyError


def get_codec_metadata_class(key: str) -> Type[CodecMetadata]:
    return _get_codec_item(key).codec_metadata_cls


def get_codec_class(key: str) -> Type[Codec]:
    return _get_codec_item(key).codec_cls


_collect_entrypoints()
