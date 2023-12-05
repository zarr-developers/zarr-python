from typing import Dict, NamedTuple, Type

from zarr.v3.abc.codec import Codec
from zarr.v3.metadata import CodecMetadata


class CodecRegistryItem(NamedTuple):
    codec_cls: Type[Codec]
    codec_metadata_cls: Type[CodecMetadata]


__codec_registry: Dict[str, CodecRegistryItem] = {}


def register_codec(
    key: str, codec_cls: Type[Codec], codec_metadata_cls: Type[CodecMetadata]
) -> None:
    __codec_registry[key] = CodecRegistryItem(codec_cls, codec_metadata_cls)


def get_codec_metadata_class(key: str) -> Type[CodecMetadata]:
    return __codec_registry[key].codec_metadata_cls


def get_codec_class(key: str) -> Type[Codec]:
    return __codec_registry[key].codec_cls
