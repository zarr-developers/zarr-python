from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Final, Literal, Self, cast

from typing_extensions import TypedDict

from zarr.core.common import (
    MemoryOrder,
    parse_bool,
    parse_fill_value,
    parse_order,
    parse_shapelike,
)
from zarr.core.config import config as zarr_config

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import NotRequired

    from zarr.abc.codec import (
        ArrayArrayCodec,
        ArrayBytesCodec,
        BytesBytesCodec,
        Codec,
        CodecPipeline,
    )
    from zarr.core.buffer import BufferPrototype
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


class CodecPipelineRequest(TypedDict):
    """
    A dictionary model of a request for a codec pipeline.
    """

    class_path: str
    options: NotRequired[dict[str, object]]


class ArrayConfigParams(TypedDict, closed=True):  # type: ignore[call-arg]
    """
    A TypedDict model of the attributes of an ArrayConfig class.
    """

    order: MemoryOrder
    write_empty_chunks: bool
    read_missing_chunks: bool
    codec_class_map: Mapping[str, type[ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec]]
    codec_pipeline_class: type[CodecPipeline]


class ArrayConfigRequest(TypedDict, closed=True):  # type: ignore[call-arg]
    """
    A TypedDict model of the attributes of an ArrayConfig class, but with no required fields.
    This allows for partial construction of an ArrayConfig, with the assumption that the unset
    keys will be taken from a global configuration.
    """

    order: NotRequired[MemoryOrder]
    write_empty_chunks: NotRequired[bool]
    read_missing_chunks: NotRequired[bool]
    codec_class_map: NotRequired[
        Mapping[str, type[ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec]]
    ]
    codec_pipeline_class: NotRequired[type[CodecPipeline]]


ArrayConfigKeys = Literal[
    "order", "write_empty_chunks", "read_missing_chunks", "codec_class_map", "codec_pipeline_class"
]

ARRAY_CONFIG_PARAMS_KEYS: Final[set[str]] = {
    "order",
    "write_empty_chunks",
    "read_missing_chunks",
    "codec_class_map",
    "codec_pipeline_class",
}
ARRAY_CONFIG_PARAMS_KEYS_STATIC: Final[set[str]] = {
    "order",
    "write_empty_chunks",
    "read_missing_chunks",
}
"""The keys of the ArrayConfigParams object that are static and retrievable from the config"""


@dataclass(frozen=True)
class ArrayConfig:
    """
    A model of the runtime configuration of an array.

    Parameters
    ----------
    order : MemoryOrder
        The memory layout of the arrays returned when reading data from the store.
    write_empty_chunks : bool
        If True, empty chunks will be written to the store.
    read_missing_chunks : bool, default is True
        If True, missing chunks will be filled with the array's fill value on read.
        If False, reading missing chunks will raise a ``ChunkNotFoundError``.
    codec_class_map : Mapping[str, object] | None, default is None
        A request for a codec name : codec class mapping that defines the codec classes available
        for array creation. Defaults to `None`, in which case a default collection of codecs
        is retrieved from the global config object.
    codec_pipeline_class : CodecPipelineRequest | None, default = None
        A request for a codec pipeline class to be used for orchestrating chunk encoding and
        decoding. Defaults to `None`, in which case the default codec pipeline request
        is retrieved from information in the global config object.

    Attributes
    ----------
    order : MemoryOrder
        The memory layout of the arrays returned when reading data from the store.
    write_empty_chunks : bool
        If True, empty chunks will be written to the store.
    read_missing_chunks : bool
        If True, missing chunks will be filled with the array's fill value on read.
        If False, reading missing chunks will raise a ``ChunkNotFoundError``.
    codec_class_map : Mapping[str, object]
        A codec name : codec class mapping that defines the codec classes available
        for array creation.
    codec_pipeline_class : type[CodecPipeline]
        A codec pipeline class that will be used for orchestrating chunk encoding and
        decoding.
    """

    order: MemoryOrder
    write_empty_chunks: bool
    read_missing_chunks: bool
    codec_class_map: Mapping[str, type[Codec]]
    codec_pipeline_class: type[CodecPipeline]

    def __init__(
        self,
        order: MemoryOrder,
        write_empty_chunks: bool,
        *,
        read_missing_chunks: bool = True,
        codec_class_map: Mapping[str, type[ArrayBytesCodec | ArrayArrayCodec | BytesBytesCodec]]
        | None = None,
        codec_pipeline_class: type[CodecPipeline] | None = None,
    ) -> None:
        order_parsed = parse_order(order)
        write_empty_chunks_parsed = parse_bool(write_empty_chunks)
        read_missing_chunks_parsed = parse_bool(read_missing_chunks)
        codec_class_map_parsed = parse_codec_class_map(codec_class_map)
        codec_pipeline_class_parsed = parse_codec_pipeline_class(codec_pipeline_class)

        object.__setattr__(self, "order", order_parsed)
        object.__setattr__(self, "write_empty_chunks", write_empty_chunks_parsed)
        object.__setattr__(self, "read_missing_chunks", read_missing_chunks_parsed)
        object.__setattr__(self, "codec_class_map", codec_class_map_parsed)
        object.__setattr__(self, "codec_pipeline_class", codec_pipeline_class_parsed)

    @classmethod
    def from_dict(cls, data: ArrayConfigRequest) -> Self:
        """
        Create an ArrayConfig from a dict. The keys of that dict are a subset of the
        attributes of the ArrayConfig class. Any keys missing from that dict will be set to the
        the values in the ``array`` namespace of ``zarr.config``.
        """
        kwargs_out: ArrayConfigRequest = {}
        for f in fields(ArrayConfig):
            field_name = cast(
                "Literal['order', 'write_empty_chunks', 'read_missing_chunks', 'codec_class_map', 'codec_pipeline_class']",
                f.name,
            )
            if field_name not in data:
                if field_name in ARRAY_CONFIG_PARAMS_KEYS_STATIC:
                    kwargs_out[field_name] = zarr_config.get(f"array.{field_name}")
                elif field_name == "codec_class_map":
                    kwargs_out["codec_class_map"] = parse_codec_class_map(None)
                elif field_name == "codec_pipeline_class":
                    kwargs_out["codec_pipeline_class"] = parse_codec_pipeline_class(None)
            else:
                kwargs_out[field_name] = data[field_name]
        return cls(**kwargs_out)

    def to_dict(self) -> ArrayConfigParams:
        """
        Serialize an instance of this class to a dict.
        """
        return {
            "order": self.order,
            "write_empty_chunks": self.write_empty_chunks,
            "read_missing_chunks": self.read_missing_chunks,
            "codec_class_map": self.codec_class_map,
            "codec_pipeline_class": self.codec_pipeline_class,
        }


ArrayConfigLike = ArrayConfig | ArrayConfigRequest


def parse_codec_pipeline_class(obj: type[CodecPipeline] | None) -> type[CodecPipeline]:
    if obj is None:
        from zarr.registry import get_pipeline_class

        return get_pipeline_class()
    return obj


def parse_codec_class_map(obj: Mapping[str, type[Codec]] | None) -> Mapping[str, type[Codec]]:
    """
    Convert a request for a codec class map into an actual Mapping[str, type[Codec]].
    If the input is `None`, build the map from the codec registry.
    """
    if obj is None:
        from zarr.registry import get_codec_class

        name_map: dict[str, str] = zarr_config.get("codecs", {})
        return {key: get_codec_class(key) for key in name_map}
    return obj


def parse_array_config(data: ArrayConfigLike | None) -> ArrayConfig:
    """
    Convert various types of data to an ArrayConfig.
    """
    if data is None:
        return ArrayConfig.from_dict({})
    elif isinstance(data, ArrayConfig):
        return data
    else:
        return ArrayConfig.from_dict(data)


@dataclass(frozen=True)
class ArraySpecConfig:
    order: MemoryOrder
    write_empty_chunks: bool
    read_missing_chunks: bool = False


@dataclass(frozen=True)
class ArraySpec:
    shape: tuple[int, ...]
    dtype: ZDType[TBaseDType, TBaseScalar]
    fill_value: Any
    config: ArraySpecConfig
    prototype: BufferPrototype

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        fill_value: Any,
        config: ArraySpecConfig,
        prototype: BufferPrototype,
    ) -> None:
        shape_parsed = parse_shapelike(shape)
        fill_value_parsed = parse_fill_value(fill_value)
        assert isinstance(config, ArraySpecConfig)
        object.__setattr__(self, "shape", shape_parsed)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "fill_value", fill_value_parsed)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "prototype", prototype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def order(self) -> MemoryOrder:
        return self.config.order
