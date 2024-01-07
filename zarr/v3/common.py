from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
import contextvars
import functools
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    TypedDict,
    Union,
)
from attr import frozen

import numpy as np
from cattr import Converter
from zarr.v3.types import Attributes, ChunkCoords

ZARR_JSON = "zarr.json"
ZARRAY_JSON = ".zarray"
ZGROUP_JSON = ".zgroup"
ZATTRS_JSON = ".zattrs"


def make_cattr():
    from zarr.v3.metadata.v3 import (
        DefaultChunkKeyEncoding,
        CodecMetadata,
        ChunkKeyEncoding,
        V2ChunkKeyEncoding,
    )
    from zarr.v3.codecs.registry import get_codec_metadata_class

    converter = Converter()

    def _structure_attributes(d: Dict[str, Any], _t) -> Attributes:
        return d

    converter.register_structure_hook_factory(
        lambda t: str(t)
        == "typing.Union[typing.Dict[str, ForwardRef('Attributes')], typing.List[ForwardRef('Attributes')], str, int, float, bool, NoneType]",
        lambda t: _structure_attributes,
    )

    def _structure_chunk_key_encoding_metadata(d: Dict[str, Any], _t) -> ChunkKeyEncoding:
        if d["name"] == "default":
            return converter.structure(d, DefaultChunkKeyEncoding)
        if d["name"] == "v2":
            return converter.structure(d, V2ChunkKeyEncoding)
        raise KeyError

    converter.register_structure_hook(ChunkKeyEncoding, _structure_chunk_key_encoding_metadata)

    def _structure_codec_metadata(d: Dict[str, Any], _t=None) -> CodecMetadata:
        codec_metadata_cls = get_codec_metadata_class(d["name"])
        return converter.structure(d, codec_metadata_cls)

    converter.register_structure_hook(CodecMetadata, _structure_codec_metadata)

    converter.register_structure_hook_factory(
        lambda t: str(t) == "ForwardRef('CodecMetadata')",
        lambda t: _structure_codec_metadata,
    )

    def _structure_order(d: Any, _t=None) -> Union[Literal["C", "F"], Tuple[int, ...]]:
        if d == "C":
            return "C"
        if d == "F":
            return "F"
        if isinstance(d, list):
            return tuple(d)
        raise KeyError

    converter.register_structure_hook_factory(
        lambda t: str(t) == "typing.Union[typing.Literal['C', 'F'], typing.Tuple[int, ...]]",
        lambda t: _structure_order,
    )

    # Needed for v2 fill_value
    def _structure_fill_value(d: Any, _t=None) -> Union[None, int, float]:
        if d is None:
            return None
        try:
            return int(d)
        except ValueError:
            pass
        try:
            return float(d)
        except ValueError:
            pass
        raise ValueError

    converter.register_structure_hook_factory(
        lambda t: str(t) == "typing.Union[NoneType, int, float]",
        lambda t: _structure_fill_value,
    )

    # Needed for v2 dtype
    converter.register_structure_hook(
        np.dtype,
        lambda d, _: np.dtype(d),
    )

    return converter


def product(tup: ChunkCoords) -> int:
    return functools.reduce(lambda x, y: x * y, tup, 1)


T = TypeVar("T", bound=Tuple)
V = TypeVar("V")


async def concurrent_map(
    items: List[T], func: Callable[..., Awaitable[V]], limit: Optional[int] = None
) -> List[V]:
    if limit is None:
        return await asyncio.gather(*[func(*item) for item in items])

    else:
        sem = asyncio.Semaphore(limit)

        async def run(item):
            async with sem:
                return await func(*item)

        return await asyncio.gather(*[asyncio.ensure_future(run(item)) for item in items])


async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)


@frozen
class RuntimeConfiguration:
    order: Literal["C", "F"] = "C"
    concurrency: Optional[int] = None
    asyncio_loop: Optional[AbstractEventLoop] = None


def runtime_configuration(
    order: Literal["C", "F"], concurrency: Optional[int] = None
) -> RuntimeConfiguration:
    return RuntimeConfiguration(order=order, concurrency=concurrency)


class ChunkMetadataDict(TypedDict):
    array_shape: Tuple[int, ...]
    chunk_shape: Tuple[int, ...]
    dtype: str
    fill_value: Any


class ChunkMetadata:
    array_shape: Tuple[int, ...]
    chunk_shape: Tuple[int, ...]
    # data_type: DataType
    dtype: np.dtype
    fill_value: Any

    def __init__(self, array_shape, chunk_shape, dtype, fill_value) -> None:
        self.array_shape = array_shape
        self.chunk_shape = chunk_shape
        self.dtype = dtype
        self.fill_value = fill_value

    @property
    def ndim(self) -> int:
        return len(self.array_shape)

    def to_dict(self) -> ChunkMetadataDict:
        return {
            "array_shape": self.array_shape,
            "chunk_shape": self.chunk_shape,
            "fill_value": self.fill_value,
            "dtype": self.dtype.str,
        }
