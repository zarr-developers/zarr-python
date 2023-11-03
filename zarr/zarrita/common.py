from __future__ import annotations

import asyncio
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
    Union,
)

import numpy as np
from cattr import Converter

ZARR_JSON = "zarr.json"
ZARRAY_JSON = ".zarray"
ZGROUP_JSON = ".zgroup"
ZATTRS_JSON = ".zattrs"

BytesLike = Union[bytes, bytearray, memoryview]
ChunkCoords = Tuple[int, ...]
SliceSelection = Tuple[slice, ...]
Selection = Union[slice, SliceSelection]


def make_cattr():
    from zarrita.metadata import (
        BloscCodecMetadata,
        BytesCodecMetadata,
        ChunkKeyEncodingMetadata,
        CodecMetadata,
        Crc32cCodecMetadata,
        DefaultChunkKeyEncodingMetadata,
        GzipCodecMetadata,
        ShardingCodecMetadata,
        TransposeCodecMetadata,
        V2ChunkKeyEncodingMetadata,
        ZstdCodecMetadata,
    )

    converter = Converter()

    def _structure_chunk_key_encoding_metadata(
        d: Dict[str, Any], _t
    ) -> ChunkKeyEncodingMetadata:
        if d["name"] == "default":
            return converter.structure(d, DefaultChunkKeyEncodingMetadata)
        if d["name"] == "v2":
            return converter.structure(d, V2ChunkKeyEncodingMetadata)
        raise KeyError

    converter.register_structure_hook(
        ChunkKeyEncodingMetadata, _structure_chunk_key_encoding_metadata
    )

    def _structure_codec_metadata(d: Dict[str, Any], _t=None) -> CodecMetadata:
        if d["name"] == "endian":
            d["name"] = "bytes"

        if d["name"] == "blosc":
            return converter.structure(d, BloscCodecMetadata)
        if d["name"] == "bytes":
            return converter.structure(d, BytesCodecMetadata)
        if d["name"] == "transpose":
            return converter.structure(d, TransposeCodecMetadata)
        if d["name"] == "gzip":
            return converter.structure(d, GzipCodecMetadata)
        if d["name"] == "zstd":
            return converter.structure(d, ZstdCodecMetadata)
        if d["name"] == "sharding_indexed":
            return converter.structure(d, ShardingCodecMetadata)
        if d["name"] == "crc32c":
            return converter.structure(d, Crc32cCodecMetadata)
        raise KeyError

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
        lambda t: str(t)
        == "typing.Union[typing.Literal['C', 'F'], typing.Tuple[int, ...]]",
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

        return await asyncio.gather(
            *[asyncio.ensure_future(run(item)) for item in items]
        )


async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)
