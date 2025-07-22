from __future__ import annotations

import math
import os
import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import HealthCheck, Verbosity, settings

from zarr import AsyncGroup, config
from zarr.abc.store import Store
from zarr.codecs.sharding import ShardingCodec, ShardingCodecIndexLocation
from zarr.core.array import (
    _parse_chunk_encoding_v2,
    _parse_chunk_encoding_v3,
    _parse_chunk_key_encoding,
)
from zarr.core.chunk_grids import RegularChunkGrid, _auto_partition
from zarr.core.common import JSON, DimensionNames, parse_shapelike
from zarr.core.config import config as zarr_config
from zarr.core.dtype import (
    get_data_type_from_native_dtype,
)
from zarr.core.dtype.common import HasItemSize
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.storage import FsspecStore, LocalStore, MemoryStore, StorePath, ZipStore

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, Literal

    from _pytest.compat import LEGACY_PATH

    from zarr.abc.codec import Codec
    from zarr.core.array import CompressorsLike, FiltersLike, SerializerLike, ShardsLike
    from zarr.core.chunk_key_encodings import ChunkKeyEncoding, ChunkKeyEncodingLike
    from zarr.core.common import ChunkCoords, MemoryOrder, ShapeLike, ZarrFormat
    from zarr.core.dtype.wrapper import ZDType


async def parse_store(
    store: Literal["local", "memory", "fsspec", "zip"], path: str
) -> LocalStore | MemoryStore | FsspecStore | ZipStore:
    if store == "local":
        return await LocalStore.open(path)
    if store == "memory":
        return await MemoryStore.open()
    if store == "fsspec":
        return await FsspecStore.open(url=path)
    if store == "zip":
        return await ZipStore.open(path + "/zarr.zip", mode="w")
    raise AssertionError


@pytest.fixture(params=[str, pathlib.Path])
def path_type(request: pytest.FixtureRequest) -> Any:
    return request.param


# todo: harmonize this with local_store fixture
@pytest.fixture
async def store_path(tmpdir: LEGACY_PATH) -> StorePath:
    store = await LocalStore.open(str(tmpdir))
    return StorePath(store)


@pytest.fixture
async def local_store(tmpdir: LEGACY_PATH) -> LocalStore:
    return await LocalStore.open(str(tmpdir))


@pytest.fixture
async def remote_store(url: str) -> FsspecStore:
    return await FsspecStore.open(url)


@pytest.fixture
async def memory_store() -> MemoryStore:
    return await MemoryStore.open()


@pytest.fixture
async def zip_store(tmpdir: LEGACY_PATH) -> ZipStore:
    return await ZipStore.open(str(tmpdir / "zarr.zip"), mode="w")


@pytest.fixture
async def store(request: pytest.FixtureRequest, tmpdir: LEGACY_PATH) -> Store:
    param = request.param
    return await parse_store(param, str(tmpdir))


@pytest.fixture
async def store2(request: pytest.FixtureRequest, tmpdir: LEGACY_PATH) -> Store:
    """Fixture to create a second store for testing copy operations between stores"""
    param = request.param
    store2_path = tmpdir.mkdir("store2")
    return await parse_store(param, str(store2_path))


@pytest.fixture(params=["local", "memory", "zip"])
def sync_store(request: pytest.FixtureRequest, tmp_path: LEGACY_PATH) -> Store:
    result = sync(parse_store(request.param, str(tmp_path)))
    if not isinstance(result, Store):
        raise TypeError("Wrong store class returned by test fixture! got " + result + " instead")
    return result


@dataclass
class AsyncGroupRequest:
    zarr_format: ZarrFormat
    store: Literal["local", "fsspec", "memory", "zip"]
    attributes: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
async def async_group(request: pytest.FixtureRequest, tmpdir: LEGACY_PATH) -> AsyncGroup:
    param: AsyncGroupRequest = request.param

    store = await parse_store(param.store, str(tmpdir))
    return await AsyncGroup.from_store(
        store,
        attributes=param.attributes,
        zarr_format=param.zarr_format,
        overwrite=False,
    )


@pytest.fixture(params=["numpy", "cupy"])
def xp(request: pytest.FixtureRequest) -> Any:
    """Fixture to parametrize over numpy-like libraries"""

    if request.param == "cupy":
        request.node.add_marker(pytest.mark.gpu)

    return pytest.importorskip(request.param)


@pytest.fixture(autouse=True)
def reset_config() -> Generator[None, None, None]:
    config.reset()
    yield
    config.reset()


@dataclass
class ArrayRequest:
    shape: ChunkCoords
    dtype: str
    order: MemoryOrder


@pytest.fixture
def array_fixture(request: pytest.FixtureRequest) -> npt.NDArray[Any]:
    array_request: ArrayRequest = request.param
    return (
        np.arange(np.prod(array_request.shape))
        .reshape(array_request.shape, order=array_request.order)
        .astype(array_request.dtype)
    )


@pytest.fixture(params=(2, 3), ids=["zarr2", "zarr3"])
def zarr_format(request: pytest.FixtureRequest) -> ZarrFormat:
    if request.param == 2:
        return 2
    elif request.param == 3:
        return 3
    msg = f"Invalid zarr format requested. Got {request.param}, expected on of (2,3)."
    raise ValueError(msg)


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--run-slow-hypothesis",
        action="store_true",
        default=False,
        help="run slow hypothesis tests",
    )


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    if config.getoption("--run-slow-hypothesis"):
        return
    skip_slow_hyp = pytest.mark.skip(reason="need --run-slow-hypothesis option to run")
    for item in items:
        if "slow_hypothesis" in item.keywords:
            item.add_marker(skip_slow_hyp)


settings.register_profile(
    "default",
    parent=settings.get_profile("default"),
    max_examples=300,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    deadline=None,
    verbosity=Verbosity.verbose,
)
settings.register_profile(
    "ci",
    parent=settings.get_profile("ci"),
    max_examples=300,
    derandomize=True,  # more like regression testing
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)
settings.register_profile(
    "nightly",
    max_examples=500,
    parent=settings.get_profile("ci"),
    derandomize=False,
    stateful_step_count=100,
)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))


# TODO: uncomment these overrides when we can get mypy to accept them
"""
@overload
def create_array_metadata(
    *,
    shape: ShapeLike,
    dtype: npt.DTypeLike,
    chunks: ChunkCoords | Literal["auto"],
    shards: None,
    filters: FiltersLike,
    compressors: CompressorsLike,
    serializer: SerializerLike,
    fill_value: Any | None,
    order: MemoryOrder | None,
    zarr_format: Literal[2],
    attributes: dict[str, JSON] | None,
    chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None,
    dimension_names: None,
) -> ArrayV2Metadata: ...


@overload
def create_array_metadata(
    *,
    shape: ShapeLike,
    dtype: npt.DTypeLike,
    chunks: ChunkCoords | Literal["auto"],
    shards: ShardsLike | None,
    filters: FiltersLike,
    compressors: CompressorsLike,
    serializer: SerializerLike,
    fill_value: Any | None,
    order: None,
    zarr_format: Literal[3],
    attributes: dict[str, JSON] | None,
    chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None,
    dimension_names: Iterable[str] | None,
) -> ArrayV3Metadata: ...
"""


def create_array_metadata(
    *,
    shape: ShapeLike,
    dtype: npt.DTypeLike,
    chunks: ChunkCoords | Literal["auto"] = "auto",
    shards: ShardsLike | None = None,
    filters: FiltersLike = "auto",
    compressors: CompressorsLike = "auto",
    serializer: SerializerLike = "auto",
    fill_value: Any = 0,
    order: MemoryOrder | None = None,
    zarr_format: ZarrFormat,
    attributes: dict[str, JSON] | None = None,
    chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None = None,
    dimension_names: DimensionNames = None,
) -> ArrayV2Metadata | ArrayV3Metadata:
    """
    Create array metadata
    """
    dtype_parsed = get_data_type_from_native_dtype(dtype)
    shape_parsed = parse_shapelike(shape)
    chunk_key_encoding_parsed = _parse_chunk_key_encoding(
        chunk_key_encoding, zarr_format=zarr_format
    )
    item_size = 1
    if isinstance(dtype_parsed, HasItemSize):
        item_size = dtype_parsed.item_size
    shard_shape_parsed, chunk_shape_parsed = _auto_partition(
        array_shape=shape_parsed,
        shard_shape=shards,
        chunk_shape=chunks,
        item_size=item_size,
    )

    if order is None:
        order_parsed = zarr_config.get("array.order")
    else:
        order_parsed = order
    chunks_out: tuple[int, ...]

    if zarr_format == 2:
        filters_parsed, compressor_parsed = _parse_chunk_encoding_v2(
            compressor=compressors, filters=filters, dtype=dtype_parsed
        )
        return ArrayV2Metadata(
            shape=shape_parsed,
            dtype=dtype_parsed,
            chunks=chunk_shape_parsed,
            order=order_parsed,
            dimension_separator=chunk_key_encoding_parsed.separator,
            fill_value=fill_value,
            compressor=compressor_parsed,
            filters=filters_parsed,
            attributes=attributes,
        )
    elif zarr_format == 3:
        array_array, array_bytes, bytes_bytes = _parse_chunk_encoding_v3(
            compressors=compressors,
            filters=filters,
            serializer=serializer,
            dtype=dtype_parsed,
        )

        sub_codecs: tuple[Codec, ...] = (*array_array, array_bytes, *bytes_bytes)
        codecs_out: tuple[Codec, ...]
        if shard_shape_parsed is not None:
            index_location = None
            if isinstance(shards, dict):
                index_location = ShardingCodecIndexLocation(shards.get("index_location", None))
            if index_location is None:
                index_location = ShardingCodecIndexLocation.end
            sharding_codec = ShardingCodec(
                chunk_shape=chunk_shape_parsed,
                codecs=sub_codecs,
                index_location=index_location,
            )
            sharding_codec.validate(
                shape=chunk_shape_parsed,
                dtype=dtype_parsed,
                chunk_grid=RegularChunkGrid(chunk_shape=shard_shape_parsed),
            )
            codecs_out = (sharding_codec,)
            chunks_out = shard_shape_parsed
        else:
            chunks_out = chunk_shape_parsed
            codecs_out = sub_codecs

        return ArrayV3Metadata(
            shape=shape_parsed,
            data_type=dtype_parsed,
            chunk_grid=RegularChunkGrid(chunk_shape=chunks_out),
            chunk_key_encoding=chunk_key_encoding_parsed,
            fill_value=fill_value,
            codecs=codecs_out,
            attributes=attributes,
            dimension_names=dimension_names,
        )

    raise ValueError(f"Invalid Zarr format: {zarr_format}")


# TODO: uncomment these overrides when we can get mypy to accept them
"""
@overload
def meta_from_array(
    array: np.ndarray[Any, Any],
    chunks: ChunkCoords | Literal["auto"],
    shards: None,
    filters: FiltersLike,
    compressors: CompressorsLike,
    serializer: SerializerLike,
    fill_value: Any | None,
    order: MemoryOrder | None,
    zarr_format: Literal[2],
    attributes: dict[str, JSON] | None,
    chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None,
    dimension_names: Iterable[str] | None,
) -> ArrayV2Metadata: ...


@overload
def meta_from_array(
    array: np.ndarray[Any, Any],
    chunks: ChunkCoords | Literal["auto"],
    shards: ShardsLike | None,
    filters: FiltersLike,
    compressors: CompressorsLike,
    serializer: SerializerLike,
    fill_value: Any | None,
    order: None,
    zarr_format: Literal[3],
    attributes: dict[str, JSON] | None,
    chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None,
    dimension_names: Iterable[str] | None,
) -> ArrayV3Metadata: ...

"""


def meta_from_array(
    array: np.ndarray[Any, Any],
    *,
    chunks: ChunkCoords | Literal["auto"] = "auto",
    shards: ShardsLike | None = None,
    filters: FiltersLike = "auto",
    compressors: CompressorsLike = "auto",
    serializer: SerializerLike = "auto",
    fill_value: Any = 0,
    order: MemoryOrder | None = None,
    zarr_format: ZarrFormat = 3,
    attributes: dict[str, JSON] | None = None,
    chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None = None,
    dimension_names: DimensionNames = None,
) -> ArrayV3Metadata | ArrayV2Metadata:
    """
    Create array metadata from an array
    """
    return create_array_metadata(
        shape=array.shape,
        dtype=array.dtype,
        chunks=chunks,
        shards=shards,
        filters=filters,
        compressors=compressors,
        serializer=serializer,
        fill_value=fill_value,
        order=order,
        zarr_format=zarr_format,
        attributes=attributes,
        chunk_key_encoding=chunk_key_encoding,
        dimension_names=dimension_names,
    )


def skip_object_dtype(dtype: ZDType[Any, Any]) -> None:
    if dtype.dtype_cls is type(np.dtype("O")):
        msg = (
            f"{dtype} uses the numpy object data type, which is not a valid target for data "
            "type resolution"
        )
        pytest.skip(msg)


def nan_equal(a: object, b: object) -> bool:
    """
    Convenience function for equality comparison between two values ``a`` and ``b``, that might both
    be NaN. Returns True if both ``a`` and ``b`` are NaN, otherwise returns a == b
    """
    if math.isnan(a) and math.isnan(b):  # type: ignore[arg-type]
        return True
    return a == b


def deep_nan_equal(a: object, b: object) -> bool:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return all(deep_nan_equal(a[k], b[k]) for k in a)
    if isinstance(a, Sequence) and isinstance(b, Sequence):
        return all(deep_nan_equal(a[i], b[i]) for i in range(len(a)))
    return nan_equal(a, b)
