from __future__ import annotations

import pathlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import HealthCheck, Verbosity, settings

from zarr import AsyncGroup, config
from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.abc.store import Store
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.sharding import ShardingCodec
from zarr.core.chunk_grids import _guess_chunks
from zarr.core.chunk_key_encodings import ChunkKeyEncoding
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.storage import LocalStore, MemoryStore, StorePath, ZipStore
from zarr.storage.remote import RemoteStore

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, Literal

    from _pytest.compat import LEGACY_PATH

    from zarr.core.common import ChunkCoords, MemoryOrder, ZarrFormat


async def parse_store(
    store: Literal["local", "memory", "remote", "zip"], path: str
) -> LocalStore | MemoryStore | RemoteStore | ZipStore:
    if store == "local":
        return await LocalStore.open(path)
    if store == "memory":
        return await MemoryStore.open()
    if store == "remote":
        return await RemoteStore.open(url=path)
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
async def remote_store(url: str) -> RemoteStore:
    return await RemoteStore.open(url)


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


@pytest.fixture(params=["local", "memory", "zip"])
def sync_store(request: pytest.FixtureRequest, tmp_path: LEGACY_PATH) -> Store:
    result = sync(parse_store(request.param, str(tmp_path)))
    if not isinstance(result, Store):
        raise TypeError("Wrong store class returned by test fixture! got " + result + " instead")
    return result


@dataclass
class AsyncGroupRequest:
    zarr_format: ZarrFormat
    store: Literal["local", "remote", "memory", "zip"]
    attributes: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
async def async_group(request: pytest.FixtureRequest, tmpdir: LEGACY_PATH) -> AsyncGroup:
    param: AsyncGroupRequest = request.param

    store = await parse_store(param.store, str(tmpdir))
    return await AsyncGroup.from_store(
        store,
        attributes=param.attributes,
        zarr_format=param.zarr_format,
        exists_ok=False,
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


settings.register_profile(
    "ci",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)
settings.register_profile(
    "local",
    max_examples=300,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    verbosity=Verbosity.verbose,
)
import numcodecs


def meta_from_array_v2(
    array: np.ndarray[Any, Any],
    chunks: ChunkCoords | Literal["auto"] = "auto",
    compressor: numcodecs.abc.Codec | Literal["auto"] | None = "auto",
    filters: Iterable[numcodecs.abc.Codec] | Literal["auto"] = "auto",
    fill_value: Any = "auto",
    order: MemoryOrder | Literal["auto"] = "auto",
    dimension_separator: Literal[".", "/", "auto"] = "auto",
    attributes: dict[str, Any] | None = None,
) -> ArrayV2Metadata:
    """
    Create a v2 metadata object from a numpy array
    """

    _chunks = auto_chunks(chunks, array.shape, array.dtype)
    _compressor = auto_compressor(compressor)
    _filters = auto_filters(filters)
    _fill_value = auto_fill_value(fill_value)
    _order = auto_order(order)
    _dimension_separator = auto_dimension_separator(dimension_separator)
    return ArrayV2Metadata(
        shape=array.shape,
        dtype=array.dtype,
        chunks=_chunks,
        compressor=_compressor,
        filters=_filters,
        fill_value=_fill_value,
        order=_order,
        dimension_separator=_dimension_separator,
        attributes=attributes,
    )


from typing import TypedDict


class ChunkEncoding(TypedDict):
    filters: tuple[ArrayArrayCodec]
    compressors: tuple[BytesBytesCodec]
    serializer: ArrayBytesCodec


class ChunkingSpec(TypedDict):
    shard_shape: tuple[int, ...]
    chunk_shape: tuple[int, ...] | None
    chunk_key_encoding: ChunkKeyEncoding


def meta_from_array_v3(
    array: np.ndarray[Any, Any],
    shard_shape: tuple[int, ...] | Literal["auto"] | None,
    chunk_shape: tuple[int, ...] | Literal["auto"],
    serializer: ArrayBytesCodec | Literal["auto"] = "auto",
    compressors: Iterable[BytesBytesCodec] | Literal["auto"] = "auto",
    filters: Iterable[ArrayArrayCodec] | Literal["auto"] = "auto",
    fill_value: Any = "auto",
    chunk_key_encoding: ChunkKeyEncoding | Literal["auto"] = "auto",
    dimension_names: Iterable[str] | None = None,
    attributes: dict[str, Any] | None = None,
) -> ArrayV3Metadata:
    _write_chunks, _read_chunks = auto_chunks_v3(
        shard_shape=shard_shape, chunk_shape=chunk_shape, array_shape=array.shape, dtype=array.dtype
    )
    _codecs = auto_codecs(serializer=serializer, compressors=compressors, filters=filters)
    if _read_chunks is not None:
        _codecs = (ShardingCodec(codecs=_codecs, chunk_shape=_read_chunks),)

    _fill_value = auto_fill_value(fill_value)
    _chunk_key_encoding = auto_chunk_key_encoding(chunk_key_encoding)
    return ArrayV3Metadata(
        shape=array.shape,
        dtype=array.dtype,
        codecs=_codecs,
        chunk_key_encoding=_chunk_key_encoding,
        fill_value=fill_value,
        chunk_grid={"name": "regular", "config": {"chunk_shape": shard_shape}},
        attributes=attributes,
        dimension_names=dimension_names,
    )


from zarr.abc.codec import Codec
from zarr.codecs import ZstdCodec


def auto_codecs(
    *,
    filters: Iterable[ArrayArrayCodec] | Literal["auto"] = "auto",
    compressors: Iterable[BytesBytesCodec] | Literal["auto"] = "auto",
    serializer: ArrayBytesCodec | Literal["auto"] = "auto",
) -> tuple[Codec, ...]:
    """
    Heuristically generate a tuple of codecs
    """
    _compressors: tuple[BytesBytesCodec, ...]
    _filters: tuple[ArrayArrayCodec, ...]
    _serializer: ArrayBytesCodec
    if filters == "auto":
        _filters = ()
    else:
        _filters = tuple(filters)

    if compressors == "auto":
        _compressors = (ZstdCodec(level=3),)
    else:
        _compressors = tuple(compressors)

    if serializer == "auto":
        _serializer = BytesCodec()
    else:
        _serializer = serializer
    return (*_filters, _serializer, *_compressors)


def auto_dimension_separator(dimension_separator: Literal[".", "/", "auto"]) -> Literal[".", "/"]:
    if dimension_separator == "auto":
        return "/"
    return dimension_separator


def auto_order(order: MemoryOrder | Literal["auto"]) -> MemoryOrder:
    if order == "auto":
        return "C"
    return order


def auto_fill_value(fill_value: Any) -> Any:
    if fill_value == "auto":
        return 0
    return fill_value


def auto_compressor(
    compressor: numcodecs.abc.Codec | Literal["auto"] | None,
) -> numcodecs.abc.Codec | None:
    if compressor == "auto":
        return numcodecs.Zstd(level=3)
    return compressor


def auto_filters(
    filters: Iterable[numcodecs.abc.Codec] | Literal["auto"],
) -> tuple[numcodecs.abc.Codec, ...]:
    if filters == "auto":
        return ()
    return tuple(filters)


def auto_chunks(
    chunks: tuple[int, ...] | Literal["auto"], shape: tuple[int, ...], dtype: npt.DTypeLike
) -> tuple[int, ...]:
    if chunks == "auto":
        return _guess_chunks(shape, np.dtype(dtype).itemsize)
    return chunks


def auto_chunks_v3(
    *,
    shard_shape: tuple[int, ...] | Literal["auto"],
    chunk_shape: tuple[int, ...] | Literal["auto"] | None,
    array_shape: tuple[int, ...],
    dtype: npt.DTypeLike,
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    match (shard_shape, chunk_shape):
        case ("auto", "auto"):
            # stupid default but easy to think about
            return ((256,) * len(array_shape), (64,) * len(array_shape))
        case ("auto", None):
            return (_guess_chunks(array_shape, np.dtype(dtype).itemsize), None)
        case ("auto", _):
            return (chunk_shape, chunk_shape)
        case (_, None):
            return (shard_shape, None)
        case (_, "auto"):
            return (shard_shape, shard_shape)
        case _:
            return (shard_shape, chunk_shape)
