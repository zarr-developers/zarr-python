import os
from collections.abc import Iterable
from typing import Any
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pytest

import zarr
import zarr.api
from zarr import zeros
from zarr.abc.codec import CodecPipeline
from zarr.abc.store import ByteSetter, Store
from zarr.codecs import (
    BloscCodec,
    BytesCodec,
    Crc32cCodec,
    ShardingCodec,
)
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import NDBuffer
from zarr.core.buffer.core import Buffer
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.config import BadConfigError, config
from zarr.core.indexing import SelectorTuple
from zarr.registry import (
    fully_qualified_name,
    get_buffer_class,
    get_codec_class,
    get_ndbuffer_class,
    get_pipeline_class,
    register_buffer,
    register_codec,
    register_ndbuffer,
    register_pipeline,
)
from zarr.testing.buffer import (
    NDBufferUsingTestNDArrayLike,
    StoreExpectingTestBuffer,
    TestBuffer,
    TestNDArrayLike,
)


def test_config_defaults_set() -> None:
    # regression test for available defaults
    assert (
        config.defaults
        == [
            {
                "default_zarr_format": 3,
                "array": {
                    "order": "C",
                    "write_empty_chunks": False,
                },
                "async": {"concurrency": 10, "timeout": None},
                "threading": {"max_workers": None},
                "json_indent": 2,
                "codec_pipeline": {
                    "path": "zarr.core.codec_pipeline.BatchedCodecPipeline",
                    "batch_size": 1,
                },
                "codecs": {
                    "blosc": "zarr.codecs.blosc.BloscCodec",
                    "gzip": "zarr.codecs.gzip.GzipCodec",
                    "zstd": "zarr.codecs.zstd.ZstdCodec",
                    "bytes": "zarr.codecs.bytes.BytesCodec",
                    "endian": "zarr.codecs.bytes.BytesCodec",  # compatibility with earlier versions of ZEP1
                    "crc32c": "zarr.codecs.crc32c_.Crc32cCodec",
                    "sharding_indexed": "zarr.codecs.sharding.ShardingCodec",
                    "transpose": "zarr.codecs.transpose.TransposeCodec",
                    "vlen-utf8": "zarr.codecs.vlen_utf8.VLenUTF8Codec",
                    "vlen-bytes": "zarr.codecs.vlen_utf8.VLenBytesCodec",
                },
                "buffer": "zarr.buffer.cpu.Buffer",
                "ndbuffer": "zarr.buffer.cpu.NDBuffer",
            }
        ]
    )
    assert config.get("array.order") == "C"
    assert config.get("async.concurrency") == 10
    assert config.get("async.timeout") is None
    assert config.get("codec_pipeline.batch_size") == 1
    assert config.get("json_indent") == 2


@pytest.mark.parametrize(
    ("key", "old_val", "new_val"),
    [("array.order", "C", "F"), ("async.concurrency", 10, 20), ("json_indent", 2, 0)],
)
def test_config_defaults_can_be_overridden(key: str, old_val: Any, new_val: Any) -> None:
    assert config.get(key) == old_val
    with config.set({key: new_val}):
        assert config.get(key) == new_val


def test_fully_qualified_name() -> None:
    class MockClass:
        pass

    assert (
        fully_qualified_name(MockClass)
        == "tests.test_config.test_fully_qualified_name.<locals>.MockClass"
    )


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_config_codec_pipeline_class(store: Store) -> None:
    # has default value
    assert get_pipeline_class().__name__ != ""

    config.set({"codec_pipeline.name": "zarr.core.codec_pipeline.BatchedCodecPipeline"})
    assert get_pipeline_class() == zarr.core.codec_pipeline.BatchedCodecPipeline

    _mock = Mock()

    class MockCodecPipeline(BatchedCodecPipeline):
        async def write(
            self,
            batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
            value: NDBuffer,
            drop_axes: tuple[int, ...] = (),
        ) -> None:
            _mock.call()

    register_pipeline(MockCodecPipeline)
    config.set({"codec_pipeline.path": fully_qualified_name(MockCodecPipeline)})

    assert get_pipeline_class() == MockCodecPipeline

    # test if codec is used
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        chunks=(10,),
        zarr_format=3,
        dtype="i4",
    )
    arr[:] = range(100)

    _mock.call.assert_called()

    config.set({"codec_pipeline.path": "wrong_name"})
    with pytest.raises(BadConfigError):
        get_pipeline_class()

    class MockEnvCodecPipeline(CodecPipeline):
        pass

    register_pipeline(MockEnvCodecPipeline)  # type: ignore[type-abstract]

    with mock.patch.dict(
        os.environ, {"ZARR_CODEC_PIPELINE__PATH": fully_qualified_name(MockEnvCodecPipeline)}
    ):
        assert get_pipeline_class(reload_config=True) == MockEnvCodecPipeline


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_config_codec_implementation(store: Store) -> None:
    # has default value
    assert fully_qualified_name(get_codec_class("blosc")) == config.defaults[0]["codecs"]["blosc"]

    _mock = Mock()

    class MockBloscCodec(BloscCodec):
        async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
            _mock.call()
            return None

    register_codec("blosc", MockBloscCodec)
    with config.set({"codecs.blosc": fully_qualified_name(MockBloscCodec)}):
        assert get_codec_class("blosc") == MockBloscCodec

        # test if codec is used
        arr = zarr.create_array(
            store=store,
            shape=(100,),
            chunks=(10,),
            zarr_format=3,
            dtype="i4",
            compressors=[{"name": "blosc", "configuration": {}}],
        )
        arr[:] = range(100)
        _mock.call.assert_called()

    # test set codec with environment variable
    class NewBloscCodec(BloscCodec):
        pass

    register_codec("blosc", NewBloscCodec)
    with mock.patch.dict(os.environ, {"ZARR_CODECS__BLOSC": fully_qualified_name(NewBloscCodec)}):
        assert get_codec_class("blosc", reload_config=True) == NewBloscCodec


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_config_ndbuffer_implementation(store: Store) -> None:
    # set custom ndbuffer with TestNDArrayLike implementation
    register_ndbuffer(NDBufferUsingTestNDArrayLike)
    with config.set({"ndbuffer": fully_qualified_name(NDBufferUsingTestNDArrayLike)}):
        assert get_ndbuffer_class() == NDBufferUsingTestNDArrayLike
        arr = zarr.create_array(
            store=store,
            shape=(100,),
            chunks=(10,),
            zarr_format=3,
            dtype="i4",
        )
        got = arr[:]
        assert isinstance(got, TestNDArrayLike)


def test_config_buffer_implementation() -> None:
    # has default value
    assert config.defaults[0]["buffer"] == "zarr.buffer.cpu.Buffer"

    arr = zeros(shape=(100,), store=StoreExpectingTestBuffer())

    # AssertionError of StoreExpectingTestBuffer when not using my buffer
    with pytest.raises(AssertionError):
        arr[:] = np.arange(100)

    register_buffer(TestBuffer)
    with config.set({"buffer": fully_qualified_name(TestBuffer)}):
        assert get_buffer_class() == TestBuffer

        # no error using TestBuffer
        data = np.arange(100)
        arr[:] = np.arange(100)
        assert np.array_equal(arr[:], data)

        data2d = np.arange(1000).reshape(100, 10)
        arr_sharding = zeros(
            shape=(100, 10),
            store=StoreExpectingTestBuffer(),
            codecs=[ShardingCodec(chunk_shape=(10, 10))],
        )
        arr_sharding[:] = data2d
        assert np.array_equal(arr_sharding[:], data2d)

        arr_Crc32c = zeros(
            shape=(100, 10),
            store=StoreExpectingTestBuffer(),
            codecs=[BytesCodec(), Crc32cCodec()],
        )
        arr_Crc32c[:] = data2d
        assert np.array_equal(arr_Crc32c[:], data2d)


def test_config_buffer_backwards_compatibility() -> None:
    # This should warn once zarr.core is private
    # https://github.com/zarr-developers/zarr-python/issues/2621
    with zarr.config.set(
        {"buffer": "zarr.core.buffer.cpu.Buffer", "ndbuffer": "zarr.core.buffer.cpu.NDBuffer"}
    ):
        get_buffer_class()
        get_ndbuffer_class()


@pytest.mark.gpu
def test_config_buffer_backwards_compatibility_gpu() -> None:
    # This should warn once zarr.core is private
    # https://github.com/zarr-developers/zarr-python/issues/2621
    with zarr.config.set(
        {"buffer": "zarr.core.buffer.gpu.Buffer", "ndbuffer": "zarr.core.buffer.gpu.NDBuffer"}
    ):
        get_buffer_class()
        get_ndbuffer_class()


@pytest.mark.filterwarnings("error")
def test_warning_on_missing_codec_config() -> None:
    class NewCodec(BytesCodec):
        pass

    class NewCodec2(BytesCodec):
        pass

    # error if codec is not registered
    with pytest.raises(KeyError):
        get_codec_class("missing_codec")

    # no warning if only one implementation is available
    register_codec("new_codec", NewCodec)
    get_codec_class("new_codec")

    # warning because multiple implementations are available but none is selected in the config
    register_codec("new_codec", NewCodec2)
    with pytest.warns(UserWarning, match="not configured in config. Selecting any implementation"):
        get_codec_class("new_codec")

    # no warning if multiple implementations are available and one is selected in the config
    with config.set({"codecs.new_codec": fully_qualified_name(NewCodec)}):
        get_codec_class("new_codec")


@pytest.mark.parametrize(
    "key",
    [
        "array.v2_default_compressor.numeric",
        "array.v2_default_compressor.string",
        "array.v2_default_compressor.bytes",
        "array.v2_default_filters.string",
        "array.v2_default_filters.bytes",
        "array.v3_default_filters.numeric",
        "array.v3_default_filters.raw",
        "array.v3_default_filters.bytes",
        "array.v3_default_serializer.numeric",
        "array.v3_default_serializer.string",
        "array.v3_default_serializer.bytes",
        "array.v3_default_compressors.string",
        "array.v3_default_compressors.bytes",
        "array.v3_default_compressors",
    ],
)
def test_deprecated_config(key: str) -> None:
    """
    Test that a valuerror is raised when setting the default chunk encoding for a given
    data type category
    """

    with pytest.raises(ValueError):
        with zarr.config.set({key: "foo"}):
            pass
