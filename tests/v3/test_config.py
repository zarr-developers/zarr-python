import os
from collections.abc import Iterable
from typing import Any
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pytest

from zarr import Array, zeros
from zarr.abc.codec import CodecInput, CodecOutput, CodecPipeline
from zarr.abc.store import ByteSetter
from zarr.array_spec import ArraySpec
from zarr.buffer import NDBuffer
from zarr.codecs import BatchedCodecPipeline, BloscCodec, BytesCodec, Crc32cCodec, ShardingCodec
from zarr.config import BadConfigError, config
from zarr.indexing import SelectorTuple
from zarr.registry import (
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
    assert config.defaults == [
        {
            "array": {"order": "C"},
            "async": {"concurrency": None, "timeout": None},
            "json_indent": 2,
            "codec_pipeline": {"name": "BatchedCodecPipeline", "batch_size": 1},
            "buffer": {"name": "Buffer"},
            "ndbuffer": {"name": "NDBuffer"},
            "codecs": {
                "blosc": {"name": "BloscCodec"},
                "gzip": {"name": "GzipCodec"},
                "zstd": {"name": "ZstdCodec"},
                "bytes": {"name": "BytesCodec"},
                "endian": {"name": "BytesCodec"},  # compatibility with earlier versions of ZEP1
                "crc32c": {"name": "Crc32cCodec"},
                "sharding_indexed": {"name": "ShardingCodec"},
                "transpose": {"name": "TransposeCodec"},
            },
        }
    ]
    assert config.get("array.order") == "C"
    assert config.get("async.concurrency") is None
    assert config.get("async.timeout") is None
    assert config.get("codec_pipeline.batch_size") == 1
    assert config.get("json_indent") == 2


@pytest.mark.parametrize(
    "key, old_val, new_val",
    [("array.order", "C", "F"), ("async.concurrency", None, 10), ("json_indent", 2, 0)],
)
def test_config_defaults_can_be_overridden(key: str, old_val: Any, new_val: Any) -> None:
    assert config.get(key) == old_val
    with config.set({key: new_val}):
        assert config.get(key) == new_val


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
def test_config_codec_pipeline_class(store):
    # has default value
    assert get_pipeline_class().__name__ != ""

    config.set({"codec_pipeline.name": "BatchedCodecPipeline"})
    assert get_pipeline_class() == BatchedCodecPipeline

    _mock = Mock()

    class MockCodecPipeline(BatchedCodecPipeline):
        async def write(
            self,
            batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]],
            value: NDBuffer,
            drop_axes: tuple[int, ...] = (),
        ) -> None:
            _mock.call()

    register_pipeline(MockCodecPipeline)
    config.set({"codec_pipeline.name": "MockCodecPipeline"})
    assert get_pipeline_class() == MockCodecPipeline

    # test if codec is used
    arr = Array.create(
        store=store,
        shape=(100,),
        chunks=(10,),
        zarr_format=3,
        dtype="i4",
    )
    arr[:] = range(100)

    _mock.call.assert_called()

    with pytest.raises(BadConfigError):
        config.set({"codec_pipeline.name": "wrong_name"})
        get_pipeline_class()

    class MockEnvCodecPipeline(CodecPipeline):
        pass

    register_pipeline(MockEnvCodecPipeline)

    with mock.patch.dict(os.environ, {"ZARR_PYTHON_CODEC_PIPELINE__NAME": "MockEnvCodecPipeline"}):
        assert get_pipeline_class(reload_config=True) == MockEnvCodecPipeline


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
def test_config_codec_implementation(store):
    # has default value
    assert get_codec_class("blosc").__name__ == config.defaults[0]["codecs"]["blosc"]["name"]

    _mock = Mock()

    class MockBloscCodec(BloscCodec):
        async def _encode_single(
            self, chunk_data: CodecInput, chunk_spec: ArraySpec
        ) -> CodecOutput | None:
            _mock.call()

    config.set({"codecs.blosc.name": "MockBloscCodec"})
    register_codec("blosc", MockBloscCodec)
    assert get_codec_class("blosc") == MockBloscCodec

    # test if codec is used
    arr = Array.create(
        store=store,
        shape=(100,),
        chunks=(10,),
        zarr_format=3,
        dtype="i4",
        codecs=[BytesCodec(), {"name": "blosc", "configuration": {}}],
    )
    arr[:] = range(100)
    _mock.call.assert_called()

    with mock.patch.dict(os.environ, {"ZARR_PYTHON_CODECS__BLOSC__NAME": "BloscCodec"}):
        assert get_codec_class("blosc", reload_config=True) == BloscCodec


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
def test_config_ndbuffer_implementation(store):
    # has default value
    assert get_ndbuffer_class().__name__ == config.defaults[0]["ndbuffer"]["name"]

    # set custom ndbuffer with MyNDArrayLike implementation
    register_ndbuffer(NDBufferUsingTestNDArrayLike)
    config.set({"ndbuffer.name": "NDBufferUsingTestNDArrayLike"})
    assert get_ndbuffer_class() == NDBufferUsingTestNDArrayLike
    arr = Array.create(
        store=store,
        shape=(100,),
        chunks=(10,),
        zarr_format=3,
        dtype="i4",
    )
    got = arr[:]
    print(type(got))
    assert isinstance(got, TestNDArrayLike)


def test_config_buffer_implementation():
    # has default value
    assert get_buffer_class().__name__ == config.defaults[0]["buffer"]["name"]

    arr = zeros(shape=(100), store=StoreExpectingTestBuffer(mode="w"))

    # AssertionError of StoreExpectingMyBuffer when not using my buffer
    with pytest.raises(AssertionError):
        arr[:] = np.arange(100)

    register_buffer(TestBuffer)
    config.set({"buffer.name": "TestBuffer"})
    assert get_buffer_class() == TestBuffer

    # no error using MyBuffer
    data = np.arange(100)
    arr[:] = np.arange(100)
    assert np.array_equal(arr[:], data)

    data2d = np.arange(1000).reshape(100, 10)
    arr_sharding = zeros(
        shape=(100, 10),
        store=StoreExpectingTestBuffer(mode="w"),
        codecs=[ShardingCodec(chunk_shape=(10, 10))],
    )
    arr_sharding[:] = data2d
    assert np.array_equal(arr_sharding[:], data2d)

    arr_Crc32c = zeros(
        shape=(100, 10),
        store=StoreExpectingTestBuffer(mode="w"),
        codecs=[BytesCodec(), Crc32cCodec()],
    )
    arr_Crc32c[:] = data2d
    assert np.array_equal(arr_Crc32c[:], data2d)
