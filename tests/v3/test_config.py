import os
from collections.abc import Iterable
from unittest import mock
from unittest.mock import Mock

import pytest

from zarr import Array
from zarr.abc.codec import CodecInput, CodecOutput, CodecPipeline
from zarr.abc.store import ByteSetter
from zarr.array_spec import ArraySpec
from zarr.buffer import NDBuffer
from zarr.codecs import BatchedCodecPipeline, BloscCodec, BytesCodec
from zarr.codecs.registry import (
    get_codec_class,
    get_pipeline_class,
    register_codec,
    register_pipeline,
)
from zarr.config import BadConfigError, config
from zarr.indexing import SelectorTuple


@pytest.fixture()
def reset_config():
    config.reset()
    yield
    config.reset()


def test_config_defaults_set():
    # regression test for available defaults
    assert config.defaults == [
        {
            "array": {"order": "C"},
            "async": {"concurrency": None, "timeout": None},
            "codec_pipeline": {"name": "BatchedCodecPipeline", "batch_size": 1},
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


def test_config_defaults_can_be_overridden():
    assert config.get("array.order") == "C"
    with config.set({"array.order": "F"}):
        assert config.get("array.order") == "F"


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
