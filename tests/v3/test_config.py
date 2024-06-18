import os
from unittest import mock

import pytest

from zarr.abc.codec import CodecPipeline
from zarr.codecs import BatchedCodecPipeline, BloscCodec
from zarr.codecs.registry import (
    get_codec_class,
    get_pipeline_class,
    register_codec,
    register_pipeline,
)
from zarr.config import BadConfigError, config


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


def test_config_codec_pipeline_class():
    # has default value
    assert get_pipeline_class().__name__ != ""

    config.set({"codec_pipeline.name": "BatchedCodecPipeline"})
    assert get_pipeline_class() == BatchedCodecPipeline

    class MockCodecPipeline(CodecPipeline):
        pass

    register_pipeline(MockCodecPipeline)

    config.set({"codec_pipeline.name": "MockCodecPipeline"})
    assert get_pipeline_class() == MockCodecPipeline

    with pytest.raises(BadConfigError):
        config.set({"codec_pipeline.name": "wrong_name"})
        get_pipeline_class()

    class MockEnvCodecPipeline(CodecPipeline):
        pass

    register_pipeline(MockEnvCodecPipeline)

    with mock.patch.dict(os.environ, {"ZARR_PYTHON_CODEC_PIPELINE__NAME": "MockEnvCodecPipeline"}):
        assert get_pipeline_class(reload_config=True) == MockEnvCodecPipeline


def test_config_codec_implementation():
    assert get_codec_class("blosc").__name__ == config.defaults[0]["codecs"]["blosc"]["name"]

    class MockBloscCodec(BloscCodec):
        pass

    register_codec("blosc", MockBloscCodec)

    config.set({"codecs.blosc.name": "MockBloscCodec"})
    assert get_codec_class("blosc") == MockBloscCodec

    with mock.patch.dict(os.environ, {"ZARR_PYTHON_CODECS__BLOSC__NAME": "BloscCodec"}):
        assert get_codec_class("blosc", reload_config=True) == BloscCodec
