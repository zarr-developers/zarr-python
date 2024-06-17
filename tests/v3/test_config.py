import os

import pytest

from zarr.abc.codec import CodecPipeline
from zarr.codecs import BatchedCodecPipeline
from zarr.config import BadConfigError, config


def test_config_defaults_set():
    # regression test for available defaults
    assert config.defaults == [
        {
            "array": {"order": "C"},
            "async": {"concurrency": None, "timeout": None},
            "codec_pipeline": {"name": "batched_codec_pipeline", "batch_size": 1},
        }
    ]
    assert config.get("array.order") == "C"


def test_config_defaults_can_be_overridden():
    assert config.get("array.order") == "C"
    with config.set({"array.order": "F"}):
        assert config.get("array.order") == "F"


def test_config_codec_pipeline_class():
    # has default value
    assert config.codec_pipeline_class.__name__ != ""

    config.set({"codec_pipeline.name": "batched_codec_pipeline"})
    assert config.codec_pipeline_class == BatchedCodecPipeline

    class MockCodecPipeline(CodecPipeline):
        pass

    config.set({"codec_pipeline.name": "mock_codec_pipeline"})
    assert config.codec_pipeline_class == MockCodecPipeline

    with pytest.raises(BadConfigError):
        config.set({"codec_pipeline.name": "wrong_name"})
        config.codec_pipeline_class

    # Camel case works, too
    config.set({"codec_pipeline.name": "MockCodecPipeline"})
    assert config.codec_pipeline_class == MockCodecPipeline


def test_config_codec_pipeline_class_in_env():
    class MockEnvCodecPipeline(CodecPipeline):
        pass

    os.environ[("ZARR_PYTHON_CODEC_PIPELINE__NAME")] = "mock_env_codec_pipeline"
    config.refresh()
    assert config.codec_pipeline_class == MockEnvCodecPipeline
