import os.path
import sys

import pytest

import zarr.codecs.registry
from zarr import config

here = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture()
def set_path():
    sys.path.append(here)
    zarr.codecs.registry._collect_entrypoints()
    yield
    sys.path.remove(here)
    lazy_load_codecs, lazy_load_pipelines = zarr.codecs.registry._collect_entrypoints()
    lazy_load_codecs.pop("test")
    lazy_load_pipelines.clear()
    config.reset()


@pytest.mark.usefixtures("set_path")
def test_entrypoint_codec():
    config.set({"codecs.test.name": "TestCodec"})
    cls = zarr.codecs.registry.get_codec_class("test")
    assert cls.__name__ == "TestCodec"


@pytest.mark.usefixtures("set_path")
def test_entrypoint_pipeline():
    config.set({"codec_pipeline.name": "TestCodecPipeline"})
    cls = zarr.codecs.registry.get_pipeline_class()
    assert cls.__name__ == "TestCodecPipeline"
