import os.path
import sys

import pytest

import zarr.registry
from zarr import config

here = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture()
def set_path():
    sys.path.append(here)
    zarr.registry._collect_entrypoints()
    yield
    sys.path.remove(here)
    lazy_load_lists = zarr.registry._collect_entrypoints()
    for lazy_load_list in lazy_load_lists:
        lazy_load_list.clear()
    config.reset()


@pytest.mark.usefixtures("set_path")
def test_entrypoint_codec():
    config.set({"codecs.test.name": "TestCodec"})
    cls = zarr.registry.get_codec_class("test")
    assert cls.__name__ == "TestCodec"


@pytest.mark.usefixtures("set_path")
def test_entrypoint_pipeline():
    config.set({"codec_pipeline.name": "TestCodecPipeline"})
    cls = zarr.registry.get_pipeline_class()
    assert cls.__name__ == "TestCodecPipeline"


@pytest.mark.usefixtures("set_path")
def test_entrypoint_buffer():
    config.set({"buffer.name": "TestBuffer", "ndbuffer.name": "TestNDBuffer"})
    assert zarr.registry.get_buffer_class().__name__ == "TestBuffer"
    assert zarr.registry.get_ndbuffer_class().__name__ == "TestNDBuffer"
