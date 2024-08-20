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
    registries = zarr.registry._collect_entrypoints()
    for registry in registries:
        registry.lazy_load_list.clear()
    config.reset()


@pytest.mark.usefixtures("set_path")
@pytest.mark.parametrize("codec_name", ["TestEntrypointCodec", "TestEntrypointGroup.Codec"])
def test_entrypoint_codec(codec_name):
    config.set({"codecs.test": "package_with_entrypoint." + codec_name})
    cls_test = zarr.registry.get_codec_class("test")
    assert cls_test.__qualname__ == codec_name


@pytest.mark.usefixtures("set_path")
def test_entrypoint_pipeline():
    config.set({"codec_pipeline.path": "package_with_entrypoint.TestEntrypointCodecPipeline"})
    cls = zarr.registry.get_pipeline_class()
    assert cls.__name__ == "TestEntrypointCodecPipeline"


@pytest.mark.usefixtures("set_path")
@pytest.mark.parametrize("buffer_name", ["TestEntrypointBuffer", "TestEntrypointGroup.Buffer"])
def test_entrypoint_buffer(buffer_name):
    config.set(
        {
            "buffer": "package_with_entrypoint." + buffer_name,
            "ndbuffer": "package_with_entrypoint.TestEntrypointNDBuffer",
        }
    )
    assert zarr.registry.get_buffer_class().__qualname__ == buffer_name
    assert zarr.registry.get_ndbuffer_class().__name__ == "TestEntrypointNDBuffer"
