import os.path
import sys
import warnings

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
    entry_points = zarr.codecs.registry._collect_entrypoints()
    entry_points.pop("test")


@pytest.mark.usefixtures("set_path")
def test_entrypoint_codec():
    with pytest.raises(UserWarning):
        cls = zarr.codecs.registry.get_codec_class("test")
        assert cls.__name__ == "TestCodec"

    config.set({"codecs.test.name": "TestCodec"})
    cls = zarr.codecs.registry.get_codec_class("test")
    assert cls.__name__ == "TestCodec"