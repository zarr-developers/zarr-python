import pytest

import zarr.registry
from zarr import config


@pytest.mark.usefixtures("set_path")
@pytest.mark.parametrize("codec_name", ["TestEntrypointCodec", "TestEntrypointGroup.Codec"])
def test_entrypoint_codec(codec_name: str) -> None:
    config.set({"codecs.test": "package_with_entrypoint." + codec_name})
    cls_test = zarr.registry.get_codec_class("test")
    assert cls_test.__qualname__ == codec_name


@pytest.mark.usefixtures("set_path")
def test_entrypoint_pipeline() -> None:
    config.set({"codec_pipeline.path": "package_with_entrypoint.TestEntrypointCodecPipeline"})
    cls = zarr.registry.get_pipeline_class()
    assert cls.__name__ == "TestEntrypointCodecPipeline"


@pytest.mark.usefixtures("set_path")
@pytest.mark.parametrize("buffer_name", ["TestEntrypointBuffer", "TestEntrypointGroup.Buffer"])
def test_entrypoint_buffer(buffer_name: str) -> None:
    config.set(
        {
            "buffer": "package_with_entrypoint." + buffer_name,
            "ndbuffer": "package_with_entrypoint.TestEntrypointNDBuffer",
        }
    )
    assert zarr.registry.get_buffer_class().__qualname__ == buffer_name
    assert zarr.registry.get_ndbuffer_class().__name__ == "TestEntrypointNDBuffer"
