from zarr.config import config


def test_config_defaults_set():
    # regression test for available defaults
    assert config.defaults == [
        {"array": {"order": "C"}, "async": {"concurrency": None, "timeout": None}}
    ]
    assert config.get("array.order") == "C"


def test_config_defaults_can_be_overridden():
    assert config.get("array.order") == "C"
    with config.set({"array.order": "F"}):
        assert config.get("array.order") == "F"
