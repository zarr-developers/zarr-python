from typing import Any

import pytest

from zarr.config import config


def test_config_defaults_set() -> None:
    # regression test for available defaults
    assert config.defaults == [
        {
            "array": {"order": "C"},
            "async": {"concurrency": None, "timeout": None},
            "codec_pipeline": {"batch_size": 1},
            "json_indent": 2,
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
