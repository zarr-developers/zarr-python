from __future__ import annotations

import pytest

from zarr.core.config import (
    DEFAULT_CODECS,
    get_path,
    make_default_config,
    replace_path,
    to_nested_dict,
)


def test_default_config_values() -> None:
    cfg = make_default_config()
    assert cfg.default_zarr_format == 3
    assert cfg.array.order == "C"
    assert cfg.array.sharding_coalesce_max_bytes == 16 << 20
    assert cfg.async_.concurrency == 10
    assert cfg.async_.timeout is None
    assert cfg.threading.max_workers is None
    assert cfg.json_indent == 2
    assert cfg.codec_pipeline.path == "zarr.core.codec_pipeline.BatchedCodecPipeline"
    assert cfg.codecs["blosc"] == "zarr.codecs.blosc.BloscCodec"
    assert cfg.codecs == DEFAULT_CODECS


def test_get_path_structured_and_async_alias() -> None:
    cfg = make_default_config()
    assert get_path(cfg, "array.order") == "C"
    assert get_path(cfg, "async.concurrency") == 10  # serialized key, not async_
    assert get_path(cfg, "json_indent") == 2
    assert get_path(cfg, "codecs") == DEFAULT_CODECS
    assert get_path(cfg, "codecs.blosc") == "zarr.codecs.blosc.BloscCodec"
    with pytest.raises(KeyError):
        get_path(cfg, "array.nonexistent")


def test_replace_path_is_immutable_and_typed() -> None:
    cfg = make_default_config()
    cfg2 = replace_path(cfg, "array.order", "F")
    assert cfg.array.order == "C"  # original unchanged (frozen)
    assert cfg2.array.order == "F"
    cfg3 = replace_path(cfg, "async.concurrency", 99)
    assert cfg3.async_.concurrency == 99
    cfg4 = replace_path(cfg, "codecs.my_codec", "my.module.MyCodec")
    assert cfg4.codecs["my_codec"] == "my.module.MyCodec"
    assert "my_codec" not in cfg.codecs


def test_to_nested_dict_uses_serialized_keys() -> None:
    nested = to_nested_dict(make_default_config())
    assert nested["array"]["order"] == "C"
    assert nested["async"]["concurrency"] == 10  # serialized key
    assert "async_" not in nested
    assert nested["codecs"]["blosc"] == "zarr.codecs.blosc.BloscCodec"
