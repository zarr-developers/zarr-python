from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from zarr.core.config import (
    DEFAULT_CODECS,
    ZarrConfigManager,
    apply_overrides,
    build_config,
    collect_env,
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


def test_collect_env_parses_nested_and_literal() -> None:
    env = {
        "ZARR_ARRAY__ORDER": "F",
        "ZARR_ASYNC__CONCURRENCY": "32",
        "ZARR_CODECS__MY_CODEC": "my.module.MyCodec",
        "UNRELATED": "ignored",
    }
    out = collect_env(env)
    assert out["array.order"] == "F"
    assert out["async.concurrency"] == 32  # ast.literal_eval -> int
    assert out["codecs.my_codec"] == "my.module.MyCodec"  # non-literal -> raw str
    assert "unrelated" not in out


def test_apply_overrides_and_build_config_precedence() -> None:
    cfg = apply_overrides(
        build_config(environ={}),
        {"array.order": "F", "codecs.x": "pkg.X"},
    )
    assert cfg.array.order == "F"
    assert cfg.codecs["x"] == "pkg.X"
    # env overrides defaults
    cfg2 = build_config(environ={"ZARR_JSON_INDENT": "4"})
    assert cfg2.json_indent == 4


def test_collect_env_skips_zarr_config_meta_var() -> None:
    """ZARR_CONFIG is a directive about where config lives, not a config key itself."""
    env = {"ZARR_CONFIG": "/some/path.yaml", "ZARR_ARRAY__ORDER": "F"}
    out = collect_env(env)
    assert "config" not in out
    assert out["array.order"] == "F"


def test_build_config_zarr_config_env_does_not_raise() -> None:
    """Setting ZARR_CONFIG to a nonexistent path must not crash build_config."""
    cfg = build_config(environ={"ZARR_CONFIG": "/nonexistent/path.yaml"})
    # The nonexistent YAML path is simply skipped; defaults remain intact.
    from zarr.core.config import make_default_config

    assert cfg == make_default_config()


def test_proxy_attribute_and_string_access() -> None:
    cfg = ZarrConfigManager()
    assert cfg.array.order == "C"
    assert cfg.get("array.order") == "C"
    assert cfg.get("async.concurrency") == 10
    assert cfg.get("codecs", {})["blosc"] == "zarr.codecs.blosc.BloscCodec"
    assert cfg.get("does.not.exist", "fallback") == "fallback"


def test_set_permanent_and_context() -> None:
    cfg = ZarrConfigManager()
    cfg.set({"array.order": "F"})
    assert cfg.get("array.order") == "F"  # permanent
    with cfg.set({"array.order": "C"}):
        assert cfg.get("array.order") == "C"
    assert cfg.get("array.order") == "F"  # restored to permanent value
    cfg.reset()
    assert cfg.get("array.order") == "C"


def test_permanent_set_visible_in_worker_thread() -> None:
    cfg = ZarrConfigManager()
    cfg.set({"async.concurrency": 77})
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            seen = ex.submit(lambda: cfg.get("async.concurrency")).result()
        assert seen == 77  # ThreadPoolExecutor does not copy contextvars
    finally:
        cfg.reset()


def test_defaults_and_enable_gpu() -> None:
    cfg = ZarrConfigManager()
    assert cfg.defaults["array"]["order"] == "C"
    with cfg.set({"buffer": "x"}):
        pass
    cfg.enable_gpu()
    try:
        assert cfg.get("buffer") == "zarr.buffer.gpu.Buffer"
        assert cfg.get("ndbuffer") == "zarr.buffer.gpu.NDBuffer"
    finally:
        cfg.reset()


def test_refresh_not_shadowed_by_prior_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """refresh() must be visible in the calling context even after a prior set()/reset()."""
    mgr = ZarrConfigManager()
    # plant a scope entry in this thread/context (as reset()/set() would)
    mgr.set({"array.order": "F"})
    assert mgr.get("array.order") == "F"
    # change the environment so a rebuild differs, then refresh
    monkeypatch.setenv("ZARR_JSON_INDENT", "7")
    mgr.refresh()
    # refresh must be visible in THIS context, not shadowed by the prior scope
    assert mgr.get("json_indent") == 7
    assert mgr.get("array.order") == "C"  # the prior permanent set is gone after rebuild
