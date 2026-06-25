from __future__ import annotations

import dataclasses
import typing
from concurrent.futures import ThreadPoolExecutor

import pytest

from zarr.core.config import (
    _SERIALIZED_NAMES,
    DEFAULT_CODECS,
    BadConfigError,
    ZarrConfig,
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


# ---------------------------------------------------------------------------
# Removed-deprecated-key behavior (donfig-faithful)
# ---------------------------------------------------------------------------

_REMOVED_KEY = "array.v2_default_compressor.numeric"


def test_get_removed_deprecated_key_with_default() -> None:
    """get() with a removed deprecated key and a default must return the default silently."""
    mgr = ZarrConfigManager()
    result = mgr.get(_REMOVED_KEY, "fallback")
    assert result == "fallback"


def test_get_removed_deprecated_key_no_default_raises_key_error() -> None:
    """get() with a removed deprecated key and no default must raise KeyError, not BadConfigError."""
    mgr = ZarrConfigManager()
    with pytest.raises(KeyError):
        mgr.get(_REMOVED_KEY)


def test_set_removed_deprecated_key_raises_bad_config_error() -> None:
    """set() with a removed deprecated key must still raise BadConfigError."""
    mgr = ZarrConfigManager()
    with pytest.raises(BadConfigError):
        mgr.set({_REMOVED_KEY: "some_value"})


# ---------------------------------------------------------------------------
# Tolerant ingest: unknown env/YAML keys must warn and be skipped, not crash
# ---------------------------------------------------------------------------


def test_build_config_unknown_env_key_warns_and_skips() -> None:
    """build_config with an unrecognized env var warns and skips it; known keys still apply."""
    with pytest.warns(UserWarning, match="future.key"):
        cfg = build_config(environ={"ZARR_FUTURE__KEY": "1", "ZARR_ARRAY__ORDER": "F"})
    # Known key was applied
    assert cfg.array.order == "F"
    # All other fields are still at default
    default = make_default_config()
    from dataclasses import fields as dc_fields

    for f in dc_fields(default):
        if f.name != "array":
            assert getattr(cfg, f.name) == getattr(default, f.name)


def test_apply_overrides_unknown_key_warns_and_returns_default() -> None:
    """apply_overrides with a totally unknown key warns and returns an otherwise-default config."""
    default = make_default_config()
    with pytest.warns(UserWarning, match="totally.bogus.key"):
        result = apply_overrides(default, {"totally.bogus.key": 123})
    assert result == default


def test_config_set_still_strict_for_unknown_keys() -> None:
    """config.set() must remain strict: unknown structured keys raise KeyError."""
    with pytest.raises(KeyError):
        ZarrConfigManager().set({"totally.bogus.key": 1})


def test_donfig_not_imported() -> None:
    import sys

    import zarr  # noqa: F401

    assert "donfig" not in sys.modules


# ---------------------------------------------------------------------------
# Drift-protection: every structured leaf key must have a get() overload
# ---------------------------------------------------------------------------


def _structured_leaf_keys(cfg_cls: type, prefix: str = "") -> list[str]:
    """Walk a settings dataclass recursively and return every dotted leaf key.

    Uses ``typing.get_type_hints`` instead of ``f.type`` so that the
    ``from __future__ import annotations`` string-annotation form is resolved
    to real types before ``dataclasses.is_dataclass`` is called.
    """
    keys: list[str] = []
    resolved_hints = typing.get_type_hints(cfg_cls)
    for f in dataclasses.fields(cfg_cls):
        serialized = _SERIALIZED_NAMES.get(f.name, f.name)
        key = f"{prefix}.{serialized}" if prefix else serialized
        resolved_type = resolved_hints[f.name]
        if dataclasses.is_dataclass(resolved_type):
            keys.extend(_structured_leaf_keys(typing.cast(type, resolved_type), key))
        elif f.name == "codecs":
            # open mapping — intentionally not enumerated
            continue
        else:
            keys.append(key)
    return keys


def test_every_structured_key_has_a_get_overload() -> None:
    """Enumerate every typed leaf key in ZarrConfig and assert a matching get() overload exists."""
    overloads = typing.get_overloads(ZarrConfigManager.get)
    literal_keys: set[str] = set()
    for ov in overloads:
        hints = typing.get_type_hints(ov)
        key_hint = hints.get("key")
        if typing.get_origin(key_hint) is typing.Literal:
            literal_keys.update(typing.get_args(key_hint))
    leaf_keys = _structured_leaf_keys(ZarrConfig)
    missing = set(leaf_keys) - literal_keys
    assert not missing, f"get() overloads missing for: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Static-typing smoke test (only checked by mypy, not executed at runtime)
# ---------------------------------------------------------------------------

if typing.TYPE_CHECKING:

    def _typing_smoke(cfg: ZarrConfigManager) -> None:
        typing.assert_type(cfg.get("array.order"), typing.Literal["C", "F"])
        typing.assert_type(cfg.array.order, typing.Literal["C", "F"])
        typing.assert_type(cfg.get("async.concurrency"), int)
