from __future__ import annotations

import dataclasses
import typing
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.conftest import Expect, ExpectFail
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

if typing.TYPE_CHECKING:
    import pathlib

# ---------------------------------------------------------------------------
# Module-level constants used in parametrize lists (evaluated at collection time)
# ---------------------------------------------------------------------------

_REMOVED_KEY = "array.v2_default_compressor.numeric"
_DEFAULT = make_default_config()

# ---------------------------------------------------------------------------
# 1. get_path — success cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(input="array.order", output="C", id="array-order"),
        Expect(input="async.concurrency", output=10, id="async-concurrency-alias"),
        Expect(input="json_indent", output=2, id="json-indent"),
        Expect(input="codecs", output=DEFAULT_CODECS, id="codecs-dict"),
        Expect(input="codecs.blosc", output="zarr.codecs.blosc.BloscCodec", id="codecs-blosc"),
    ],
    ids=lambda c: c.id,
)
def test_get_path(case: Expect[str, object]) -> None:
    assert get_path(make_default_config(), case.input) == case.output


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(input="array.nonexistent", exception=KeyError, id="nonexistent-key"),
    ],
    ids=lambda c: c.id,
)
def test_get_path_raises(case: ExpectFail[str]) -> None:
    with case.raises():
        get_path(make_default_config(), case.input)


# ---------------------------------------------------------------------------
# 2. replace_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(input=("array.order", "F"), output="F", id="array-order"),
        Expect(input=("async.concurrency", 99), output=99, id="async-concurrency-alias"),
        Expect(
            input=("codecs.my_codec", "my.module.MyCodec"),
            output="my.module.MyCodec",
            id="codec-new-key",
        ),
    ],
    ids=lambda c: c.id,
)
def test_replace_path(case: Expect[tuple[str, object], object]) -> None:
    key, value = case.input
    result = replace_path(make_default_config(), key, value)
    assert get_path(result, key) == case.output


def test_replace_path_is_immutable() -> None:
    """Original config is unchanged after replace_path (frozen dataclass)."""
    cfg = make_default_config()
    _ = replace_path(cfg, "array.order", "F")
    assert cfg.array.order == "C"


# ---------------------------------------------------------------------------
# 3. collect_env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input={
                "ZARR_ARRAY__ORDER": "F",
                "ZARR_ASYNC__CONCURRENCY": "32",
                "ZARR_CODECS__MY_CODEC": "my.module.MyCodec",
                "UNRELATED": "ignored",
            },
            output={
                "array.order": "F",
                "async.concurrency": 32,
                "codecs.my_codec": "my.module.MyCodec",
            },
            id="nested-and-literal",
        ),
        Expect(
            input={"ZARR_CONFIG": "/some/path.yaml", "ZARR_ARRAY__ORDER": "F"},
            output={"array.order": "F"},
            id="zarr-config-meta-var-skipped",
        ),
    ],
    ids=lambda c: c.id,
)
def test_collect_env(case: Expect[dict[str, str], dict[str, object]]) -> None:
    assert collect_env(case.input) == case.output


# ---------------------------------------------------------------------------
# 4. build_config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(input={}, output=_DEFAULT, id="empty-environ"),
        Expect(
            input={"ZARR_CONFIG": "/nonexistent/path.yaml"},
            output=_DEFAULT,
            id="zarr-config-nonexistent",
        ),
        Expect(
            input={"ZARR_JSON_INDENT": "4"},
            output=replace_path(_DEFAULT, "json_indent", 4),
            id="json-indent-env",
        ),
    ],
    ids=lambda c: c.id,
)
def test_build_config(case: Expect[dict[str, str], ZarrConfig]) -> None:
    assert build_config(environ=case.input) == case.output


# ---------------------------------------------------------------------------
# 5. apply_overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input={"array.order": "F", "codecs.x": "pkg.X"},
            output=replace_path(replace_path(_DEFAULT, "array.order", "F"), "codecs.x", "pkg.X"),
            id="array-order-and-codec",
        ),
    ],
    ids=lambda c: c.id,
)
def test_apply_overrides(case: Expect[dict[str, object], ZarrConfig]) -> None:
    assert apply_overrides(build_config(environ={}), case.input) == case.output


# ---------------------------------------------------------------------------
# 6. to_nested_dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=make_default_config(),
            output=("C", 10, "zarr.codecs.blosc.BloscCodec"),
            id="default-serialized-keys",
        ),
    ],
    ids=lambda c: c.id,
)
def test_to_nested_dict(case: Expect[ZarrConfig, tuple[str, int, str]]) -> None:
    nested = to_nested_dict(case.input)
    order, concurrency, blosc = case.output
    assert nested["array"]["order"] == order
    assert nested["async"]["concurrency"] == concurrency
    assert "async_" not in nested  # serialized key, not the Python attribute name
    assert nested["codecs"]["blosc"] == blosc


# ---------------------------------------------------------------------------
# 7. ZarrConfigManager.get — proxy string access
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(input="array.order", output="C", id="array-order"),
        Expect(input="async.concurrency", output=10, id="async-concurrency-alias"),
        Expect(input="codecs", output=DEFAULT_CODECS, id="codecs-dict"),
    ],
    ids=lambda c: c.id,
)
def test_proxy_get(case: Expect[str, object]) -> None:
    assert ZarrConfigManager().get(case.input) == case.output


@pytest.mark.parametrize(
    "case",
    [
        Expect(input=("does.not.exist", "fallback"), output="fallback", id="default-fallback"),
    ],
    ids=lambda c: c.id,
)
def test_proxy_get_with_default(case: Expect[tuple[str, object], object]) -> None:
    key, default = case.input
    assert ZarrConfigManager().get(key, default) == case.output


# ---------------------------------------------------------------------------
# 8. Removed-deprecated-key behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(input="fallback", output="fallback", id="get-with-default"),
    ],
    ids=lambda c: c.id,
)
def test_removed_deprecated_key_get_default(case: Expect[str, str]) -> None:
    """get() with a removed deprecated key and a default returns the default silently."""
    assert ZarrConfigManager().get(_REMOVED_KEY, case.input) == case.output


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(input=_REMOVED_KEY, exception=KeyError, id="get-no-default"),
    ],
    ids=lambda c: c.id,
)
def test_removed_deprecated_key_get_raises(case: ExpectFail[str]) -> None:
    """get() with a removed deprecated key and no default raises KeyError."""
    with case.raises():
        ZarrConfigManager().get(case.input)


# ---------------------------------------------------------------------------
# 9. set() must raise for both removed-deprecated keys and totally unknown keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            input={_REMOVED_KEY: "some_value"},
            exception=BadConfigError,
            id="set-removed-deprecated",
        ),
        ExpectFail(
            input={"totally.bogus.key": 1},
            exception=KeyError,
            id="set-unknown-key",
        ),
    ],
    ids=lambda c: c.id,
)
def test_set_invalid_key_raises(case: ExpectFail[dict[str, object]]) -> None:
    """set() raises for both removed deprecated keys and totally unknown structured keys."""
    with case.raises():
        ZarrConfigManager().set(case.input)


# ---------------------------------------------------------------------------
# Default config values (dedicated — direct attribute assertions are clearest here)
# ---------------------------------------------------------------------------


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
    # proxy attribute access via ZarrConfigManager
    mgr = ZarrConfigManager()
    assert mgr.array.order == "C"


# ---------------------------------------------------------------------------
# Stateful / behavioral tests (kept as dedicated functions)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# donfig not imported
# ---------------------------------------------------------------------------


def test_donfig_not_imported() -> None:
    import sys

    import zarr  # noqa: F401

    assert "donfig" not in sys.modules


# ---------------------------------------------------------------------------
# YAML codec block merging — regression for the "wipes all defaults" bug
# ---------------------------------------------------------------------------


def test_yaml_codecs_block_merges_not_replaces(tmp_path: pathlib.Path) -> None:
    """A YAML file with a codecs: block must MERGE into the defaults, not replace them."""
    yaml_file = tmp_path / "zarr.yaml"
    yaml_file.write_text("codecs:\n  bytes: my.custom.BytesCodec\n  mycodec: my.Mod.MyCodec\n")
    cfg = build_config(environ={"ZARR_CONFIG": str(yaml_file)})
    # overrides applied
    assert cfg.codecs["bytes"] == "my.custom.BytesCodec"
    assert cfg.codecs["mycodec"] == "my.Mod.MyCodec"
    # defaults PRESERVED
    assert cfg.codecs["blosc"] == "zarr.codecs.blosc.BloscCodec"
    assert cfg.codecs["zstd"] == "zarr.codecs.zstd.ZstdCodec"
    # exactly one net-new key added ("bytes" overwrites existing; "mycodec" is new)
    assert len(cfg.codecs) == len(DEFAULT_CODECS) + 1


def test_yaml_dotted_codec_name_merges(tmp_path: pathlib.Path) -> None:
    """Dotted codec keys like numcodecs.bz2 in YAML must merge, not replace the whole dict."""
    yaml_file = tmp_path / "zarr.yaml"
    yaml_file.write_text("codecs:\n  numcodecs.bz2: my.Override\n")
    cfg = build_config(environ={"ZARR_CONFIG": str(yaml_file)})
    # dotted key correctly round-tripped
    assert cfg.codecs["numcodecs.bz2"] == "my.Override"
    # all other defaults preserved
    assert cfg.codecs["blosc"] == "zarr.codecs.blosc.BloscCodec"
    assert len(cfg.codecs) == len(DEFAULT_CODECS)  # bz2 was already there; just overwritten


def test_build_config_environ_yaml_path_is_read(tmp_path: pathlib.Path) -> None:
    """ZARR_CONFIG supplied via build_config(environ=...) must actually be read."""
    yaml_file = tmp_path / "zarr.yaml"
    yaml_file.write_text("json_indent: 9\n")
    cfg = build_config(environ={"ZARR_CONFIG": str(yaml_file)})
    assert cfg.json_indent == 9
    # Non-existent path must still not raise
    cfg2 = build_config(environ={"ZARR_CONFIG": "/nonexistent/path.yaml"})
    assert cfg2.json_indent == make_default_config().json_indent


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
