from __future__ import annotations

import copy
import dataclasses
import os
import pickle
import threading
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


def _build_config_with_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> ZarrConfig:
    """Run `build_config` under a controlled ``ZARR_*`` environment.

    `build_config` delegates env/YAML reading to donfig, which reads
    ``os.environ``. This clears any ambient ``ZARR_*`` variables for determinism,
    sets the requested ones, then builds.
    """
    for name in list(os.environ):
        if name.startswith("ZARR_"):
            monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return build_config()


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
    # the open `codecs` dict must not be mutated in place either: a frozen
    # dataclass forbids attribute re-assignment but not `dict.__setitem__`.
    _ = replace_path(cfg, "codecs.my_codec", "my.module.MyCodec")
    assert "my_codec" not in cfg.codecs


# ---------------------------------------------------------------------------
# 3. build_config — env ingest (donfig reads ZARR_* from the environment)
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
        Expect(
            input={
                "ZARR_ARRAY__ORDER": "F",
                "ZARR_ASYNC__CONCURRENCY": "32",
                "ZARR_CODECS__MY_CODEC": "my.module.MyCodec",
            },
            output=replace_path(
                replace_path(
                    replace_path(_DEFAULT, "array.order", "F"),
                    "async.concurrency",
                    32,
                ),
                "codecs.my_codec",
                "my.module.MyCodec",
            ),
            id="nested-literal-and-codecs",
        ),
    ],
    ids=lambda c: c.id,
)
def test_build_config_env(
    case: Expect[dict[str, str], ZarrConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ZARR_*` environment variables are read (via donfig) and applied onto the
    typed defaults: dotted nesting (`ZARR_ARRAY__ORDER`), literal parsing
    (`ZARR_ASYNC__CONCURRENCY=32` -> int), and the open `codecs` namespace all
    work; a `ZARR_CONFIG` pointing nowhere is a no-op."""
    assert _build_config_with_env(monkeypatch, case.input) == case.output


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
    assert apply_overrides(make_default_config(), case.input) == case.output


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
# 10. Unknown keys produce a helpful "did you mean" message (get and set)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        # close match at the deepest resolvable level -> "Did you mean ...?"
        ExpectFail(
            input="arr4y", exception=KeyError, msg=r"Did you mean .array.", id="suggest-top"
        ),
        ExpectFail(
            input="array.0rder",
            exception=KeyError,
            msg=r"Did you mean .array\.order.",
            id="suggest-nested",
        ),
        ExpectFail(
            input="codecs.bl0sc",
            exception=KeyError,
            msg=r"Did you mean .codecs\.blosc.",
            id="suggest-codec",
        ),
        # no close match -> roster of available keys at the last resolvable level
        ExpectFail(input="foo", exception=KeyError, msg=r"Valid keys: .*array", id="roster-top"),
        ExpectFail(
            input="array.foo",
            exception=KeyError,
            msg=r"Valid keys under .array.: .*order",
            id="roster-nested",
        ),
        ExpectFail(
            input="codecs.zzzzzzzz",
            exception=KeyError,
            msg=r"under .codecs.: .*more\)",
            id="roster-truncated",
        ),
    ],
    ids=lambda c: c.id,
)
def test_get_unknown_key_message(case: ExpectFail[str]) -> None:
    """get() on an unknown key suggests the closest key or lists what's available."""
    with case.raises():
        ZarrConfigManager().get(case.input)


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            input={"array.0rder": "F"},
            exception=KeyError,
            msg=r"Did you mean .array\.order.",
            id="set-suggest",
        ),
        ExpectFail(
            input={"array.foo": "F"},
            exception=KeyError,
            msg=r"Valid keys under .array.: .*order",
            id="set-roster",
        ),
    ],
    ids=lambda c: c.id,
)
def test_set_unknown_key_message(case: ExpectFail[dict[str, object]]) -> None:
    """set() shares the same helpful unknown-key error as get()."""
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
        assert seen == 77  # a permanent set is on the shared global base
    finally:
        cfg.reset()


def test_permanent_set_cross_thread_last_writer_wins() -> None:
    """A permanent `set` from any thread updates the shared global base, so a later
    permanent `set` in another thread is visible everywhere — even after the first
    thread already did its own permanent `set`. (Regression: the first set used to
    pin the setting thread's view, so it kept seeing its own stale value.)"""
    cfg = ZarrConfigManager()
    cfg.set({"async.concurrency": 1})
    worker = threading.Thread(target=lambda: cfg.set({"async.concurrency": 999}))
    worker.start()
    worker.join()
    assert cfg.get("async.concurrency") == 999


def test_with_block_is_global_and_reverts_only_its_keys() -> None:
    """A `with config.set(...)` applies globally (donfig semantics: it is NOT
    isolated to the calling thread) and, on exit, reverts only the keys it set —
    leaving a concurrent permanent `set` to a *different* key intact."""
    cfg = ZarrConfigManager()
    cfg.set({"async.concurrency": 5})
    with cfg.set({"async.concurrency": 999}):
        assert cfg.get("async.concurrency") == 999  # visible in this context
        with ThreadPoolExecutor(max_workers=1) as ex:
            seen = ex.submit(lambda: cfg.get("async.concurrency")).result()
        assert seen == 999  # globally visible: the worker sees the override too
        # a concurrent permanent set to a DIFFERENT key must survive block exit
        cfg.set({"array.order": "F"})
    assert cfg.get("async.concurrency") == 5  # the block's key reverted
    assert cfg.get("array.order") == "F"  # the other key was not clobbered


def test_permanent_set_inside_with_block_persists() -> None:
    """On `with config.set(...)` exit, only the block's own keys are reverted, so a
    permanent `set` to a different key made inside the block persists."""
    cfg = ZarrConfigManager()
    with cfg.set({"array.order": "F"}):
        cfg.set({"async.concurrency": 5})  # permanent set to a different key
    assert cfg.get("array.order") == "C"  # the block's key reverted
    assert cfg.get("async.concurrency") == 5  # the other key persisted


def test_with_block_removes_newly_added_codec_key_on_exit() -> None:
    """A scoped `set` that *adds* a new codec key removes it again on block exit,
    while a pre-existing key it overrode is restored to its prior value."""
    cfg = ZarrConfigManager()
    assert "brand_new_codec" not in cfg.codecs
    with cfg.set({"codecs.brand_new_codec": "pkg.New", "codecs.blosc": "pkg.OverrideBlosc"}):
        assert cfg.codecs["brand_new_codec"] == "pkg.New"
        assert cfg.codecs["blosc"] == "pkg.OverrideBlosc"
    assert "brand_new_codec" not in cfg.codecs  # newly added key removed
    assert cfg.codecs["blosc"] == DEFAULT_CODECS["blosc"]  # existing key restored


# ---------------------------------------------------------------------------
# Subtree item access (donfig back-compat for `config.get("array")["order"]`)
# ---------------------------------------------------------------------------


def test_subtree_get_item_access_matches_attribute() -> None:
    """A subtree `get` returns a typed dataclass that also supports donfig-style
    item access: `["order"]`, `.order`, and `get("array.order")` all agree, and
    an unknown key raises `KeyError` like the old dicts did."""
    cfg = ZarrConfigManager()
    array = cfg.get("array")
    assert array["order"] == array.order == cfg.get("array.order")
    assert array["sharding_coalesce_max_bytes"] == array.sharding_coalesce_max_bytes
    with pytest.raises(KeyError):
        array["does_not_exist"]


def test_config_node_dotted_and_alias_item_access() -> None:
    """Item access on a config node resolves dotted keys, the `async` alias, and
    the open `codecs` mapping, mirroring `get_path`."""
    cfg = make_default_config()
    assert cfg["array.order"] == "C"
    assert cfg["async.concurrency"] == 10  # the `async` alias resolves
    assert cfg.async_["concurrency"] == cfg.async_.concurrency  # node item access
    assert cfg["codecs.bytes"] == DEFAULT_CODECS["bytes"]


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
    # apply a permanent override in this context, then rebuild over a changed env
    mgr.set({"array.order": "F"})
    assert mgr.get("array.order") == "F"
    # change the environment so a rebuild differs, then refresh
    monkeypatch.setenv("ZARR_JSON_INDENT", "7")
    mgr.refresh()
    # refresh rebuilds the global base and must be visible in THIS context
    assert mgr.get("json_indent") == 7
    assert mgr.get("array.order") == "C"  # the prior permanent set is gone after rebuild


# ---------------------------------------------------------------------------
# Tolerant ingest: unknown env/YAML keys must warn and be skipped, not crash
# ---------------------------------------------------------------------------


def test_build_config_unknown_env_key_warns_and_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_config with an unrecognized env var warns and skips it; known keys still apply."""
    with pytest.warns(UserWarning, match="future.key"):
        cfg = _build_config_with_env(
            monkeypatch, {"ZARR_FUTURE__KEY": "1", "ZARR_ARRAY__ORDER": "F"}
        )
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
# donfig is used as the env/YAML reader
# ---------------------------------------------------------------------------


def test_donfig_is_used_for_ingest() -> None:
    """donfig backs env/YAML ingest, so building a config imports it and its
    reader produces a mapping we can consume."""
    import donfig

    assert isinstance(donfig.Config("zarr").config, dict)
    # build_config runs donfig's reader and returns the typed representation
    assert isinstance(build_config(), ZarrConfig)


# ---------------------------------------------------------------------------
# YAML config files (read by donfig, applied onto the typed defaults)
# ---------------------------------------------------------------------------


def test_yaml_codecs_block_merges_not_replaces(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A YAML file with a codecs: block must MERGE into the defaults, not replace them."""
    yaml_file = tmp_path / "zarr.yaml"
    yaml_file.write_text("codecs:\n  bytes: my.custom.BytesCodec\n  mycodec: my.Mod.MyCodec\n")
    cfg = _build_config_with_env(monkeypatch, {"ZARR_CONFIG": str(yaml_file)})
    # overrides applied
    assert cfg.codecs["bytes"] == "my.custom.BytesCodec"
    assert cfg.codecs["mycodec"] == "my.Mod.MyCodec"
    # defaults PRESERVED
    assert cfg.codecs["blosc"] == "zarr.codecs.blosc.BloscCodec"
    assert cfg.codecs["zstd"] == "zarr.codecs.zstd.ZstdCodec"
    # exactly one net-new key added ("bytes" overwrites existing; "mycodec" is new)
    assert len(cfg.codecs) == len(DEFAULT_CODECS) + 1


def test_yaml_dotted_codec_name_merges(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dotted codec keys like numcodecs.bz2 in YAML must merge, not replace the whole dict."""
    yaml_file = tmp_path / "zarr.yaml"
    yaml_file.write_text("codecs:\n  numcodecs.bz2: my.Override\n")
    cfg = _build_config_with_env(monkeypatch, {"ZARR_CONFIG": str(yaml_file)})
    # dotted key correctly round-tripped
    assert cfg.codecs["numcodecs.bz2"] == "my.Override"
    # all other defaults preserved
    assert cfg.codecs["blosc"] == "zarr.codecs.blosc.BloscCodec"
    assert len(cfg.codecs) == len(DEFAULT_CODECS)  # bz2 was already there; just overwritten


def test_yaml_file_via_zarr_config_is_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A YAML file pointed to by ZARR_CONFIG is read; a non-existent path is a no-op."""
    yaml_file = tmp_path / "zarr.yaml"
    yaml_file.write_text("json_indent: 9\n")
    cfg = _build_config_with_env(monkeypatch, {"ZARR_CONFIG": str(yaml_file)})
    assert cfg.json_indent == 9
    # Non-existent path must still not raise, and leaves defaults intact
    cfg2 = _build_config_with_env(monkeypatch, {"ZARR_CONFIG": "/nonexistent/path.yaml"})
    assert cfg2.json_indent == make_default_config().json_indent


def test_yaml_config_directory_is_scanned(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ZARR_CONFIG may point at a directory; a zarr.yaml inside it is loaded."""
    (tmp_path / "zarr.yaml").write_text("array:\n  order: F\njson_indent: 8\n")
    cfg = _build_config_with_env(monkeypatch, {"ZARR_CONFIG": str(tmp_path)})
    assert cfg.array.order == "F"
    assert cfg.json_indent == 8


def test_env_var_overrides_yaml(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Precedence is defaults < YAML < environment: an env var wins over a YAML file."""
    yaml_file = tmp_path / "zarr.yaml"
    yaml_file.write_text("json_indent: 3\n")
    cfg = _build_config_with_env(
        monkeypatch, {"ZARR_CONFIG": str(yaml_file), "ZARR_JSON_INDENT": "9"}
    )
    assert cfg.json_indent == 9


def test_yaml_unknown_key_warns_and_skips(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrecognized key in a YAML file warns and is skipped; known keys still apply
    (so a version-skewed config file can't prevent `import zarr`)."""
    yaml_file = tmp_path / "zarr.yaml"
    yaml_file.write_text("future_key: 1\narray:\n  order: F\n")
    with pytest.warns(UserWarning, match="future_key"):
        cfg = _build_config_with_env(monkeypatch, {"ZARR_CONFIG": str(yaml_file)})
    assert cfg.array.order == "F"


def test_discovers_home_config_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `zarr.yaml` in `~/.config/zarr` is discovered (donfig's primary location),
    and `ZARR_CONFIG` takes precedence over it while home-only keys still apply.

    Regression guard against silently dropping donfig's config-file locations.
    """
    home = tmp_path / "home"
    config_dir = home / ".config" / "zarr"
    config_dir.mkdir(parents=True)
    (config_dir / "zarr.yaml").write_text("json_indent: 1\narray:\n  order: F\n")
    # Redirect `~` to the temporary home so the real user config is not consulted;
    # donfig computes its `~/.config/zarr` path via os.path.expanduser.
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(home) if p == "~" else p)

    # Without ZARR_CONFIG, the home config directory is picked up.
    cfg = _build_config_with_env(monkeypatch, {})
    assert cfg.json_indent == 1
    assert cfg.array.order == "F"

    # ZARR_CONFIG (highest precedence) overrides the home file's overlapping keys.
    override = tmp_path / "override.yaml"
    override.write_text("json_indent: 2\n")
    cfg2 = _build_config_with_env(monkeypatch, {"ZARR_CONFIG": str(override)})
    assert cfg2.json_indent == 2  # ZARR_CONFIG wins
    assert cfg2.array.order == "F"  # home-only key still applies


# ---------------------------------------------------------------------------
# Drift-protection: every structured leaf key must have a get() overload
# ---------------------------------------------------------------------------


def _structured_leaf_specs(cfg_cls: type, prefix: str = "") -> dict[str, object]:
    """Walk a settings dataclass recursively and return ``{dotted_key: resolved_type}``.

    Uses ``typing.get_type_hints`` instead of ``f.type`` so that the
    ``from __future__ import annotations`` string-annotation form is resolved
    to real types before ``dataclasses.is_dataclass`` is called.  The open
    ``codecs`` mapping is intentionally excluded.
    """
    specs: dict[str, object] = {}
    resolved_hints = typing.get_type_hints(cfg_cls)
    for f in dataclasses.fields(cfg_cls):
        serialized = _SERIALIZED_NAMES.get(f.name, f.name)
        key = f"{prefix}.{serialized}" if prefix else serialized
        resolved_type = resolved_hints[f.name]
        if dataclasses.is_dataclass(resolved_type):
            specs.update(_structured_leaf_specs(typing.cast(type, resolved_type), key))
        elif f.name == "codecs":
            # open mapping — intentionally not enumerated
            continue
        else:
            specs[key] = resolved_type
    return specs


def _structured_leaf_keys(cfg_cls: type, prefix: str = "") -> list[str]:
    """Return every dotted leaf key for a settings dataclass (derived from specs)."""
    return list(_structured_leaf_specs(cfg_cls, prefix))


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


def test_get_overload_return_types_match_fields() -> None:
    """Assert that each get() overload's return type matches the dataclass field type.

    Builds two maps using ``typing.get_type_hints`` — one from the dataclass
    field annotations, one from the overload return hints — then compares them
    key by key.  A mismatch (e.g. ``-> str`` instead of ``-> Literal["C","F"]``)
    is reported as a clear failure rather than a missing-overload failure.
    """
    # Build map: key -> return type from overloads
    overloads = typing.get_overloads(ZarrConfigManager.get)
    overload_return: dict[str, object] = {}
    for ov in overloads:
        hints = typing.get_type_hints(ov)
        key_hint = hints.get("key")
        if typing.get_origin(key_hint) is typing.Literal:
            (literal_val,) = typing.get_args(key_hint)
            overload_return[literal_val] = hints["return"]

    # Build map: key -> field type from the dataclass schema
    field_specs = _structured_leaf_specs(ZarrConfig)

    missing: list[str] = []
    mismatched: list[str] = []
    for key, expected_type in field_specs.items():
        if key not in overload_return:
            missing.append(f"  {key!r}: missing overload")
        elif overload_return[key] != expected_type:
            mismatched.append(
                f"  {key!r}: overload returns {overload_return[key]!r},"
                f" field type is {expected_type!r}"
            )

    errors: list[str] = []
    if missing:
        errors.append("get() overloads missing for keys:\n" + "\n".join(missing))
    if mismatched:
        errors.append(
            "get() overload return types do not match field types:\n" + "\n".join(mismatched)
        )
    assert not errors, "\n\n".join(errors)


# ---------------------------------------------------------------------------
# Regression tests for the #4101 review
# ---------------------------------------------------------------------------


def test_env_codec_override_canonicalizes_hyphenated_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ZARR_CODECS__VLEN_UTF8` must override the hyphenated default `vlen-utf8`.

    Environment variables can't contain hyphens, so the flattened key is
    `codecs.vlen_utf8`; without canonicalization it lands under a dead key while
    the registry keeps reading the untouched `vlen-utf8` default.
    """
    cfg = _build_config_with_env(monkeypatch, {"ZARR_CODECS__VLEN_UTF8": "my.Override"})
    assert cfg.codecs["vlen-utf8"] == "my.Override"
    assert "vlen_utf8" not in cfg.codecs  # no dead underscore key
    # underscore-named defaults (e.g. sharding_indexed) are untouched
    cfg2 = _build_config_with_env(monkeypatch, {"ZARR_CODECS__SHARDING_INDEXED": "my.Shard"})
    assert cfg2.codecs["sharding_indexed"] == "my.Shard"
    # a brand-new codec name with underscores stays as written
    cfg3 = _build_config_with_env(monkeypatch, {"ZARR_CODECS__MY_NEW": "my.New"})
    assert cfg3.codecs["my_new"] == "my.New"


@pytest.mark.parametrize("key", ["array.order.upper", "default_zarr_format.numerator"])
def test_get_does_not_descend_into_scalar_attributes(key: str) -> None:
    """A dotted key that walks past a scalar leaf must raise, not resolve a stray
    Python attribute (e.g. `str.upper` / `int.numerator`)."""
    with pytest.raises(KeyError):
        ZarrConfigManager().get(key)


def test_set_rejects_descending_into_scalar() -> None:
    with pytest.raises(KeyError):
        ZarrConfigManager().set({"array.order.upper": "X"})


def test_codecs_public_mapping_is_read_only() -> None:
    """`config.codecs` exposes a read-only view, so a live snapshot can't be
    mutated in place; a mutable copy is still available via `dict(...)`."""
    cfg = ZarrConfigManager()
    with pytest.raises(TypeError):
        cfg.codecs["blosc"] = "x"  # type: ignore[index]
    assert dict(cfg.codecs)["blosc"] == cfg.codecs["blosc"]


def test_config_snapshot_is_picklable_and_deepcopyable() -> None:
    """A `ZarrConfig` snapshot must stay picklable / deep-copyable: the `codecs`
    field is a plain dict, and only the public view is a read-only proxy."""
    cfg = replace_path(make_default_config(), "codecs.x", "pkg.X")
    assert pickle.loads(pickle.dumps(cfg)) == cfg
    assert copy.deepcopy(cfg) == cfg


def test_set_structured_subtree_dict_is_rejected() -> None:
    """Assigning a dict to a whole structured subtree is rejected (it would drop
    sibling fields and break attribute access); leaf keys must be used instead."""
    cfg = ZarrConfigManager()
    with pytest.raises(TypeError):
        cfg.set({"array": {"order": "F"}})
    # the leaf-key form works and preserves siblings
    cfg.set({"array.order": "F"})
    assert cfg.get("array.order") == "F"
    assert cfg.get("array.write_empty_chunks") is False


def test_manager_is_not_iterable() -> None:
    """Iterating the manager raises a clear TypeError rather than falling into the
    legacy integer-index protocol (which gave a confusing error)."""
    with pytest.raises(TypeError):
        list(ZarrConfigManager())


def test_subtree_node_supports_mapping_style_reads() -> None:
    """A subtree returned by `get` supports donfig-style `in`, iteration, `keys`,
    and `dict()` alongside attribute and item access."""
    array = ZarrConfigManager().get("array")
    assert "order" in array
    assert "bogus" not in array
    assert "order" in list(array)
    assert "order" in set(array.keys())
    assert dict(array)["order"] == array.order == "C"
    assert len(array) == len(dataclasses.fields(array))


def test_manager_item_and_membership_access() -> None:
    """donfig-style `config["k"]` and `"k" in config` mirror `get`."""
    cfg = ZarrConfigManager()
    assert cfg["array.order"] == cfg.get("array.order")
    assert "array.order" in cfg
    assert "bogus.key" not in cfg
    assert 123 not in cfg  # non-string key is absent, not an error


def test_config_alias_preserved() -> None:
    """`Config` remains importable as an alias of `ZarrConfigManager` (pre-typed
    imports and `isinstance` checks keep working)."""
    from zarr.core.config import Config

    assert Config is ZarrConfigManager
    assert isinstance(ZarrConfigManager(), Config)


def test_concurrent_permanent_sets_to_distinct_keys_all_survive() -> None:
    """A permanent `set` rebuilds the whole snapshot from `_base`; the manager
    locks that read-modify-write so concurrent sets to distinct keys don't lose
    updates."""
    cfg = ZarrConfigManager()
    n = 64
    barrier = threading.Barrier(n)

    def worker(i: int) -> None:
        barrier.wait()
        cfg.set({f"codecs.k{i}": f"v{i}"})

    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(worker, range(n)))

    codecs = cfg.get("codecs")
    assert all(codecs.get(f"k{i}") == f"v{i}" for i in range(n))


# ---------------------------------------------------------------------------
# Static-typing smoke test (only checked by mypy, not executed at runtime)
# ---------------------------------------------------------------------------

if typing.TYPE_CHECKING:

    def _typing_smoke(cfg: ZarrConfigManager) -> None:
        # --- positive assertions: each distinct return shape ---
        typing.assert_type(cfg.get("array.order"), typing.Literal["C", "F"])
        typing.assert_type(cfg.get("async.concurrency"), int)
        typing.assert_type(cfg.get("array.write_empty_chunks"), bool)
        typing.assert_type(cfg.get("async.timeout"), float | None)
        typing.assert_type(cfg.get("threading.max_workers"), int | None)
        typing.assert_type(cfg.get("default_zarr_format"), typing.Literal[2, 3])
        typing.assert_type(cfg.get("buffer"), str)
        typing.assert_type(cfg.array.order, typing.Literal["C", "F"])

        # --- negative: precision-from-above guards ---
        # The return type is Literal["C","F"], which is narrower than str.
        # If the overload were widened to -> str, assert_type would pass and
        # the ignore below would become unused, causing warn_unused_ignores to
        # fail CI.
        typing.assert_type(cfg.get("array.order"), str)  # type: ignore[assert-type]
        typing.assert_type(cfg.get("default_zarr_format"), int)  # type: ignore[assert-type]

        # --- negative: bad key type must be rejected by all overloads ---
        cfg.get(123)  # type: ignore[call-overload]
