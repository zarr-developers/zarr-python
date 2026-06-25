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
    # the open `codecs` dict must not be mutated in place either: a frozen
    # dataclass forbids attribute re-assignment but not `dict.__setitem__`.
    _ = replace_path(cfg, "codecs.my_codec", "my.module.MyCodec")
    assert "my_codec" not in cfg.codecs


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
# 10. Unknown keys produce a helpful "did you mean" message (get and set)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(input="array.0rder", exception=KeyError, msg=r"array\.order", id="get-typo"),
        ExpectFail(
            input="zzzzzzzz",
            exception=KeyError,
            msg="not a valid configuration key",
            id="get-no-match",
        ),
    ],
    ids=lambda c: c.id,
)
def test_get_unknown_key_message(case: ExpectFail[str]) -> None:
    """get() on an unknown key reports it and suggests the closest valid key."""
    with case.raises():
        ZarrConfigManager().get(case.input)


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            input={"array.0rder": "F"}, exception=KeyError, msg=r"array\.order", id="set-typo"
        ),
    ],
    ids=lambda c: c.id,
)
def test_set_unknown_key_message(case: ExpectFail[dict[str, object]]) -> None:
    """set() on an unknown structured key suggests the closest valid key."""
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
