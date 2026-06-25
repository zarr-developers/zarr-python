# Statically-typed configuration (drop donfig) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace donfig with a hand-typed, dataclass-backed configuration object that preserves donfig's dotted-string API exactly while adding precise static types.

**Architecture:** A tree of frozen dataclasses is the schema/source-of-truth. A process-global base snapshot plus a `ContextVar` scope provide donfig-compatible mutable-global semantics with `with`-restore. A proxy object (`config`) exposes typed attribute access and a hand-written overloaded `get`/`set` string API, plus env-var and YAML ingest and deprecation handling.

**Tech Stack:** Python 3.11+, `dataclasses`, `typing.overload`, `contextvars`, PyYAML, pytest, mypy (strict).

## Global Constraints

- Backwards compatibility is the top priority. These must keep identical behavior: `config.get("a.b.c")`, `config.get("a.b.c", default)`, `config.get("codecs", {}).get(key)`, permanent `config.set({...})`, `with config.set({...})`, `config.reset()`, `config.refresh()`, `config.enable_gpu()`, `config.defaults`, `BadConfigError`, `parse_indexing_order`, `ZARR_FOO__BAR` env ingest, YAML ingest, deprecation warnings.
- Public import paths unchanged: `from zarr.core.config import config, BadConfigError, parse_indexing_order`; `zarr.config`.
- mypy strict must pass; PEP8, max line length 100 (prefer <90); numpydoc docstrings on public API.
- Use `uv run` for all pytest/mypy/python invocations (e.g. `uv run pytest ...`).
- No new runtime dependency on `tytr`. Overloads are hand-written.
- The serialized key for the async namespace stays `"async"`; the dataclass field is `async_`.
- `codecs` is an open `Mapping[str, str]` subtree.
- Keep all current config keys, defaults, and the existing `deprecations` mapping verbatim.
- Frequent commits; one logical change per commit.

## File Structure

- `src/zarr/core/config.py` — **rewritten** (single module, preserves import paths). Contains: schema dataclasses, path helpers, ingest functions, deprecations, the `ZarrConfigManager` proxy, the module-level `config` instance, `BadConfigError`, `parse_indexing_order`.
- `src/zarr/__init__.py` — remove `"donfig"` from the `required` version-report list.
- `pyproject.toml` — remove the three donfig entries; ensure `pyyaml` is a declared runtime dependency.
- `tests/test_config.py` — existing suite; update only the `defaults` structural assertion.
- `tests/test_config_typed.py` — **new**: schema/helpers/ingest/state/drift/typing unit tests.
- `changes/<pr>.misc.md` — **new**: changelog entry.

---

### Task 1: Schema dataclasses + path helpers

**Files:**
- Modify: `src/zarr/core/config.py` (add new code; do not remove donfig yet)
- Test: `tests/test_config_typed.py`

**Interfaces:**
- Produces:
  - Frozen dataclasses `ArraySettings`, `AsyncSettings`, `ThreadingSettings`, `CodecPipelineSettings`, `ZarrConfig`.
  - `DEFAULT_CODECS: dict[str, str]` — the default codec-name→import-path map.
  - `make_default_config() -> ZarrConfig`.
  - `get_path(cfg: ZarrConfig, key: str) -> Any` — read a dotted key; raises `KeyError` if absent.
  - `replace_path(cfg: ZarrConfig, key: str, value: Any) -> ZarrConfig` — return a new snapshot with the dotted key updated.
  - `to_nested_dict(cfg: ZarrConfig) -> dict[str, Any]` — donfig-style nested dict using serialized keys (`"async"`, not `"async_"`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config_typed.py`:

```python
from __future__ import annotations

import pytest

from zarr.core.config import (
    DEFAULT_CODECS,
    ZarrConfig,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_typed.py -v`
Expected: FAIL — `ImportError` (names not yet defined).

- [ ] **Step 3: Write minimal implementation**

Add near the top of `src/zarr/core/config.py` (after `from __future__ import annotations` and imports; add `from dataclasses import dataclass, field, fields, replace`, `from collections.abc import Mapping`, `from typing import Any`):

```python
DEFAULT_CODECS: dict[str, str] = {
    "blosc": "zarr.codecs.blosc.BloscCodec",
    "gzip": "zarr.codecs.gzip.GzipCodec",
    "zstd": "zarr.codecs.zstd.ZstdCodec",
    "bytes": "zarr.codecs.bytes.BytesCodec",
    "endian": "zarr.codecs.bytes.BytesCodec",
    "crc32c": "zarr.codecs.crc32c_.Crc32cCodec",
    "sharding_indexed": "zarr.codecs.sharding.ShardingCodec",
    "transpose": "zarr.codecs.transpose.TransposeCodec",
    "vlen-utf8": "zarr.codecs.vlen_utf8.VLenUTF8Codec",
    "vlen-bytes": "zarr.codecs.vlen_utf8.VLenBytesCodec",
    "numcodecs.bz2": "zarr.codecs.numcodecs.BZ2",
    "numcodecs.crc32": "zarr.codecs.numcodecs.CRC32",
    "numcodecs.crc32c": "zarr.codecs.numcodecs.CRC32C",
    "numcodecs.lz4": "zarr.codecs.numcodecs.LZ4",
    "numcodecs.lzma": "zarr.codecs.numcodecs.LZMA",
    "numcodecs.zfpy": "zarr.codecs.numcodecs.ZFPY",
    "numcodecs.adler32": "zarr.codecs.numcodecs.Adler32",
    "numcodecs.astype": "zarr.codecs.numcodecs.AsType",
    "numcodecs.bitround": "zarr.codecs.numcodecs.BitRound",
    "numcodecs.blosc": "zarr.codecs.numcodecs.Blosc",
    "numcodecs.delta": "zarr.codecs.numcodecs.Delta",
    "numcodecs.fixedscaleoffset": "zarr.codecs.numcodecs.FixedScaleOffset",
    "numcodecs.fletcher32": "zarr.codecs.numcodecs.Fletcher32",
    "numcodecs.gzip": "zarr.codecs.numcodecs.GZip",
    "numcodecs.jenkins_lookup3": "zarr.codecs.numcodecs.JenkinsLookup3",
    "numcodecs.pcodec": "zarr.codecs.numcodecs.PCodec",
    "numcodecs.packbits": "zarr.codecs.numcodecs.PackBits",
    "numcodecs.shuffle": "zarr.codecs.numcodecs.Shuffle",
    "numcodecs.quantize": "zarr.codecs.numcodecs.Quantize",
    "numcodecs.zlib": "zarr.codecs.numcodecs.Zlib",
    "numcodecs.zstd": "zarr.codecs.numcodecs.Zstd",
}

# Map serialized dotted-key segments to Python field names where they differ
# (Python keywords cannot be used as identifiers).
_FIELD_ALIASES: dict[str, str] = {"async": "async_"}
_SERIALIZED_NAMES: dict[str, str] = {v: k for k, v in _FIELD_ALIASES.items()}


@dataclass(frozen=True, slots=True)
class ArraySettings:
    order: Literal["C", "F"] = "C"
    write_empty_chunks: bool = False
    read_missing_chunks: bool = True
    target_shard_size_bytes: int | None = None
    rectilinear_chunks: bool = False
    sharding_coalesce_max_gap_bytes: int = 1 << 20
    sharding_coalesce_max_bytes: int = 16 << 20


@dataclass(frozen=True, slots=True)
class AsyncSettings:
    concurrency: int = 10
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class ThreadingSettings:
    max_workers: int | None = None


@dataclass(frozen=True, slots=True)
class CodecPipelineSettings:
    path: str = "zarr.core.codec_pipeline.BatchedCodecPipeline"
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class ZarrConfig:
    default_zarr_format: Literal[2, 3] = 3
    array: ArraySettings = field(default_factory=ArraySettings)
    async_: AsyncSettings = field(default_factory=AsyncSettings)
    threading: ThreadingSettings = field(default_factory=ThreadingSettings)
    json_indent: int = 2
    codec_pipeline: CodecPipelineSettings = field(default_factory=CodecPipelineSettings)
    codecs: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_CODECS))
    buffer: str = "zarr.buffer.cpu.Buffer"
    ndbuffer: str = "zarr.buffer.cpu.NDBuffer"


def make_default_config() -> ZarrConfig:
    """Return a fresh `ZarrConfig` populated with the built-in defaults."""
    return ZarrConfig()


def _resolve_field(obj: Any, segment: str) -> str:
    """Translate a serialized key segment to the dataclass field name."""
    return _FIELD_ALIASES.get(segment, segment)


def get_path(cfg: ZarrConfig, key: str) -> Any:
    """Read a dotted-string key from a `ZarrConfig` snapshot.

    Raises
    ------
    KeyError
        If the key does not resolve to a value.
    """
    obj: Any = cfg
    segments = key.split(".")
    for i, segment in enumerate(segments):
        if isinstance(obj, Mapping):
            # remaining segments index into an open mapping (e.g. codecs.*)
            remainder = ".".join(segments[i:])
            try:
                return obj[remainder]
            except KeyError:
                raise KeyError(key) from None
        field_name = _resolve_field(obj, segment)
        if not hasattr(obj, field_name):
            raise KeyError(key)
        obj = getattr(obj, field_name)
    return obj


def replace_path(cfg: ZarrConfig, key: str, value: Any) -> ZarrConfig:
    """Return a new `ZarrConfig` with the dotted-string key set to ``value``."""
    segments = key.split(".")
    return _replace_recursive(cfg, segments, value, key)


def _replace_recursive(obj: Any, segments: list[str], value: Any, key: str) -> Any:
    segment = segments[0]
    if isinstance(obj, Mapping):
        remainder = ".".join(segments)
        return {**obj, remainder: value}
    field_name = _resolve_field(obj, segment)
    if not hasattr(obj, field_name):
        raise KeyError(key)
    if len(segments) == 1:
        return replace(obj, **{field_name: value})
    child = getattr(obj, field_name)
    new_child = _replace_recursive(child, segments[1:], value, key)
    return replace(obj, **{field_name: new_child})


def to_nested_dict(cfg: ZarrConfig) -> dict[str, Any]:
    """Convert a `ZarrConfig` to a donfig-style nested dict (serialized keys)."""

    def convert(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return dict(obj)
        if hasattr(type(obj), "__dataclass_fields__"):
            out: dict[str, Any] = {}
            for f in fields(obj):
                serialized = _SERIALIZED_NAMES.get(f.name, f.name)
                out[serialized] = convert(getattr(obj, f.name))
            return out
        return obj

    return convert(cfg)  # type: ignore[no-any-return]
```

Ensure `Literal` and `Any` are imported at the top of the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_typed.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/zarr/core/config.py tests/test_config_typed.py
git commit -m "feat(config): add frozen dataclass schema and path helpers"
```

---

### Task 2: Env-var and YAML ingest

**Files:**
- Modify: `src/zarr/core/config.py`
- Test: `tests/test_config_typed.py`

**Interfaces:**
- Consumes: `ZarrConfig`, `replace_path` (Task 1).
- Produces:
  - `collect_env(environ: Mapping[str, str]) -> dict[str, Any]` — flat dotted-key → value map from `ZARR_*` vars.
  - `collect_yaml(paths: list[str]) -> dict[str, Any]` — flat dotted-key map merged from YAML files (missing files skipped).
  - `apply_overrides(cfg: ZarrConfig, overrides: Mapping[str, Any]) -> ZarrConfig`.
  - `build_config(environ: Mapping[str, str] | None = None) -> ZarrConfig` — defaults < YAML < env.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config_typed.py`:

```python
from zarr.core.config import apply_overrides, build_config, collect_env


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_typed.py -k "env or precedence" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/zarr/core/config.py` (add `import ast`, `import os`, `import contextlib` at top):

```python
ENV_PREFIX = "ZARR_"


def _parse_env_value(raw: str) -> Any:
    """Parse an env value with ``ast.literal_eval``; fall back to the raw string."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def collect_env(environ: Mapping[str, str]) -> dict[str, Any]:
    """Collect ``ZARR_*`` environment variables into a flat dotted-key map.

    ``ZARR_FOO__BAR_BAZ=1`` becomes ``{"foo.bar_baz": 1}`` — the key is
    lower-cased and ``__`` denotes nested access.
    """
    out: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        body = name[len(ENV_PREFIX) :]
        dotted = body.lower().replace("__", ".")
        out[dotted] = _parse_env_value(raw)
    return out


def _config_search_paths() -> list[str]:
    """Standard YAML config locations, mirroring donfig's search order."""
    paths: list[str] = []
    env_path = os.environ.get("ZARR_CONFIG")
    if env_path:
        paths.append(env_path)
    paths.append(os.path.join(os.path.expanduser("~"), ".config", "zarr"))
    return paths


def collect_yaml(paths: list[str]) -> dict[str, Any]:
    """Merge YAML config files found at ``paths`` into a flat dotted-key map."""
    import yaml

    merged: dict[str, Any] = {}
    for path in paths:
        candidates: list[str] = []
        if os.path.isdir(path):
            for fn in sorted(os.listdir(path)):
                if fn.endswith((".yaml", ".yml")):
                    candidates.append(os.path.join(path, fn))
        elif os.path.isfile(path):
            candidates.append(path)
        for candidate in candidates:
            with contextlib.suppress(FileNotFoundError):
                with open(candidate) as fh:
                    data = yaml.safe_load(fh)
                if isinstance(data, Mapping):
                    merged.update(_flatten_mapping(data))
    return merged


def _flatten_mapping(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, Mapping) and k not in ("codecs",):
            out.update(_flatten_mapping(v, key))
        else:
            out[key] = v
    return out


def apply_overrides(cfg: ZarrConfig, overrides: Mapping[str, Any]) -> ZarrConfig:
    """Apply a flat dotted-key override map to a snapshot."""
    for key, value in overrides.items():
        cfg = replace_path(cfg, key, value)
    return cfg


def build_config(environ: Mapping[str, str] | None = None) -> ZarrConfig:
    """Build the base snapshot: defaults < YAML files < environment variables."""
    if environ is None:
        environ = os.environ
    cfg = make_default_config()
    cfg = apply_overrides(cfg, collect_yaml(_config_search_paths()))
    cfg = apply_overrides(cfg, collect_env(environ))
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_typed.py -k "env or precedence" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zarr/core/config.py tests/test_config_typed.py
git commit -m "feat(config): add env-var and YAML ingest"
```

---

### Task 3: State holder + proxy with typed get/set/reset

**Files:**
- Modify: `src/zarr/core/config.py`
- Test: `tests/test_config_typed.py`

**Interfaces:**
- Consumes: `ZarrConfig`, `build_config`, `get_path`, `replace_path`, `to_nested_dict`, `deprecations` (Tasks 1–2 + existing).
- Produces:
  - `class ZarrConfigManager` with: typed properties (`array`, `async_`, `threading`, `codec_pipeline`, `default_zarr_format`, `json_indent`, `codecs`, `buffer`, `ndbuffer`); overloaded `get(key, default=...)`; `set(mapping) -> _ConfigSet`; `reset()`; `refresh()`; `enable_gpu()`; `defaults` property; compat shims `to_dict()`, `update(mapping)`, `pprint()`.
  - module-level `config: ZarrConfigManager`.
  - `_ConfigSet` context manager.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config_typed.py`:

```python
from concurrent.futures import ThreadPoolExecutor

from zarr.core.config import ZarrConfigManager


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_typed.py -k "proxy or set_permanent or worker or defaults_and" -v`
Expected: FAIL — `ImportError` / attribute errors.

- [ ] **Step 3: Write minimal implementation**

Add to `src/zarr/core/config.py` (add `from contextvars import ContextVar`, `from typing import overload`, `import warnings` if not present):

```python
_MISSING = object()


class _ConfigSet:
    """Context manager returned by ``ZarrConfigManager.set``.

    The change is applied immediately (permanent by default); using the object
    as a ``with`` block restores the prior state on exit.
    """

    def __init__(self, manager: ZarrConfigManager, prev_base: ZarrConfig, token: Any) -> None:
        self._manager = manager
        self._prev_base = prev_base
        self._token = token

    def __enter__(self) -> _ConfigSet:
        return self

    def __exit__(self, *exc: object) -> None:
        self._manager._restore(self._prev_base, self._token)


class ZarrConfigManager:
    """Typed, donfig-compatible configuration object."""

    def __init__(self) -> None:
        self._base: ZarrConfig = build_config()
        self._scope: ContextVar[ZarrConfig] = ContextVar("zarr_config_scope")

    # --- state resolution -------------------------------------------------
    def _current(self) -> ZarrConfig:
        return self._scope.get(self._base)

    def _restore(self, prev_base: ZarrConfig, token: Any) -> None:
        self._base = prev_base
        self._scope.reset(token)

    # --- typed attribute access ------------------------------------------
    @property
    def default_zarr_format(self) -> Literal[2, 3]:
        return self._current().default_zarr_format

    @property
    def array(self) -> ArraySettings:
        return self._current().array

    @property
    def async_(self) -> AsyncSettings:
        return self._current().async_

    @property
    def threading(self) -> ThreadingSettings:
        return self._current().threading

    @property
    def codec_pipeline(self) -> CodecPipelineSettings:
        return self._current().codec_pipeline

    @property
    def json_indent(self) -> int:
        return self._current().json_indent

    @property
    def codecs(self) -> Mapping[str, str]:
        return self._current().codecs

    @property
    def buffer(self) -> str:
        return self._current().buffer

    @property
    def ndbuffer(self) -> str:
        return self._current().ndbuffer

    # --- string API: get --------------------------------------------------
    @overload
    def get(self, key: Literal["default_zarr_format"]) -> Literal[2, 3]: ...
    @overload
    def get(self, key: Literal["array.order"]) -> Literal["C", "F"]: ...
    @overload
    def get(self, key: Literal["array.write_empty_chunks"]) -> bool: ...
    @overload
    def get(self, key: Literal["array.read_missing_chunks"]) -> bool: ...
    @overload
    def get(self, key: Literal["array.target_shard_size_bytes"]) -> int | None: ...
    @overload
    def get(self, key: Literal["array.rectilinear_chunks"]) -> bool: ...
    @overload
    def get(self, key: Literal["array.sharding_coalesce_max_gap_bytes"]) -> int: ...
    @overload
    def get(self, key: Literal["array.sharding_coalesce_max_bytes"]) -> int: ...
    @overload
    def get(self, key: Literal["async.concurrency"]) -> int: ...
    @overload
    def get(self, key: Literal["async.timeout"]) -> float | None: ...
    @overload
    def get(self, key: Literal["threading.max_workers"]) -> int | None: ...
    @overload
    def get(self, key: Literal["json_indent"]) -> int: ...
    @overload
    def get(self, key: Literal["codec_pipeline.path"]) -> str: ...
    @overload
    def get(self, key: Literal["codec_pipeline.batch_size"]) -> int: ...
    @overload
    def get(self, key: Literal["buffer"]) -> str: ...
    @overload
    def get(self, key: Literal["ndbuffer"]) -> str: ...
    @overload
    def get(self, key: str, default: Any = ...) -> Any: ...

    def get(self, key: str, default: Any = _MISSING) -> Any:
        resolved = self._apply_deprecation(key)
        if resolved is None:
            if default is _MISSING:
                raise KeyError(key)
            return default
        try:
            return get_path(self._current(), resolved)
        except KeyError:
            if default is _MISSING:
                raise
            return default

    # --- string API: set --------------------------------------------------
    def set(self, updates: Mapping[str, Any]) -> _ConfigSet:
        prev_base = self._base
        new = self._current()
        for key, value in updates.items():
            resolved = self._apply_deprecation(key)
            if resolved is None:
                continue
            new = replace_path(new, resolved, value)
        self._base = new
        token = self._scope.set(new)
        return _ConfigSet(self, prev_base, token)

    # --- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        self._base = build_config()
        with contextlib.suppress(LookupError):
            self._scope.set(self._base)

    def refresh(self) -> None:
        self._base = build_config()

    def enable_gpu(self) -> _ConfigSet:
        return self.set(
            {"buffer": "zarr.buffer.gpu.Buffer", "ndbuffer": "zarr.buffer.gpu.NDBuffer"}
        )

    # --- compat / introspection ------------------------------------------
    @property
    def defaults(self) -> dict[str, Any]:
        return to_nested_dict(make_default_config())

    def to_dict(self) -> dict[str, Any]:
        return to_nested_dict(self._current())

    def update(self, updates: Mapping[str, Any]) -> None:
        self.set(updates)

    def pprint(self) -> None:
        import pprint as _pp

        _pp.pprint(self.to_dict())

    # --- deprecations -----------------------------------------------------
    def _apply_deprecation(self, key: str) -> str | None:
        if key not in deprecations:
            return key
        new_key = deprecations[key]
        if new_key is None:
            warnings.warn(
                f"Configuration key {key!r} has been removed and no longer has "
                f"any effect.",
                ZarrDeprecationWarning,
                stacklevel=3,
            )
            return None
        warnings.warn(
            f"Configuration key {key!r} has been renamed to {new_key!r}.",
            ZarrDeprecationWarning,
            stacklevel=3,
        )
        return new_key
```

Add `from zarr.errors import ZarrDeprecationWarning` to the imports — this class already exists in `src/zarr/errors.py` (a `DeprecationWarning` subclass). Define the module-level instance at the bottom of the schema/proxy section but BEFORE the existing donfig `config = Config(...)` (which Task 4 removes):

```python
# Provisional new instance; Task 4 makes this THE module-level `config`.
_typed_config = ZarrConfigManager()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_typed.py -k "proxy or set_permanent or worker or defaults_and" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zarr/core/config.py tests/test_config_typed.py
git commit -m "feat(config): add typed proxy with get/set/reset and deprecations"
```

---

### Task 4: Swap out donfig (make `config` the new proxy)

**Files:**
- Modify: `src/zarr/core/config.py` (remove donfig `Config` subclass and instance; promote proxy)
- Test: existing `tests/test_config.py` (and the full suite)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: module-level `config: ZarrConfigManager`; unchanged exports `BadConfigError`, `parse_indexing_order`.

- [ ] **Step 1: Update the existing `defaults` assertion test**

In `tests/test_config.py::test_config_defaults_set`, replace the `config.defaults == [ {...} ]` list-of-one-dict assertion with the new nested-dict form:

```python
def test_config_defaults_set() -> None:
    assert config.defaults == {
        "default_zarr_format": 3,
        "array": {
            "order": "C",
            "write_empty_chunks": False,
            "read_missing_chunks": True,
            "target_shard_size_bytes": None,
            "rectilinear_chunks": False,
            "sharding_coalesce_max_gap_bytes": 1 << 20,
            "sharding_coalesce_max_bytes": 16 << 20,
        },
        "async": {"concurrency": 10, "timeout": None},
        "threading": {"max_workers": None},
        "json_indent": 2,
        "codec_pipeline": {
            "path": "zarr.core.codec_pipeline.BatchedCodecPipeline",
            "batch_size": 1,
        },
        "codecs": dict(DEFAULT_CODECS),
        "buffer": "zarr.buffer.cpu.Buffer",
        "ndbuffer": "zarr.buffer.cpu.NDBuffer",
    }
    assert config.get("array.order") == "C"
    assert config.get("async.concurrency") == 10
    assert config.get("async.timeout") is None
    assert config.get("codec_pipeline.batch_size") == 1
    assert config.get("json_indent") == 2
```

Add `from zarr.core.config import DEFAULT_CONFIG` is not needed; import `DEFAULT_CODECS` in the test's existing config import line.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config.py::test_config_defaults_set -v`
Expected: FAIL — `config.defaults` is still donfig's list form.

- [ ] **Step 3: Remove donfig and promote the proxy**

In `src/zarr/core/config.py`:
1. Delete the `from donfig import Config as DConfig` import and the `if TYPE_CHECKING: from donfig.config_obj import ConfigSet` block.
2. Delete the `class Config(DConfig): ...` definition (its `reset`/`enable_gpu` now live on `ZarrConfigManager`).
3. Delete the `config = Config("zarr", defaults=[...], deprecations=deprecations)` block. The big defaults dict is now expressed by the dataclasses + `DEFAULT_CODECS`; keep the `deprecations` dict (it is consumed by `ZarrConfigManager`).
4. Replace the provisional `_typed_config = ZarrConfigManager()` line with:

```python
config = ZarrConfigManager()
```

5. Update the module docstring at the top: replace donfig references with a description of the typed config and the `ZARR_FOO__BAR` env-var behavior (keep the example showing `config.set({"codecs.bytes": "your.module.NewBytesCodec"})` and the `ZARR_CODECS__BYTES` env var — both still work).
6. Keep `parse_indexing_order` and `BadConfigError` exactly as-is.

- [ ] **Step 4: Run the full config + dependent suites**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS.

Run: `uv run pytest tests/test_api.py tests/test_buffer.py tests/test_codec_entrypoints.py tests/test_v2.py tests/test_sync.py tests/test_common.py -q`
Expected: PASS (these import/use `config`).

Run: `uv run pytest tests -q`
Expected: PASS (full suite; backwards-compat gate).

- [ ] **Step 5: Run mypy**

Run: `uv run mypy src/zarr/core/config.py src/zarr/registry.py src/zarr/core/sync.py`
Expected: no errors. Confirm `reveal_type` is not needed here; fix any typing fallout in consumers (e.g. casts that referenced donfig types).

- [ ] **Step 6: Commit**

```bash
git add src/zarr/core/config.py tests/test_config.py
git commit -m "feat(config): replace donfig with typed config object"
```

---

### Task 5: Remove the donfig dependency

**Files:**
- Modify: `pyproject.toml` (lines ~39, ~246, ~272), `src/zarr/__init__.py` (~line 71)
- Test: import smoke test

**Interfaces:** none new.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config_typed.py`:

```python
def test_donfig_not_imported() -> None:
    import sys

    import zarr  # noqa: F401

    assert "donfig" not in sys.modules
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config_typed.py::test_donfig_not_imported -v`
Expected: FAIL — donfig still imported somewhere / installed and pulled in.

- [ ] **Step 3: Edit dependency declarations**

In `pyproject.toml`:
- Remove `'donfig>=0.8',` from the `dependencies` list (~line 39).
- Add `'pyyaml',` to the `dependencies` list (donfig previously pulled YAML support transitively; we now use it directly).
- Remove `'donfig @ git+https://github.com/pytroll/donfig',` from the `dynamic`/upstream group (~line 246).
- Remove `'donfig==0.8.*',` from the minimal-pins group (~line 272).

In `src/zarr/__init__.py`, remove `"donfig",` from the `required` list (~line 71).

- [ ] **Step 4: Re-sync the environment and verify**

Run: `uv run --reinstall-package zarr pytest tests/test_config_typed.py::test_donfig_not_imported -v`
Expected: PASS.

Run: `uv run python -c "import zarr; print(zarr.config.get('array.order'))"`
Expected: prints `C`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/zarr/__init__.py tests/test_config_typed.py
git commit -m "build: drop donfig dependency, add pyyaml"
```

---

### Task 6: Drift-protection, typing assertions, docs, changelog

**Files:**
- Test: `tests/test_config_typed.py`
- Modify: `src/zarr/core/config.py` docstring (if not already done in Task 4)
- Create: `changes/<pr>.misc.md`

**Interfaces:** none new.

- [ ] **Step 1: Write the drift-protection + typing tests**

Add to `tests/test_config_typed.py`:

```python
import typing

from zarr.core.config import ZarrConfig, ZarrConfigManager, _SERIALIZED_NAMES


def _structured_leaf_keys(cfg_cls: type, prefix: str = "") -> list[str]:
    import dataclasses

    keys: list[str] = []
    for f in dataclasses.fields(cfg_cls):
        serialized = _SERIALIZED_NAMES.get(f.name, f.name)
        key = f"{prefix}{serialized}" if not prefix else f"{prefix}.{serialized}"
        ftype = f.type
        if dataclasses.is_dataclass(ftype):
            keys.extend(_structured_leaf_keys(ftype, key))
        elif f.name == "codecs":
            continue  # open mapping, intentionally not enumerated
        else:
            keys.append(key)
    return keys


def test_every_structured_key_has_a_get_overload() -> None:
    overloads = typing.get_overloads(ZarrConfigManager.get)
    literal_keys: set[str] = set()
    for ov in overloads:
        hints = typing.get_type_hints(ov)
        key_hint = hints.get("key")
        if typing.get_origin(key_hint) is typing.Literal:
            literal_keys.update(typing.get_args(key_hint))
    missing = set(_structured_leaf_keys(ZarrConfig)) - literal_keys
    assert not missing, f"get() overloads missing for: {sorted(missing)}"


if typing.TYPE_CHECKING:

    def _typing_smoke(cfg: ZarrConfigManager) -> None:
        typing.assert_type(cfg.get("array.order"), typing.Literal["C", "F"])
        typing.assert_type(cfg.array.order, typing.Literal["C", "F"])
        typing.assert_type(cfg.get("async.concurrency"), int)
```

Note: `f.type` may be a string under `from __future__ import annotations`. If so, resolve with `typing.get_type_hints(cfg_cls)` inside `_structured_leaf_keys` instead of reading `f.type` directly. Adjust the helper accordingly so dataclass detection works on resolved types.

- [ ] **Step 2: Run to verify it fails (then passes once overloads complete)**

Run: `uv run pytest tests/test_config_typed.py -k "overload" -v`
Expected: PASS if all overloads from Task 3 are present; if it lists missing keys, add the corresponding `get` overloads in `config.py` and re-run until PASS.

- [ ] **Step 3: Type-check the typing smoke test**

Run: `uv run mypy tests/test_config_typed.py`
Expected: no errors (`assert_type` calls confirm the precise static types).

- [ ] **Step 4: Add the changelog entry**

Create `changes/<pr>.misc.md` (replace `<pr>` with the PR number) with:

```markdown
Replaced the ``donfig``-based configuration with a statically-typed
configuration object. ``zarr.config`` now provides precise static types for
attribute access (``zarr.config.array.order``) and for the dotted-string API
(``zarr.config.get("array.order")``). The string API, environment-variable
ingestion (``ZARR_FOO__BAR``), YAML config files, ``config.set`` (permanent and
as a context manager), ``config.reset``, ``config.enable_gpu``, and the
``deprecations`` mechanism are all preserved. The ``donfig`` dependency has been
removed.
```

- [ ] **Step 5: Update the module docstring (if not done in Task 4)**

Confirm `src/zarr/core/config.py`'s top docstring no longer references donfig and documents the typed API + `ZARR_*` env vars + YAML. (Use single-backtick markdown — docs are mkdocs.)

- [ ] **Step 6: Full verification + commit**

Run: `uv run pytest tests/test_config.py tests/test_config_typed.py -q`
Expected: PASS.

Run: `uv run pytest tests -q`
Expected: PASS.

Run: `uv run mypy src tests/test_config_typed.py`
Expected: no errors.

```bash
git add tests/test_config_typed.py src/zarr/core/config.py changes/
git commit -m "test(config): drift-protection + typing assertions; docs + changelog"
```

---

## Self-Review

**Spec coverage:**
- Schema dataclasses → Task 1. Open `codecs` mapping → Task 1 (`get_path`/`replace_path` mapping handling) + tests. State holder (base + contextvar) → Task 3. Proxy + typed attribute access → Task 3. Hand-written overloads → Task 3, completeness enforced Task 6. Env + YAML ingest → Task 2. Deprecations → Task 3. Backwards-compat surface → Task 4 (full suite) + preserved methods (`to_dict`/`update`/`pprint`/`refresh`/`reset`/`enable_gpu`/`defaults`). donfig removal → Task 5. Drift protection + typing assertions + changelog + docs → Task 6.
- `async_` alias rationale → realized via `_FIELD_ALIASES`/`_SERIALIZED_NAMES` in Tasks 1/3/6.

**Type consistency:** `ZarrConfig`, `ArraySettings`, `AsyncSettings`, `ThreadingSettings`, `CodecPipelineSettings`, `get_path`, `replace_path`, `to_nested_dict`, `build_config`, `collect_env`, `collect_yaml`, `apply_overrides`, `ZarrConfigManager`, `_ConfigSet`, `_FIELD_ALIASES`, `_SERIALIZED_NAMES`, `DEFAULT_CODECS` are used consistently across tasks.

**Known follow-ups for the implementer (not placeholders — explicit decisions):**
- If `from __future__ import annotations` makes `dataclasses.fields(...).type` a string, resolve via `get_type_hints` in the drift helper (Task 6, Step 1).
- `set` semantics note: a top-level `config.set({...})` updates `_base` (cross-thread, permanent) and the contextvar scope; `with config.set({...})` restores both on exit. This matches donfig's permanent-by-default behavior while keeping cross-thread visibility (verified by `test_permanent_set_visible_in_worker_thread`).
