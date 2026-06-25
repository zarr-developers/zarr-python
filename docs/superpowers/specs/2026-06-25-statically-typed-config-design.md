# Statically-typed configuration for zarr-python

**Date:** 2026-06-25
**Status:** Approved design, ready for implementation planning

## Problem

zarr-python's configuration is built on [donfig](https://github.com/pytroll/donfig).
donfig stores config as an untyped nested `dict`, so there is no static type
information for any configuration value. `config.get("array.order")` is typed as
`Any`, `config.array` does not exist as a typed attribute, and there is no way for
a type checker to catch a misspelled key or a wrong-typed value.

We want to drop donfig entirely and model the configuration as plain frozen
dataclasses, which gives native static typing for attribute access
(`config.array.order`), while retaining donfig's ergonomic dotted-string API
(`config.get("array.order")`, `config.set({"array.order": "F"})`) with precise
static types via hand-written overloads. This is the technique demonstrated in the
[`tytr`](https://github.com/d-v-b/tytr) project: a flattened mapping from dotted
keys to value types, surfaced through an overloaded getter/setter.

## Non-negotiable constraint: backwards compatibility

**Backwards compatibility is extremely important for this work.** The public
`zarr.config` object is widely used in downstream code, notebooks, and
documentation. The replacement MUST be a drop-in for every documented and
commonly-used pattern. Concretely:

- All of these must continue to work with identical behavior and (where they
  returned values) identical return values:
  - `config.get("a.b.c")` and `config.get("a.b.c", default)`
  - subtree retrieval: `config.get("codecs", {}).get(key)`
  - `config.set({"a.b.c": value})` applied **permanently**
  - `with config.set({"a.b.c": value}):` applied **scoped**, restored on exit
  - `config.reset()`
  - `config.enable_gpu()`
  - `config.defaults`
  - `BadConfigError`
  - the `ZARR_FOO__BAR` environment-variable ingestion
  - YAML config-file ingestion from standard locations
  - the `deprecations` key-redirection/removal warnings
- Public import paths are unchanged: `from zarr.core.config import config,
  BadConfigError, parse_indexing_order` and `zarr.config`.
- donfig provides a broader method surface (`to_dict`, `update`, `merge`,
  `pprint`, `clear`, `refresh`, `collect`, ...). We preserve the subset zarr
  itself uses (`get`, `set`, `reset`, `enable_gpu`, `defaults`, `clear`,
  `refresh`) and additionally provide compatible shims for `to_dict`/`update`/
  `pprint` since these are plausible downstream uses. Any donfig method we do not
  reimplement must raise a clear, actionable error pointing at the new API rather
  than an `AttributeError`.
- Behavior changes are only acceptable where they are strictly additive (new
  precise types) or where donfig behavior was undocumented/incidental. Any
  observable change is called out in the changelog with migration guidance.
- A `towncrier` changelog entry under `changes/` documents the donfig removal and
  confirms the API is preserved.

## Architecture

Three layers with clear boundaries.

### Layer A — schema (frozen dataclasses)

The configuration shape is a tree of frozen, slotted dataclasses. This is the
single source of truth for both structure and defaults.

> **Naming note:** a distinct `ArrayConfig` already exists in
> `src/zarr/core/array_spec.py` (a runtime per-array object, unrelated to the
> global config). To avoid collision and confusion, the global-config schema
> dataclasses are named with a `Config` suffix scoped under the config module
> (e.g. the array-namespace schema below). If the names below would still read
> ambiguously next to the existing `ArrayConfig`, prefer an explicit suffix such
> as `ArraySettings` / `ZarrSettings` during implementation. The final names are
> an implementation detail; the structure is what matters.

```python
@dataclass(frozen=True, slots=True)
class ArrayConfig:
    order: Literal["C", "F"] = "C"
    write_empty_chunks: bool = False
    read_missing_chunks: bool = True
    target_shard_size_bytes: int | None = None
    rectilinear_chunks: bool = False
    sharding_coalesce_max_gap_bytes: int = 1 << 20      # 1 MiB
    sharding_coalesce_max_bytes: int = 16 << 20         # 16 MiB

@dataclass(frozen=True, slots=True)
class AsyncConfig:
    concurrency: int = 10
    timeout: float | None = None

@dataclass(frozen=True, slots=True)
class ThreadingConfig:
    max_workers: int | None = None

@dataclass(frozen=True, slots=True)
class CodecPipelineConfig:
    path: str = "zarr.core.codec_pipeline.BatchedCodecPipeline"
    batch_size: int = 1

@dataclass(frozen=True, slots=True)
class ZarrConfig:
    default_zarr_format: Literal[2, 3] = 3
    array: ArrayConfig = field(default_factory=ArrayConfig)
    async_: AsyncConfig = field(default_factory=AsyncConfig)   # serialized key: "async"
    threading: ThreadingConfig = field(default_factory=ThreadingConfig)
    json_indent: int = 2
    codec_pipeline: CodecPipelineConfig = field(default_factory=CodecPipelineConfig)
    codecs: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_CODECS))
    buffer: str = "zarr.buffer.cpu.Buffer"
    ndbuffer: str = "zarr.buffer.cpu.NDBuffer"
```

Notes:
- `config.array.order` etc. are natively typed by the dataclass — no overloads
  needed for the attribute-access path.
- `async_` carries the serialized key `"async"` (an illegal Python identifier).
  The mapping between Python field name and serialized dotted key is recorded in a
  small per-class `__key_aliases__` (or equivalent) so the string API and ingest
  layers translate correctly. Attribute access for `async` is only available via
  the string API (`config.get("async.concurrency")`); this matches donfig, which
  also has no `config.async` attribute.
- `codecs` is an open `Mapping[str, str]` subtree (per design decision): users
  register arbitrary codec names at runtime via `config.set({"codecs.foo": ...})`
  and `ZARR_CODECS__FOO=...`. Structured keys get precise static types; codec keys
  degrade to the string fallback. `DEFAULT_CODECS` holds the current default codec
  name → import-path mapping verbatim.

### Layer B — state holder (base snapshot + contextvar overlay)

State is held as immutable `ZarrConfig` snapshots. To preserve donfig's exact
runtime semantics — in particular cross-thread visibility of permanent sets — we
use a **hybrid** of a process-global base and a context-local overlay rather than a
pure `ContextVar`.

Rationale: zarr runs work in `ThreadPoolExecutor` (`src/zarr/core/sync.py`).
`ThreadPoolExecutor` does **not** copy `contextvars` into worker threads. A pure
`ContextVar` would make a permanent `config.set({...})` invisible inside worker
threads — a silent regression versus donfig's process-global dict mutation. The
hybrid avoids this.

- `_base: ZarrConfig` — a module-global snapshot, process-wide, visible across all
  threads. A **permanent** `config.set(...)` (not used as a `with` block) replaces
  this reference.
- `_overlay: ContextVar[ZarrConfig | None]` — a context-local override. `with
  config.set(...)` sets this and resets it via the returned `Token` on exit.
  Provides async-safe and thread-safe scoping for the common `with config.set(...)`
  idiom.
- Resolution: the effective snapshot is `_overlay.get() or _base`.
- Every mutation produces a **new** frozen `ZarrConfig` by applying the requested
  dotted-key updates through `dataclasses.replace` along the path (a small
  recursive `replace_path(snapshot, "a.b.c", value) -> ZarrConfig` helper). For the
  open `codecs` mapping, updates copy-and-extend the dict.

`config.set(...)` semantics, matching donfig:
- Applies immediately (mutates effective state) **and** returns a context-manager
  object.
- If used as `with config.set(...):`, the prior state is restored on `__exit__`.
- If not used as a context manager, the change persists (permanent set updates
  `_base`).

### Layer C — proxy (`config`)

`config` is the shared singleton everyone imports. It is **not** the data; it reads
the current resolved snapshot on each access, so existing `from zarr.core.config
import config` references continue to observe live updates (preserving donfig's
import-by-reference behavior). It exposes:

- Typed attribute properties delegating to the resolved snapshot: `config.array ->
  ArrayConfig`, `config.async_ -> AsyncConfig`, `config.json_indent -> int`, etc.
- The donfig-compatible string API: `get`, `set`, `reset`, `enable_gpu`,
  `defaults`, plus compat shims (`to_dict`, `update`, `pprint`).

## The typed string API (hand-written overloads)

Per the design decision, the dotted-key → value-type overloads are **hand-written**
(no codegen, no `tytr` runtime dependency). This is the `tytr` getter pattern,
authored directly:

```python
class _ConfigProxy:
    @overload
    def get(self, key: Literal["default_zarr_format"]) -> Literal[2, 3]: ...
    @overload
    def get(self, key: Literal["array.order"]) -> Literal["C", "F"]: ...
    @overload
    def get(self, key: Literal["array.write_empty_chunks"]) -> bool: ...
    @overload
    def get(self, key: Literal["async.concurrency"]) -> int: ...
    @overload
    def get(self, key: Literal["async.timeout"]) -> float | None: ...
    @overload
    def get(self, key: Literal["json_indent"]) -> int: ...
    # ... one overload per structured leaf key ...
    @overload
    def get(self, key: str, default: object = ...) -> Any: ...   # codecs.*, subtrees, unknown keys
    def get(self, key: str, default: object = _MISSING) -> Any: ...
```

`set` mirrors this: an overloaded surface (or a `TypedDict` of optional dotted
keys) so that `config.set({"array.order": "F"})` type-checks the value against the
key. The open `codecs.*` keys and whole-subtree gets (`config.get("codecs", {})`)
resolve through the `str` fallback overload.

### Drift protection

Hand-written overloads can drift from the dataclass schema. A regression test walks
`ZarrConfig` recursively, enumerates every structured dotted leaf key, and asserts
each has a corresponding `get` overload with a matching return type (introspected
via `typing.get_overloads`). CI fails on any missing/mismatched overload. This
neutralizes the main downside of the hand-written approach.

## Ingest sources

Both retained (per design decision). Reimplemented in zarr (~a few dozen lines)
rather than vendoring donfig's loader.

Precedence, lowest to highest:

1. dataclass defaults
2. YAML config files
3. environment variables
4. runtime `config.set(...)`

- **Environment variables:** collect `ZARR_*`, lower-case the key, treat `__` as
  nested access, `ast.literal_eval` the value (with literal-eval failure falling
  back to the raw string, matching donfig). Builds overrides merged into the base
  snapshot at construction.
- **YAML files:** read from standard locations — `ZARR_CONFIG` env var path(s) and
  the default config directory (e.g. `~/.config/zarr`), matching donfig's search
  behavior. Parsed with the existing YAML dependency and merged under env vars.

Ingested values are validated/coerced into the dataclass field types where the key
is structured; unknown keys under open subtrees (`codecs.*`) pass through as
strings.

## Deprecations

donfig's `deprecations` mechanism (old-key → new-key, or `None` for removed) is
reimplemented. Accessing or setting a deprecated key emits the same warning and
redirects to the new key (or raises/warns for removed keys). The existing
`deprecations` mapping in `config.py` is carried over verbatim:

```python
deprecations = {
    "array.v2_default_compressor.numeric": None,
    # ... unchanged ...
}
```

## Backwards-compatibility verification

Beyond the per-feature preservation above:

- A compatibility test module exercises every pattern in the "Non-negotiable
  constraint" list against the new implementation.
- `config.defaults` returns a representation equivalent to today's (the existing
  `test_config_defaults_set` is updated to the new snapshot representation while
  asserting the same values).
- Methods not reimplemented raise an informative error naming the supported
  replacement, never a bare `AttributeError`.

## Testing

- Existing `tests/test_config.py` remains largely valid since the string API is
  preserved; only `config.defaults` structural assertions are updated.
- New tests:
  - overload ↔ dataclass sync (drift protection).
  - env-var ingestion (including `ZARR_CODECS__*` dynamic keys).
  - YAML-file ingestion and precedence ordering.
  - permanent-set visibility inside a `ThreadPoolExecutor` worker (the hybrid
    state-model regression).
  - `with config.set(...)` scoping under threads and asyncio tasks.
  - deprecation warnings/redirects.
- Static-typing assertions in the test suite (the repo type-checks tests):
  `reveal_type(config.get("array.order"))` is `Literal['C', 'F']`,
  `reveal_type(config.array.order)` is `Literal['C', 'F']`, and a wrong-typed
  `config.set({"array.order": "Q"})` is a type error.

## Files affected

- `src/zarr/core/config.py` — rewritten: dataclasses, proxy, state holder, string
  API, ingest, deprecations. Keeps `config`, `BadConfigError`,
  `parse_indexing_order` exports.
- `pyproject.toml` — remove `donfig` dependency; ensure a YAML dependency is
  declared (currently transitive via donfig).
- `src/zarr/__init__.py` — remove `donfig` from the version-reporting table.
- `tests/test_config.py` and new test modules — as above.
- `changes/<issue>.misc.md` (or `.feature.md`) — changelog entry.
- Documentation referencing donfig (`config.py` module docstring, any docs/ pages)
  — updated to describe the new typed API while keeping the string-API examples.

## Out of scope

- Changing the set of configuration keys or their defaults.
- Migrating the `codecs` registry out of config (the open `dict[str, str]` subtree
  is retained).
- Any change to the `ArrayConfig`/`ArraySpec` runtime objects in
  `core/array_spec.py` beyond what is needed to read from the new config.
