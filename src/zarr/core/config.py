"""
Typed configuration for zarr.

The module exposes a single `config` object (a `ZarrConfigManager` instance) that
holds all runtime settings.  Values can be read, overridden, and restored through a
simple string-key API:

- `config.get(key)` — read a dotted-key value (e.g. `config.get("async.concurrency")`).
- `config.set({key: value})` — permanent override; also usable as a context manager to
  restore the previous state on exit.
- `config.reset()` — rebuild from defaults + environment.
- `config.refresh()` — alias for `reset`; called by the registry after env changes.
- `config.defaults` — nested dict of built-in default values.
- `config.enable_gpu()` — switch buffer/ndbuffer to GPU implementations.

Environment variables use the `ZARR_` prefix and `__` for nesting:

```bash
export ZARR_CODECS__BYTES="your.module.NewBytesCodec"
```

Programmatic override:

```python
from your.module import NewBytesCodec
from zarr.core.config import config

config.set({"codecs.bytes": "your.module.NewBytesCodec"})
```

For selecting custom implementations of codecs, pipelines, buffers, and ndbuffers,
register the implementation in the registry first, then set the path via `config.set`.
"""

from __future__ import annotations

import difflib
import threading
import warnings
from collections.abc import Iterator, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any, Literal, Self, cast, overload

from zarr.errors import ZarrDeprecationWarning, ZarrUserWarning

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


class _ConfigNode:
    """Mixin giving the frozen config dataclasses donfig-style item access.

    donfig returned configuration subtrees as plain `dict`s, so callers could
    write `config.get("array")["order"]`.  The typed config returns dataclass
    instances instead; this mixin restores subscripting (`node["order"]`, and
    dotted `node["a.b"]`) alongside the typed attribute access (`node.order`),
    raising `KeyError` for unknown keys just like the old dicts did.

    `__slots__ = ()` keeps the subclasses fully slotted (no `__dict__`).
    """

    __slots__ = ()

    def __getitem__(self, key: str) -> object:
        # `self` is always a config node (a frozen dataclass); `get_path` walks
        # such nodes structurally, so the cast is sound.
        return get_path(cast("ZarrConfig", self), key)

    def __contains__(self, key: object) -> bool:
        # Mirror donfig's `"array" in config.get(...)` support. Only string keys
        # can resolve; a non-string (or unknown key) is simply not contained.
        if not isinstance(key, str):
            return False
        try:
            get_path(cast("ZarrConfig", self), key)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        # Yield immediate child key names (serialized), so iteration, `keys()`,
        # and `dict(node)` behave like the plain dicts donfig returned. Note this
        # deliberately does NOT make the node a `collections.abc.Mapping`: the
        # internal `isinstance(obj, Mapping)` checks must stay false for config
        # nodes so they are distinguished from the open `codecs` mapping.
        return iter(_children(self))

    def keys(self) -> list[str]:
        return _children(self)

    def __len__(self) -> int:
        return len(_children(self))


@dataclass(frozen=True, slots=True)
class ArraySettings(_ConfigNode):
    order: Literal["C", "F"] = "C"
    write_empty_chunks: bool = False
    read_missing_chunks: bool = True
    target_shard_size_bytes: int | None = None
    rectilinear_chunks: bool = False
    sharding_coalesce_max_gap_bytes: int = 1 << 20
    sharding_coalesce_max_bytes: int = 16 << 20


@dataclass(frozen=True, slots=True)
class AsyncSettings(_ConfigNode):
    concurrency: int = 10
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class ThreadingSettings(_ConfigNode):
    max_workers: int | None = None


@dataclass(frozen=True, slots=True)
class CodecPipelineSettings(_ConfigNode):
    path: str = "zarr.core.codec_pipeline.BatchedCodecPipeline"
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class ZarrConfig(_ConfigNode):
    default_zarr_format: Literal[2, 3] = 3
    array: ArraySettings = field(default_factory=ArraySettings)
    async_: AsyncSettings = field(default_factory=AsyncSettings)
    threading: ThreadingSettings = field(default_factory=ThreadingSettings)
    json_indent: int = 2
    codec_pipeline: CodecPipelineSettings = field(default_factory=CodecPipelineSettings)
    # A plain dict (not MappingProxyType) so snapshots stay picklable /
    # deep-copyable; public read access goes through the manager's `codecs`
    # property, which wraps this in a read-only view.
    codecs: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_CODECS))
    buffer: str = "zarr.buffer.cpu.Buffer"
    ndbuffer: str = "zarr.buffer.cpu.NDBuffer"


def make_default_config() -> ZarrConfig:
    """Return a fresh `ZarrConfig` populated with the built-in defaults."""
    return ZarrConfig()


def _resolve_field(obj: object, segment: str) -> str:
    """Translate a serialized key segment to the dataclass field name."""
    return _FIELD_ALIASES.get(segment, segment)


def get_path(cfg: ZarrConfig, key: str) -> object:
    """Read a dotted-string key from a `ZarrConfig` snapshot.

    Raises
    ------
    KeyError
        If the key does not resolve to a value.
    """
    obj: object = cfg
    segments = key.split(".")
    for i, segment in enumerate(segments):
        if isinstance(obj, Mapping):
            # remaining segments index into an open mapping (e.g. codecs.*)
            remainder = ".".join(segments[i:])
            try:
                return obj[remainder]
            except KeyError:
                raise KeyError(key) from None
        # A prior segment resolved to a scalar leaf, but the key has more
        # segments — descend no further. Without this guard, `hasattr` would
        # match ordinary Python attributes/methods (e.g. `array.order.upper`
        # returning `str.upper`, or `default_zarr_format.numerator` returning
        # the int's numerator) instead of raising for the invalid key.
        if not is_dataclass(obj):
            raise KeyError(key)
        field_name = _resolve_field(obj, segment)
        if field_name not in {f.name for f in fields(obj)}:
            raise KeyError(key)
        obj = getattr(obj, field_name)
    return obj


def replace_path(cfg: ZarrConfig, key: str, value: object) -> ZarrConfig:
    """Return a new `ZarrConfig` with the dotted-string key set to ``value``."""
    segments = key.split(".")
    return cast(ZarrConfig, _replace_recursive(cfg, segments, value, key))


# `obj: Any` is load-bearing here: the function dispatches dynamically between a
# `Mapping` (codecs subtree) and a dataclass instance, and `dataclasses.replace`
# requires a dataclass-typed argument that `object` would reject.
def _replace_recursive(obj: Any, segments: list[str], value: object, key: str) -> object:
    segment = segments[0]
    if isinstance(obj, Mapping):
        remainder = ".".join(segments)
        # Plain dict (see the `codecs` field note); the manager property wraps
        # it read-only for public access.
        return {**obj, remainder: value}
    if not is_dataclass(obj):
        # `key` tries to descend past a scalar leaf (e.g. `array.order.upper`).
        raise KeyError(key)
    field_name = _resolve_field(obj, segment)
    if field_name not in {f.name for f in fields(obj)}:
        raise KeyError(key)
    # `is_dataclass` narrows `obj` to `... | type[...]`, which `replace` rejects;
    # at runtime `obj` is always a dataclass *instance* here, so re-widen to Any.
    node: Any = obj
    child = getattr(node, field_name)
    if len(segments) == 1:
        # Refuse to overwrite a structured subtree (a nested dataclass) wholesale
        # — doing so would drop its sibling fields and break typed attribute
        # access. Set leaf keys instead. The open `codecs` mapping is not a
        # dataclass, so wholesale replacement there is still allowed.
        if is_dataclass(child):
            raise TypeError(
                f"Cannot assign to the structured config subtree {key!r} directly; "
                f"set leaf keys instead, e.g. config.set({{'{key}.<field>': ...}})."
            )
        return replace(node, **{field_name: value})
    new_child = _replace_recursive(child, segments[1:], value, key)
    return replace(node, **{field_name: new_child})


_ROSTER_LIMIT = 10


def _children(obj: object) -> list[str]:
    """Return the immediate child key names of a config node (else an empty list)."""
    if isinstance(obj, Mapping):
        return list(obj)
    if is_dataclass(obj):
        return [_SERIALIZED_NAMES.get(f.name, f.name) for f in fields(obj)]
    return []


def _resolve_for_suggestion(cfg: ZarrConfig, key: str) -> tuple[str, list[str], str]:
    """Walk ``key`` as far as it resolves.

    Returns the deepest resolvable dotted prefix, that node's child key names,
    and the first segment that failed to resolve (the remainder is treated as a
    single key once an open mapping like ``codecs`` is reached). For
    ``"array.bogus"`` this is ``("array", [<ArraySettings fields>], "bogus")``;
    for an unknown top-level key, ``("", [<top-level keys>], <key>)``.
    """
    obj: object = cfg
    prefix = ""
    segments = key.split(".")
    for i, segment in enumerate(segments):
        if isinstance(obj, Mapping):
            # the remainder indexes into an open mapping as a single key
            return prefix, _children(obj), ".".join(segments[i:])
        if not is_dataclass(obj):
            # a prior segment resolved to a scalar leaf; nothing deeper is valid
            return prefix, _children(obj), segment
        field_name = _resolve_field(obj, segment)
        if field_name not in {f.name for f in fields(obj)}:
            return prefix, _children(obj), segment
        obj = getattr(obj, field_name)
        prefix = f"{prefix}.{segment}" if prefix else segment
    return prefix, _children(obj), ""


def _unknown_key_error(key: str, cfg: ZarrConfig) -> KeyError:
    """Build a `KeyError` for an unknown config key.

    Resolves ``key`` to the deepest valid level, then suggests the closest child
    key there if one is similar enough; otherwise lists the available keys at
    that level (capped at `_ROSTER_LIMIT`).
    """
    msg = f"{key!r} is not a valid configuration key."
    prefix, children, failed = _resolve_for_suggestion(cfg, key)
    matches = difflib.get_close_matches(failed, children, n=1) if failed != "" else []
    if len(matches) > 0:
        suggestion = f"{prefix}.{matches[0]}" if prefix != "" else matches[0]
        return KeyError(f"{msg} Did you mean {suggestion!r}?")
    if len(children) > 0:
        shown = sorted(children)
        roster = ", ".join(shown[:_ROSTER_LIMIT])
        if len(shown) > _ROSTER_LIMIT:
            roster += f", ... ({len(shown) - _ROSTER_LIMIT} more)"
        where = f" under {prefix!r}" if prefix != "" else ""
        msg = f"{msg} Valid keys{where}: {roster}."
    return KeyError(msg)


def to_nested_dict(cfg: ZarrConfig) -> dict[str, Any]:
    """Convert a `ZarrConfig` to a donfig-style nested dict (serialized keys).

    Returns a heterogeneous, JSON-like tree (nested dicts and scalars) that
    callers navigate by key, so `Any` values are appropriate here.
    """

    # `obj: Any` is also load-bearing: `dataclasses.fields` requires a
    # dataclass-typed argument that `object` would reject.
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


def _flatten_mapping(data: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in data.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, Mapping):
            out.update(_flatten_mapping(v, key))
        else:
            out[key] = v
    return out


def apply_overrides(cfg: ZarrConfig, overrides: Mapping[str, object]) -> ZarrConfig:
    """Apply a flat dotted-key override map to a snapshot.

    Used exclusively by `build_config` for env/YAML ingest.  Unknown keys are
    skipped with a warning rather than raising, so a stray environment variable
    or extra YAML key never prevents `import zarr` from succeeding.
    """
    for key, value in overrides.items():
        try:
            cfg = replace_path(cfg, key, value)
        except KeyError:
            warnings.warn(
                f"Unrecognized zarr config key {key!r} from environment or YAML — ignoring.",
                ZarrUserWarning,
                stacklevel=2,
            )
    return cfg


# donfig's env collection also surfaces the `ZARR_CONFIG` / `ZARR_ROOT_CONFIG`
# path directives as if they were config values (keys `config` / `root_config`);
# drop them so they don't trip `apply_overrides`'s unknown-key warning.
_DONFIG_META_KEYS: frozenset[str] = frozenset({"config", "root_config"})


def _canonicalize_override_keys(overrides: Mapping[str, object]) -> dict[str, object]:
    """Map underscore codec names onto their hyphenated built-in defaults.

    Environment variables cannot contain hyphens, so ``ZARR_CODECS__VLEN_UTF8``
    flattens to the key ``codecs.vlen_utf8``. The built-in codec is registered
    under the hyphenated name ``vlen-utf8`` (likewise ``vlen-bytes``), so without
    this remapping the override would land under a dead ``vlen_utf8`` key and be
    silently ignored while the registry keeps reading the untouched default.
    When a ``codecs.<name>`` key does not match a default but its hyphenated
    variant does, rewrite it to the hyphenated form. New (non-default) codec
    names and underscore-named defaults (e.g. ``sharding_indexed``) are untouched.
    """
    out: dict[str, object] = {}
    for key, value in overrides.items():
        if key.startswith("codecs."):
            name = key[len("codecs.") :]
            if name not in DEFAULT_CODECS and name.replace("_", "-") in DEFAULT_CODECS:
                key = f"codecs.{name.replace('_', '-')}"
        out[key] = value
    return out


def build_config() -> ZarrConfig:
    """Build the base snapshot: typed defaults overlaid with donfig's ingest.

    `donfig` reads `ZARR_*` environment variables and YAML config files from its
    standard locations
    (https://donfig.readthedocs.io/en/latest/configuration.html#yaml-files) and
    merges them into a nested override mapping. That mapping is flattened to
    dotted keys and applied on top of the typed defaults. `donfig` owns discovery,
    parsing, and precedence; this module owns the typed representation. Unknown
    keys are warned about and skipped by `apply_overrides`, so a stray variable or
    a version-skewed config file never blocks `import zarr`.
    """
    import donfig

    overrides = _flatten_mapping(donfig.Config("zarr").config)
    overrides = {
        key: value
        for key, value in overrides.items()
        if key.split(".", 1)[0] not in _DONFIG_META_KEYS
    }
    overrides = _canonicalize_override_keys(overrides)
    return apply_overrides(make_default_config(), overrides)


_MISSING = object()


class _ConfigSet:
    """Context manager returned by ``ZarrConfigManager.set``.

    ``set`` applies the override immediately to the process-global base, so a
    bare ``config.set(...)`` is permanent and visible from every thread (matching
    donfig's last-writer-wins semantics, including inside `ThreadPoolExecutor`
    workers, which do not copy context variables).

    Using the result as a ``with`` block *promotes* the override to a
    context-local scope: ``__enter__`` undoes the global apply and re-applies the
    new snapshot through a `ContextVar`, so the change is isolated to the calling
    context (thread / async task) and unwound on ``__exit__``.

    A bare (permanent) ``set`` nested inside an active ``with`` block writes only
    its own delta onto the global base, so it persists after the block exits and
    does not leak the block's overlay into the base. The trade-off is that such a
    nested permanent ``set`` is not visible *within* the block (the overlay keeps
    shadowing the base until the block exits).

    Note: like donfig, config writes are not synchronized. The promotion restores
    the base snapshot captured at ``set`` time, so a ``with config.set(...)`` that
    overlaps a *concurrent* permanent ``set`` from another thread may drop that
    concurrent write for the duration of the block. Configuration is normally set
    at startup, so this race does not arise in practice.
    """

    def __init__(self, manager: ZarrConfigManager, prev_base: ZarrConfig, new: ZarrConfig) -> None:
        self._manager = manager
        self._prev_base = prev_base
        self._new = new
        self._token: Token[ZarrConfig] | None = None

    def __enter__(self) -> Self:
        self._token = self._manager._enter_scope(self._prev_base, self._new)
        return self

    def __exit__(self, *exc: object) -> None:
        # `__enter__` always runs first under `with`, so `_token` is set; guard
        # only to satisfy the type checker and tolerate manual misuse.
        if self._token is not None:
            self._manager._exit_scope(self._token)


class ZarrConfigManager:
    """Typed, donfig-compatible configuration object."""

    def __init__(self) -> None:
        self._base: ZarrConfig = build_config()
        self._scope: ContextVar[ZarrConfig] = ContextVar("zarr_config_scope")
        # Serializes read-modify-write of the process-global `_base` so
        # concurrent permanent `set`s to different keys don't lose updates
        # (each `set` rebuilds a whole immutable snapshot from `_base`).
        self._lock = threading.Lock()

    # --- state resolution -------------------------------------------------
    def _current(self) -> ZarrConfig:
        # An active `with config.set(...)` overlay (context-local) shadows the
        # global base; otherwise every context reads the shared `_base` live, so
        # a permanent `set` in any thread is visible everywhere.
        return self._scope.get(self._base)

    def _enter_scope(self, prev_base: ZarrConfig, new: ZarrConfig) -> Token[ZarrConfig]:
        # Promote a `with config.set(...)` to a context-local override: undo the
        # immediate global apply performed by `set`, then pin `new` to this
        # context only. Concurrent threads/tasks keep seeing the global base.
        with self._lock:
            self._base = prev_base
        return self._scope.set(new)

    def _exit_scope(self, token: Token[ZarrConfig]) -> None:
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
        # Read-only view: the underlying snapshot field is a plain dict (kept
        # picklable), but callers must not mutate a live snapshot in place.
        return MappingProxyType(self._current().codecs)

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
    # The fallback `-> Any` is deliberate: it lets `config.get("codecs", {})` be
    # used as a mapping (e.g. `.get(name)` in the registry) and supports unknown
    # keys. `object` here would force every such call site to narrow first.
    def get(self, key: str, default: object = ...) -> Any: ...

    def get(self, key: str, default: object = _MISSING) -> Any:
        resolved = self._apply_deprecation(key, raise_on_removed=False)
        if resolved is None:
            # Key was removed; treat as absent — honour the caller's default.
            if default is _MISSING:
                raise KeyError(key)
            return default
        current = self._current()
        try:
            return get_path(current, resolved)
        except KeyError:
            if default is _MISSING:
                raise _unknown_key_error(key, current) from None
            return default

    # --- string API: set --------------------------------------------------
    #
    # NOTE: `set` accepts `Mapping[str, Any]`, so — unlike `get`, which is fully
    # typed via per-key overloads — it does NOT statically validate values:
    # `config.set({"array.order": "Q"})` is not a type error; it is caught at
    # runtime instead. This is a deliberate, documented limitation.
    #
    # Static value typing would require an *open* TypedDict — declared structured
    # keys validated by type, PLUS arbitrary `codecs.<name>` string keys allowed
    # (PEP 728 `extra_items`/`closed`). mypy (2.x) supports PEP 728 in no syntax
    # and offers no feature flag for it. A *closed* TypedDict would instead reject
    # the open codec-selection idiom
    # `config.set({"codecs.bytes": "your.module.NewBytesCodec"})` and any
    # dynamically built `dict[str, Any]` — a backwards-compatibility regression
    # (the `codecs` namespace maps a codec name to a class path and is extended at
    # runtime by users/plugins, so its keys cannot be enumerated statically).
    # So `set` is intentionally permissive and validated at runtime: unknown
    # structured keys raise (see `replace_path`), while `codecs.*` stays writable.
    #
    # REVISIT when mypy ships PEP 728 open-TypedDict support, or if zarr adopts a
    # type checker that supports it (e.g. pyright's open/closed TypedDicts). At
    # that point `set` can take an open TypedDict for static value validation
    # while keeping `codecs.*` open.
    def set(self, updates: Mapping[str, object] | None = None, **kwargs: object) -> _ConfigSet:
        """Apply one or more config overrides.

        Accepts either a mapping of dotted keys to values, keyword arguments
        (for top-level keys), or both::

            config.set({"array.order": "F"})
            config.set(default_zarr_format=2)

        `set` validates *keys* — an unknown key raises with a suggestion — but
        does **not** validate *values*: `config.set({"array.order": "Q"})` is
        accepted, and the invalid value surfaces later at its use site rather
        than here. Static value typing is prevented by the open `codecs.*`
        namespace (see the implementation comment above); runtime value
        validation is planned via the unified `parse_json` checker (gh-3285).
        """
        all_updates: dict[str, object] = {}
        if updates:
            all_updates.update(updates)
        all_updates.update(kwargs)
        # Hold the lock across the whole read-modify-write of `_base` so two
        # concurrent permanent `set`s to different keys can't clobber each other
        # (each rebuilds a full snapshot from the base it read).
        with self._lock:
            prev_base = self._base
            # Two snapshots so an override applies to the right layer:
            # - `scoped` layers on the current view (any active `with` overlay),
            #   and is what a `with config.set(...)` pins as its context-local
            #   scope;
            # - `permanent` layers on the *global* base, and is what a bare `set`
            #   writes. Basing the permanent write on `prev_base` rather than the
            #   current view keeps a bare `set` nested inside a `with` block from
            #   leaking that block's overlay into the global base.
            # Outside any `with` block the two are identical (`_current()` is
            # `_base`).
            scoped = self._current()
            permanent = prev_base
            for key, value in all_updates.items():
                resolved = self._apply_deprecation(key, raise_on_removed=True)
                try:
                    scoped = replace_path(scoped, resolved, value)
                    permanent = replace_path(permanent, resolved, value)
                except KeyError:
                    raise _unknown_key_error(key, permanent) from None
            # Apply immediately to the global base. A bare `set` stays here
            # (permanent, cross-thread); a `with config.set(...)` is promoted to a
            # context-local overlay by `_ConfigSet.__enter__`, which undoes this
            # global apply and pins `scoped` instead.
            self._base = permanent
        return _ConfigSet(self, prev_base, scoped)

    # --- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        # Rebuild the global base. A bare `set` no longer pins a context-local
        # scope, so the rebuilt base is visible in every context that is not
        # inside an active `with config.set(...)` block. Build outside the lock
        # (it reads env/YAML) and swap atomically under it.
        new_base = build_config()
        with self._lock:
            self._base = new_base

    def refresh(self) -> None:
        self.reset()

    def enable_gpu(self) -> _ConfigSet:
        return self.set(
            {"buffer": "zarr.buffer.gpu.Buffer", "ndbuffer": "zarr.buffer.gpu.NDBuffer"}
        )

    # --- compat / introspection ------------------------------------------
    def __getitem__(self, key: str) -> Any:
        # donfig-style item access: `config["array.order"]` mirrors `get`.
        return self.get(key)

    def __contains__(self, key: object) -> bool:
        # donfig-style membership: `"array.order" in config`.
        if not isinstance(key, str):
            return False
        try:
            self.get(key)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        # Defining `__getitem__` above would otherwise make Python fall back to
        # the legacy integer-index iteration protocol (`config[0]` -> confusing
        # `'int' object has no attribute 'split'`). The manager is a keyed config,
        # not a sequence, so fail clearly instead.
        raise TypeError(
            "ZarrConfigManager is not iterable; use config.to_dict() to iterate "
            "its contents, or config.get(<key>) for a single value."
        )

    @property
    def defaults(self) -> dict[str, Any]:
        return to_nested_dict(make_default_config())

    def to_dict(self) -> dict[str, Any]:
        return to_nested_dict(self._current())

    def update(self, updates: Mapping[str, object]) -> None:
        self.set(updates)

    def pprint(self) -> None:
        import pprint as _pp

        _pp.pprint(self.to_dict())

    # --- deprecations -----------------------------------------------------
    @overload
    def _apply_deprecation(self, key: str, *, raise_on_removed: Literal[True]) -> str: ...
    @overload
    def _apply_deprecation(self, key: str, *, raise_on_removed: Literal[False]) -> str | None: ...

    def _apply_deprecation(self, key: str, *, raise_on_removed: bool) -> str | None:
        """Resolve a possibly-deprecated config key.

        Parameters
        ----------
        key : str
            The dotted config key supplied by the caller.
        raise_on_removed : bool
            When `True` (used by `set`), raise `BadConfigError` if the key has been
            removed.  When `False` (used by `get`), return `None` instead so the
            caller can treat the key as absent and honour the caller's default.

        Returns
        -------
        str or None
            The canonical (possibly redirected) key, or `None` when the key was
            removed and `raise_on_removed` is `False`.
        """
        if key not in deprecations:
            return key
        new_key = deprecations[key]
        if new_key is None:
            if raise_on_removed:
                raise BadConfigError(
                    f"Configuration key {key!r} has been removed and no longer has any effect."
                )
            return None
        warnings.warn(
            f"Configuration key {key!r} has been renamed to {new_key!r}.",
            ZarrDeprecationWarning,
            stacklevel=3,
        )
        return new_key


class BadConfigError(ValueError):
    _msg = "bad Config: %r"


# these keys were removed from the config as part of the 3.1.0 release.
# These deprecations should be removed in 3.1.1 or thereabouts.
deprecations: dict[str, str | None] = {
    "array.v2_default_compressor.numeric": None,
    "array.v2_default_compressor.string": None,
    "array.v2_default_compressor.bytes": None,
    "array.v2_default_filters.string": None,
    "array.v2_default_filters.bytes": None,
    "array.v3_default_filters.numeric": None,
    "array.v3_default_filters.raw": None,
    "array.v3_default_filters.bytes": None,
    "array.v3_default_serializer.numeric": None,
    "array.v3_default_serializer.string": None,
    "array.v3_default_serializer.bytes": None,
    "array.v3_default_compressors.string": None,
    "array.v3_default_compressors.bytes": None,
    "array.v3_default_compressors": None,
}

# Backwards-compatible alias: before the typed rewrite, this module exposed the
# donfig subclass as `Config`. Keep the name so `from zarr.core.config import
# Config` and `isinstance(x, Config)` continue to work for downstream code.
Config = ZarrConfigManager

config = ZarrConfigManager()


def parse_indexing_order(data: object) -> Literal["C", "F"]:
    if data in ("C", "F"):
        # the membership check narrows `data` to Literal["C", "F"]
        return data
    msg = f"Expected one of ('C', 'F'), got {data} instead."
    raise ValueError(msg)
