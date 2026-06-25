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

import ast
import contextlib
import os
import warnings
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal, Self, cast, overload

from zarr.errors import ZarrDeprecationWarning

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
    return cast(ZarrConfig, _replace_recursive(cfg, segments, value, key))


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


ENV_PREFIX = "ZARR_"

# Meta-variables that control WHERE config is loaded from, not config values themselves.
# These must be excluded from the env-override map to avoid spurious KeyErrors.
_ENV_META_VARS: frozenset[str] = frozenset({"ZARR_CONFIG"})


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

    Variables listed in ``_ENV_META_VARS`` (e.g. ``ZARR_CONFIG``) are
    directives about where config lives and are skipped.
    """
    out: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        if name in _ENV_META_VARS:
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
            candidates.extend(
                os.path.join(path, fn)
                for fn in sorted(os.listdir(path))
                if fn.endswith((".yaml", ".yml"))
            )
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
        if isinstance(v, Mapping) and k != "codecs":
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
    return apply_overrides(
        apply_overrides(make_default_config(), collect_yaml(_config_search_paths())),
        collect_env(environ),
    )


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

    def __enter__(self) -> Self:
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
        resolved = self._apply_deprecation(key, raise_on_removed=False)
        if resolved is None:
            # Key was removed; treat as absent — honour the caller's default.
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
    def set(self, updates: Mapping[str, Any] | None = None, **kwargs: Any) -> _ConfigSet:
        """Apply one or more config overrides.

        Accepts either a mapping of dotted keys to values, keyword arguments
        (for top-level keys), or both::

            config.set({"array.order": "F"})
            config.set(default_zarr_format=2)
        """
        all_updates: dict[str, Any] = {}
        if updates:
            all_updates.update(updates)
        all_updates.update(kwargs)
        prev_base = self._base
        new = self._current()
        for key, value in all_updates.items():
            resolved = self._apply_deprecation(key, raise_on_removed=True)
            new = replace_path(new, resolved, value)
        self._base = new
        token = self._scope.set(new)
        return _ConfigSet(self, prev_base, token)

    # --- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        self._base = build_config()
        # Sync the scope so _current() returns the new base in this context.
        self._scope.set(self._base)

    def refresh(self) -> None:
        self._base = build_config()
        # Sync the scope so the rebuilt base is visible in the calling context.
        # Without this, any prior reset()/set() scope entry would shadow the refresh.
        self._scope.set(self._base)

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

config = ZarrConfigManager()


def parse_indexing_order(data: Any) -> Literal["C", "F"]:
    if data in ("C", "F"):
        return cast("Literal['C', 'F']", data)
    msg = f"Expected one of ('C', 'F'), got {data} instead."
    raise ValueError(msg)
