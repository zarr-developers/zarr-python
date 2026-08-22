"""
Parsing and resolution of URL pipelines (https://github.com/jbms/url-pipeline).

Importing this module is cheap and has no side effects: third-party adapters
(registered through the `zarr.url_adapters` entry-point group) are loaded
only when a pipeline URL naming their scheme is actually resolved.

The `|` character is reserved as the pipeline delimiter in every string
store specification: a string containing `|` is always routed through the
pipeline machinery, and no percent-escape is decoded. To address a local
file whose *name* contains `|` (or `#`), pass a `pathlib.Path` instead of a
string.

As a zarr-python extension to the specification (which requires the root
sub-URL of an absolute pipeline to carry a scheme), the root sub-URL may be
a schemeless local filesystem path, e.g. `data/example.zip|zip:`. Schemeless
roots are treated as opaque text — no query or fragment splitting is applied
to them. Such pipelines are not portable to other URL pipeline
implementations; portable pipelines should spell the root as a `file:` URL.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any

from zarr.abc.url_pipeline import (
    AdapterResolution,
    PipelineContext,
    PipelineSegment,
)
from zarr.errors import URLPipelineError
from zarr.registry import get_url_adapter, list_url_adapter_schemes
from zarr.storage._memory import ManagedMemoryStore
from zarr.storage._utils import parse_store_url

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.common import AccessModeLiteral

__all__ = ["is_url_pipeline", "parse_pipeline", "resolve_pipeline"]

# Adapter scheme per RFC 3986 plus "." to permit vendor-prefixed
# nonstandard schemes (e.g. "earthmover.myscheme").
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*$")

# Root scheme at the very start of the sub-URL. The negative lookahead
# excludes fsspec's chained-URL syntax (``zip::file://...``), which is not a
# URL pipeline and keeps flowing through the fsspec machinery.
_ROOT_SCHEME_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*):(?!:)")

# Schemes zarr resolves natively at the pipeline root. These are never
# dispatched to a registered root adapter, so an installed package cannot
# intercept zarr's own local-path and in-memory routing.
_NATIVE_ROOT_SCHEMES = frozenset({"file", "memory"})

# An absolute Windows drive path (C:\... or C:/...), which counts as an
# absolute path in a `file:` pipeline root.
_WINDOWS_DRIVE_RE = re.compile(r"[A-Za-z]:[/\\]")


def _root_scheme(root_sub_url: str) -> str:
    """
    Detect the scheme of the root sub-URL, or `""` for an opaque root.

    Scheme extraction happens on the raw string (so nothing `urlparse`
    would strip or reject — whitespace, exotic authorities — can desync the
    detected scheme from the body). Single-letter candidates are delegated
    to `parse_store_url`, which knows Windows drive letters are not schemes.
    """
    match = _ROOT_SCHEME_RE.match(root_sub_url)
    if match is None:
        return ""
    scheme = match.group(1)
    if len(scheme) == 1:
        try:
            return parse_store_url(root_sub_url).scheme.lower()
        except ValueError:
            return scheme.lower()
    return scheme.lower()


def _split_query(sub_url: str) -> tuple[str, str | None]:
    """Split a sub-URL on the first `?`. Fragments are not supported."""
    if "#" in sub_url:
        raise URLPipelineError(
            f"URL pipeline sub-URLs do not support fragments: {sub_url!r}. "
            "Percent-encode '#' as '%23' if it is part of the path."
        )
    body, sep, query = sub_url.partition("?")
    return body, query if sep else None


def parse_pipeline(url: str) -> tuple[PipelineSegment, ...]:
    """
    Parse a URL pipeline into its `|`-delimited segments.

    The first segment is the *root* sub-URL; its scheme is detected with the
    same rules as ordinary store URLs (Windows drive letters are not
    schemes). Subsequent segments are *adapter* sub-URLs of the form
    `scheme:body` where the trailing colon is optional when the body is
    empty (`zip` is equivalent to `zip:`).

    A schemeless root (a bare local path) is accepted as a zarr-python
    extension to the specification and treated as opaque text: no query or
    fragment splitting applies, since `?` and `#` are ordinary filename
    characters there. See the module docstring.

    Segment text is preserved verbatim (no case or percent-encoding
    normalization) except that schemes are lowercased.
    """
    parts = url.split("|")
    if any(not part for part in parts):
        raise URLPipelineError(f"URL pipeline contains an empty sub-URL: {url!r}")

    segments: list[PipelineSegment] = []
    for index, part in enumerate(parts):
        if index == 0:
            scheme = _root_scheme(part)
            if not scheme:
                # opaque root (bare local path, fsspec chained URL, ...)
                segments.append(PipelineSegment(scheme="", body=part, query=None, raw=part))
                continue
            body_and_scheme, query = _split_query(part)
            body = body_and_scheme[len(scheme) + 1 :]
            segments.append(PipelineSegment(scheme=scheme, body=body, query=query, raw=part))
        else:
            body_and_scheme, query = _split_query(part)
            scheme, _, body = body_and_scheme.partition(":")
            scheme = scheme.lower()
            if not _SCHEME_RE.match(scheme):
                raise URLPipelineError(
                    f"invalid adapter scheme {scheme!r} in pipeline segment {part!r}"
                )
            segments.append(PipelineSegment(scheme=scheme, body=body, query=query, raw=part))
    return tuple(segments)


def _root_routes_to_adapter(scheme: str) -> bool:
    """
    Whether a root sub-URL with this scheme is dispatched to a registered
    root adapter. Schemes zarr resolves natively (`file:`, `memory:`) and
    opaque roots are excluded; the registry check inspects entry-point
    names only — no adapter code is imported here.
    """
    return (
        bool(scheme) and scheme not in _NATIVE_ROOT_SCHEMES and scheme in list_url_adapter_schemes()
    )


def is_url_pipeline(url: str) -> bool:
    """
    Whether `url` should be routed through the URL pipeline machinery.

    True when the URL contains a `|` separator, or when its scheme has a
    registered URL pipeline adapter (a *root adapter* such as `gh:`).
    fsspec chained URLs (`zip::file://...`) and zarr's native `file:` /
    `memory:` schemes are never routed to a root adapter.
    """
    if "|" in url:
        return True
    return _root_routes_to_adapter(_root_scheme(url))


async def resolve_pipeline(
    url: str,
    *,
    mode: AccessModeLiteral | None = None,
    storage_options: dict[str, Any] | None = None,
) -> AdapterResolution:
    """
    Resolve a URL pipeline into a store and a residual path.

    Parameters
    ----------
    url : str
        A URL pipeline, e.g. `"s3://bucket/data.zip|zip:|zarr3:"`.
    mode : AccessModeLiteral | None
        The caller's access mode. `"r"` requires a read-only store; the
        resolver enforces this on whatever the final adapter returns.
    storage_options : dict | None
        Options forwarded to the root sub-URL's store (and visible to
        adapters via the context). Non-dict forms are reserved for future
        per-segment configuration (one mapping per pipeline segment).
    """
    segments = parse_pipeline(url)
    if len(segments) == 1 and not _root_routes_to_adapter(segments[0].scheme):
        raise URLPipelineError(
            f"{url!r} is not a URL pipeline: it has no '|' separator and no "
            f"URL pipeline adapter is registered for scheme {segments[0].scheme!r}"
        )
    return await _resolve(segments, mode=mode, storage_options=storage_options)


async def _resolve(
    segments: tuple[PipelineSegment, ...],
    *,
    mode: AccessModeLiteral | None,
    storage_options: dict[str, Any] | None,
) -> AdapterResolution:
    if len(segments) == 1 and not _root_routes_to_adapter(segments[0].scheme):
        return AdapterResolution(
            store=await _resolve_root(segments[0], mode=mode, storage_options=storage_options)
        )

    *preceding, last = segments
    try:
        adapter_cls = get_url_adapter(last.scheme)
    except URLPipelineError as exc:
        raise URLPipelineError(
            f"{exc} Note: '|' is reserved as the URL pipeline delimiter; to "
            "address a local file whose name contains '|', pass a "
            "pathlib.Path instead of a string."
        ) from None

    context = PipelineContext(
        preceding=tuple(preceding),
        mode=mode,
        storage_options=storage_options,
    )
    resolution = await adapter_cls.open_pipeline_segment(last, context)
    if mode == "r" and not resolution.store.read_only:
        # The caller required read-only; enforce it rather than trusting
        # the adapter to have honored context.read_only.
        try:
            read_only_store = resolution.store.with_read_only(True)
        except NotImplementedError as exc:
            raise URLPipelineError(
                f"adapter {last.scheme!r} returned a writable store for mode 'r', "
                "and the store does not support read-only conversion via "
                ".with_read_only()"
            ) from exc
        await read_only_store._ensure_open()
        resolution = dataclasses.replace(resolution, store=read_only_store)
    return resolution


async def _resolve_root(
    segment: PipelineSegment,
    *,
    mode: AccessModeLiteral | None,
    storage_options: dict[str, Any] | None,
) -> Store:
    """
    Resolve the root sub-URL of a pipeline into a store.

    `memory:` and `file:` roots are resolved here with the URL pipeline
    spec's semantics (spelling equivalences, mandatory absolute `file:`
    paths); everything else — bare local paths, fsspec URLs — delegates to
    the existing `StoreLike` machinery unchanged.
    """
    if segment.scheme == "memory":
        return _resolve_memory_root(segment, mode=mode, storage_options=storage_options)
    from zarr.storage._common import make_store  # circular import

    if segment.scheme == "file":
        # Per the spec: file://localhost/p and file:///p are equivalent to
        # file:/p; other authorities are unsupported; relative paths are
        # forbidden; file: URLs carry no query.
        if segment.query is not None:
            raise URLPipelineError(f"'file:' pipeline roots do not accept a query: {segment.raw!r}")
        body = segment.body
        if body.startswith("//"):
            authority, sep, rest = body[2:].partition("/")
            if authority not in ("", "localhost"):
                raise URLPipelineError(
                    f"unsupported authority {authority!r} in 'file:' pipeline "
                    f"root {segment.raw!r}; only an empty authority or "
                    "'localhost' is allowed"
                )
            body = f"/{rest}" if sep else ""
        # a file URL spells a Windows drive path as /C:/...; strip the
        # leading slash so the local-path machinery sees the drive
        if body.startswith("/") and _WINDOWS_DRIVE_RE.match(body[1:]):
            body = body[1:]
        if not (body.startswith("/") or _WINDOWS_DRIVE_RE.match(body)):
            raise URLPipelineError(
                f"'file:' pipeline roots must carry an absolute path: {segment.raw!r}"
            )
        return await make_store(f"file:{body}", mode=mode, storage_options=storage_options)

    try:
        return await make_store(segment.raw, mode=mode, storage_options=storage_options)
    except ValueError as exc:
        raise URLPipelineError(
            f"could not resolve the pipeline root {segment.raw!r}: {exc}"
        ) from exc


def _resolve_memory_root(
    segment: PipelineSegment,
    *,
    mode: AccessModeLiteral | None,
    storage_options: dict[str, Any] | None,
) -> Store:
    """
    Resolve a `memory:` pipeline root per the spec's spelling equivalences:
    `memory:` ≡ `memory:/` ≡ `memory://`, and `memory:a` ≡ `memory:/a` ≡
    `memory://a`. The first path component names the managed store; the
    remainder is a path within it.
    """
    if segment.query is not None:
        raise URLPipelineError(f"'memory:' pipeline roots do not accept a query: {segment.raw!r}")
    if storage_options:
        raise TypeError(
            "'storage_options' was provided but unused. "
            "'storage_options' is only used when the store is passed as an FSSpec URI string.",
        )
    name, _, path = segment.body.lstrip("/").partition("/")
    return ManagedMemoryStore(name=name, path=path, read_only=mode == "r")
