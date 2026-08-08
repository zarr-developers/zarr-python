"""
Parsing and resolution of URL pipelines (https://github.com/jbms/url-pipeline).

Importing this module is cheap and has no side effects: third-party adapters
(registered through the `zarr.url_adapters` entry-point group) are loaded
only when a pipeline URL naming their scheme is actually resolved.

As a zarr-python extension to the specification (which requires the root
sub-URL of an absolute pipeline to carry a scheme), the root sub-URL may be
a schemeless local filesystem path, e.g. `data/example.zip|zip:`. Such
pipelines are not portable to other URL pipeline implementations; portable
pipelines should spell the root as a `file:` URL.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from zarr.abc.url_pipeline import (
    AdapterResolution,
    PipelineContext,
    PipelineSegment,
)
from zarr.errors import URLPipelineError
from zarr.registry import get_url_adapter, list_url_adapter_schemes
from zarr.storage._utils import parse_store_url

if TYPE_CHECKING:
    from zarr.core.common import AccessModeLiteral

__all__ = ["is_url_pipeline", "parse_pipeline", "resolve_pipeline"]

# Adapter scheme per RFC 3986 plus "." to permit vendor-prefixed
# nonstandard schemes (e.g. "earthmover.myscheme").
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*$")


def _root_scheme(root_sub_url: str) -> str:
    """
    Detect the scheme of the root sub-URL.

    Uses `parse_store_url` (which knows Windows drive letters are not
    schemes), falling back to plain scheme extraction when `urlparse`
    rejects an exotic authority (e.g. bracketed non-IP text) — the pipeline
    grammar constrains only the scheme, not the authority.
    """
    try:
        return parse_store_url(root_sub_url).scheme.lower()
    except ValueError:
        match = re.match(r"([a-zA-Z][a-zA-Z0-9+.\-]*):", root_sub_url)
        return match.group(1).lower() if match else ""


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
    extension to the specification; see the module docstring.

    Segment text is preserved verbatim (no case or percent-encoding
    normalization) except that schemes are lowercased.
    """
    parts = url.split("|")
    if any(not part for part in parts):
        raise URLPipelineError(f"URL pipeline contains an empty sub-URL: {url!r}")

    segments: list[PipelineSegment] = []
    for index, part in enumerate(parts):
        body_and_scheme, query = _split_query(part)
        if index == 0:
            scheme = _root_scheme(body_and_scheme)
            body = body_and_scheme
            if scheme and body_and_scheme.lower().startswith(f"{scheme}:"):
                body = body_and_scheme[len(scheme) + 1 :]
            segments.append(PipelineSegment(scheme=scheme, body=body, query=query, raw=part))
        else:
            scheme, _, body = body_and_scheme.partition(":")
            scheme = scheme.lower()
            if not _SCHEME_RE.match(scheme):
                raise URLPipelineError(
                    f"invalid adapter scheme {scheme!r} in pipeline segment {part!r}"
                )
            segments.append(PipelineSegment(scheme=scheme, body=body, query=query, raw=part))
    return tuple(segments)


def is_url_pipeline(url: str) -> bool:
    """
    Whether `url` should be routed through the URL pipeline machinery.

    True when the URL contains a `|` separator, or when its scheme has a
    registered URL pipeline adapter (a *root adapter* such as `gh:`).
    The registry check inspects entry-point names only — no adapter code is
    imported here.
    """
    if "|" in url:
        return True
    scheme = _root_scheme(url)
    return bool(scheme) and scheme in list_url_adapter_schemes()


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
        The caller's access mode. `"r"` requires adapters to construct
        read-only stores.
    storage_options : dict | None
        Options forwarded to the root sub-URL's store (and visible to
        adapters via the context).
    """
    segments = parse_pipeline(url)
    if len(segments) == 1 and segments[0].scheme not in list_url_adapter_schemes():
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
    if len(segments) == 1 and segments[0].scheme not in list_url_adapter_schemes():
        # Base case: a plain root URL. Delegate to the existing StoreLike
        # machinery (local paths, memory://, fsspec fallthrough) unchanged.
        from zarr.storage._common import make_store  # circular import

        store = await make_store(segments[0].raw, mode=mode, storage_options=storage_options)
        return AdapterResolution(store=store)

    *preceding, last = segments
    adapter_cls = get_url_adapter(last.scheme)

    async def _resolver(preceding: tuple[PipelineSegment, ...]) -> AdapterResolution:
        if not preceding:
            raise URLPipelineError(
                f"adapter {last.scheme!r} was used as a pipeline root but "
                "requires a preceding sub-URL"
            )
        return await _resolve(preceding, mode=mode, storage_options=storage_options)

    context = PipelineContext(
        preceding=tuple(preceding),
        mode=mode,
        read_only=mode == "r",
        storage_options=storage_options,
        _resolver=_resolver,
    )
    return await adapter_cls.open_pipeline_segment(last, context)
