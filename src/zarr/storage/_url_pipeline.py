"""
Parsing and resolution of URL pipelines (https://github.com/jbms/url-pipeline).

Importing this module is cheap and has no side effects: third-party adapters
(registered through the `zarr.url_adapters` entry-point group) are loaded
only when a pipeline URL naming their scheme is actually resolved.

A string is never interpreted as a pipeline on its own: the user wraps it
in a [`URLPipeline`][zarr.storage.URLPipeline], and that object is what the `StoreLike` machinery resolves. Plain string
store specifications, including local paths that happen to contain `|`,
keep their pre-existing meaning. No percent-escape is decoded.

As the specification requires, the root sub-URL carries a URL scheme: a
local path is spelled as an absolute `file:` URL.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zarr.errors import URLPipelineError
from zarr.registry import _get_url_adapter, _list_url_adapter_schemes
from zarr.storage._memory import ManagedMemoryStore
from zarr.storage._utils import _join_paths, normalize_path

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.common import AccessModeLiteral, ZarrFormat
    from zarr.storage._common import StorePath

__all__ = ["PipelineContext", "PipelineSegment", "URLPipeline"]


@dataclasses.dataclass(frozen=True)
class PipelineSegment:
    """
    One `|`-delimited sub-URL of a URL pipeline.

    Attributes
    ----------
    scheme : str
        The lowercased URL scheme.
    body : str
        The text after `scheme:` and before any `?`. Interpretation is
        scheme-defined; it is **not** URL-normalized, so case-significant
        content (e.g. icechunk snapshot IDs) is preserved.
    query : str | None
        The raw query string after `?`, or None. Interpretation is
        scheme-defined.
    raw : str
        The exact original sub-URL text, preserved for lossless
        reconstruction of the pipeline.
    """

    scheme: str
    body: str
    query: str | None
    raw: str

    def __post_init__(self) -> None:
        for name in ("scheme", "body", "raw"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(
                    f"PipelineSegment.{name} must be a str, "
                    f"not {type(getattr(self, name)).__name__}"
                )
        if self.query is not None and not isinstance(self.query, str):
            raise TypeError(
                f"PipelineSegment.query must be a str or None, not {type(self.query).__name__}"
            )
        if not _SCHEME_RE.match(self.scheme) or self.scheme != self.scheme.lower():
            raise URLPipelineError(
                f"PipelineSegment.scheme must be a lowercase URL scheme, not {self.scheme!r}"
            )

    def __str__(self) -> str:
        return self.raw


@dataclasses.dataclass(frozen=True, eq=False)
class PipelineContext:
    """
    Context handed to a [`URLPipelineAdapter`][zarr.abc.url_pipeline.URLPipelineAdapter]
    describing the pipeline to the left of its segment.

    Instances are immutable but compare and hash by identity. An adapter that
    needs to resolve the preceding pipeline with a different mode or options
    derives a new context with `dataclasses.replace` (see
    [`resolve_preceding`][zarr.storage.PipelineContext.resolve_preceding]).

    Attributes
    ----------
    preceding : tuple[PipelineSegment, ...]
        The parsed sub-URLs to the left of the adapter's segment, outer to
        inner. Empty when the adapter's segment is the pipeline root.
    mode : AccessModeLiteral | None
        The access mode requested by the caller (e.g.
        `zarr.open(mode=...)`), or None when unspecified. Adapters for
        read-only resources should raise for every mode that may write, and
        open read-only for `"r"` and None.
    storage_options : dict[str, Any] | None
        Options passed by the caller. By convention these configure the
        *root* sub-URL (e.g. fsspec options); adapters may consume
        adapter-specific keys, and should namespace them (e.g.
        `myscheme_credentials`) to avoid collisions with other segments'
        backends. An adapter that consumes keys should strip them before
        resolving the rest of the pipeline.
    """

    preceding: tuple[PipelineSegment, ...]
    mode: AccessModeLiteral | None
    storage_options: dict[str, Any] | None

    @property
    def read_only(self) -> bool:
        """
        True when the caller requires a read-only store (`mode == "r"`).

        Adapters must construct their store read-only when this is set
        (the resolver enforces it afterwards); when it is False, they may
        construct a writable store if the underlying resource supports
        writing.
        """
        return self.mode == "r"

    @property
    def preceding_url(self) -> str:
        """
        The pipeline to the left of this segment, reconstructed exactly.

        An adapter that consumes this string instead of calling
        [`resolve_preceding`][zarr.storage.PipelineContext.resolve_preceding]
        takes ownership of the *entire* preceding pipeline: it must
        validate every preceding segment itself and raise
        [`URLPipelineError`][zarr.errors.URLPipelineError] for segments it
        does not understand, so that no segment is ever silently ignored.
        """
        return "|".join(segment.raw for segment in self.preceding)

    async def resolve_preceding(self) -> StorePath:
        """
        Resolve the preceding pipeline into a store and a residual path.

        This is the entry point for *wrapper* adapters (e.g. `zip:`) that
        operate on the resource produced by the segments to their left. It
        composes with any preceding adapters, because each segment is
        resolved by its own adapter. Adapters backed by their own I/O
        machinery (e.g. `icechunk:`) may instead consume
        [`preceding_url`][zarr.storage.PipelineContext.preceding_url]
        and never materialize the intermediate store — subject to the
        ownership contract documented there.

        The preceding pipeline is resolved with this context's `mode` and
        `storage_options`. To use others, resolve through a derived
        context:

            preceding = await dataclasses.replace(
                context, mode="r", storage_options=remaining_options
            ).resolve_preceding()

        Wrapper adapters that only read the preceding resource should pass
        `mode="r"`, so the root is opened read-only and without create-on-open
        side effects. Note that a local *file* root (e.g. `file:/data.zip|zip:`)
        can currently only be resolved in mode `"r"`: every other mode creates
        a directory at the root and fails with `FileExistsError`, because
        there is no file-resource primitive yet.
        """
        if not self.preceding:
            raise URLPipelineError(
                "this adapter segment is at the pipeline root; "
                "there is no preceding sub-URL to resolve"
            )
        return await _resolve(self.preceding, mode=self.mode, storage_options=self.storage_options)


# Format segments are consumed by zarr-python itself rather than by a
# registered adapter: they select the zarr format of the node the pipeline
# addresses (`zarr:` detects it), and their body is a path to that node.
_FORMAT_SCHEMES: dict[str, ZarrFormat | None] = {"zarr": None, "zarr2": 2, "zarr3": 3}

# Adapter scheme per RFC 3986 plus "." to permit vendor-prefixed
# nonstandard schemes (e.g. "earthmover.myscheme").
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*$")

# Root scheme at the very start of the sub-URL.
_ROOT_SCHEME_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*):")

# Something shaped like a scheme (text up to the first ':' with no '/', '?'
# or '#' before it) that is not a valid one, e.g. `a_b://x`.
_SCHEME_LIKE_RE = re.compile(r"[^:/?#]+:")

# Schemes zarr resolves natively at the pipeline root. These are never
# dispatched to a registered root adapter, so an installed package cannot
# intercept zarr's own local-path and in-memory routing.
_NATIVE_ROOT_SCHEMES = frozenset({"file", "memory"})

# Schemes zarr-python handles itself, which therefore cannot be given to a
# registered adapter.
_RESERVED_SCHEMES = _NATIVE_ROOT_SCHEMES | frozenset(_FORMAT_SCHEMES)

# An absolute Windows drive path (C:\... or C:/...), which counts as an
# absolute path in a `file:` pipeline root.
_WINDOWS_DRIVE_RE = re.compile(r"[A-Za-z]:[/\\]")

# A bare Windows drive path as a pipeline root (C:\data, C:/data), as
# opposed to a single-letter scheme with an authority (x://host/path).
_BARE_DRIVE_ROOT_RE = re.compile(r"[A-Za-z]:(\\|/(?!/))")

# ASCII control characters (C0 and DEL). No sub-URL may contain them.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


def _root_scheme(root_sub_url: str) -> str:
    """
    Detect the scheme of the root sub-URL, or `""` when it has none.

    Scheme extraction happens on the raw string, so nothing `urlparse`
    would strip or reject can desync the detected scheme from the body. A
    single letter followed by `:` is a scheme on every platform; a bare
    Windows drive path (`C:\\...`, `C:/...`) is rejected before this is
    called, since a pipeline spells it `file:/C:/...`.
    """
    match = _ROOT_SCHEME_RE.match(root_sub_url)
    if match is None:
        return ""
    return match.group(1).lower()


def _split_query(sub_url: str) -> tuple[str, str | None]:
    """Split a sub-URL on the first `?`. Fragments are not supported."""
    if "#" in sub_url:
        raise URLPipelineError(
            f"URL pipeline sub-URLs do not support fragments ('#'): {sub_url!r}. "
            "Percent-escapes are not decoded, so a '#' in a path cannot be addressed."
        )
    body, sep, query = sub_url.partition("?")
    return body, query if sep else None


def parse_pipeline(url: str) -> tuple[PipelineSegment, ...]:
    """
    Parse a URL pipeline into its `|`-delimited segments.

    The first segment is the *root* sub-URL. Subsequent segments are *adapter* sub-URLs of the form
    `scheme:body` where the trailing colon is optional when the body is
    empty (`zip` is equivalent to `zip:`).

    The root must carry a scheme; a bare local path is rejected, since the
    specification has no schemeless sub-URLs and `|` could not be escaped
    in one.

    Segment text is preserved verbatim (no case or percent-encoding
    normalization) except that schemes are lowercased. ASCII control
    characters, and whitespace at either end of a sub-URL, are rejected
    rather than stripped.
    """
    if not isinstance(url, str):
        raise TypeError(f"a URL pipeline must be a str, not {type(url).__name__}")
    if (match := _CONTROL_CHAR_RE.search(url)) is not None:
        raise URLPipelineError(
            f"URL pipeline contains the control character {match.group()!r}: {url!r}"
        )
    parts = url.split("|")
    if any(not part for part in parts):
        raise URLPipelineError(f"URL pipeline contains an empty sub-URL: {url!r}")
    for part in parts:
        if part != part.strip():
            raise URLPipelineError(
                f"sub-URL {part!r} of the URL pipeline {url!r} has leading or trailing "
                "whitespace; remove it"
            )

    segments: list[PipelineSegment] = []
    for index, part in enumerate(parts):
        if index == 0:
            if _BARE_DRIVE_ROOT_RE.match(part):
                raise URLPipelineError(
                    f"the root sub-URL {part!r} of the URL pipeline {url!r} is a Windows drive "
                    "path; spell it as a 'file:' URL (e.g. 'file:/C:/data')."
                )
            scheme = _root_scheme(part)
            if not scheme and _SCHEME_LIKE_RE.match(part):
                invalid = part.partition(":")[0]
                raise URLPipelineError(
                    f"invalid URL scheme {invalid!r} in the root sub-URL {part!r} "
                    f"of the URL pipeline {url!r}"
                )
            if not scheme:
                raise URLPipelineError(
                    f"the root sub-URL {part!r} of the URL pipeline {url!r} has no URL scheme. "
                    "Spell a local path as an absolute 'file:' URL. Percent-escapes are not "
                    "decoded, so a local path containing '|' cannot be addressed by a URL "
                    "pipeline; open it with a plain path or a store object instead."
                )
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


@dataclasses.dataclass(frozen=True)
class URLPipeline:
    """
    A parsed URL pipeline, ready to be passed wherever a `StoreLike` is accepted.

    Construct one with [`from_url`][zarr.storage.URLPipeline.from_url] and open it
    through any `StoreLike`-taking function. Because a
    `URLPipeline` is an explicit object, no plain string store specification is
    ever interpreted as a pipeline: `zarr.open("data|x.zarr")` still addresses a
    local directory of that name, while
    `zarr.open(URLPipeline.from_url("s3://bucket/data.zip|zip:|zarr3:"))` resolves
    the pipeline through registered adapters.

    A pipeline is a value: instances are immutable, hashable, and compare equal
    when they hold the same segments. Nothing is resolved until the pipeline is
    opened; resolution needs the access mode and storage options of the open
    call.

    Attributes
    ----------
    segments : tuple[PipelineSegment, ...]
        Every `|`-delimited sub-URL of the pipeline, in order: the root, any
        adapter segments, and an optional trailing format segment.

    Raises
    ------
    TypeError
        If `segments` is not a tuple of `PipelineSegment`.
    URLPipelineError
        If there are no segments, if a format segment (`zarr:`/`zarr2:`/`zarr3:`)
        appears anywhere but last (including as the root), or if a format segment
        carries a query or a node path with `.` or `..` components.
    """

    segments: tuple[PipelineSegment, ...]

    def __post_init__(self) -> None:
        segments: object = self.segments  # runtime check of a hand-built pipeline
        if not isinstance(segments, tuple) or not all(
            isinstance(segment, PipelineSegment) for segment in segments
        ):
            raise TypeError(
                "URLPipeline.segments must be a tuple of PipelineSegment; "
                "use URLPipeline.from_url to parse a string"
            )
        if not self.segments:
            raise URLPipelineError("a URL pipeline needs at least one segment")
        if self.segments[0].scheme in _FORMAT_SCHEMES:
            raise URLPipelineError(
                f"the format segment {self.segments[0].raw!r} cannot be the root of "
                f"the URL pipeline {str(self)!r}; the root must locate a resource"
            )
        for segment in self.segments[1:-1]:
            if segment.scheme in _FORMAT_SCHEMES:
                raise URLPipelineError(
                    f"the format segment {segment.raw!r} must be the last segment "
                    f"of the URL pipeline {str(self)!r}"
                )
        format_segment = self._format_segment
        if format_segment is not None and format_segment.query is not None:
            raise URLPipelineError(
                f"'{format_segment.scheme}:' segments do not accept a query: {format_segment.raw!r}"
            )
        if format_segment is not None:
            # validate the node path now, before anything is resolved
            try:
                normalize_path(format_segment.body)
            except ValueError as exc:
                raise URLPipelineError(
                    f"invalid node path in the format segment {format_segment.raw!r}: {exc}"
                ) from None

    def __repr__(self) -> str:
        return f"URLPipeline({str(self)!r})"

    @classmethod
    def from_url(cls, url: str) -> URLPipeline:
        """
        Parse a URL pipeline string.

        A URL without a `|` is a trivial pipeline consisting of its root alone.
        Parsing is purely syntactic: nothing is resolved, and no adapter is
        loaded, until the pipeline is opened.

        Raises
        ------
        TypeError
            If `url` is not a `str`.
        URLPipelineError
            If the string cannot be parsed: an empty sub-URL, a root without a URL
            scheme (including a bare local or Windows drive path), an invalid
            adapter scheme, a fragment, an ASCII control character, or whitespace
            at either end of a sub-URL; or if the resulting pipeline is invalid
            (see the class docstring).
        """
        return cls(parse_pipeline(url))

    @property
    def _format_segment(self) -> PipelineSegment | None:
        last = self.segments[-1]
        if len(self.segments) > 1 and last.scheme in _FORMAT_SCHEMES:
            return last
        return None

    @property
    def _store_segments(self) -> tuple[PipelineSegment, ...]:
        """
        The segments that address a store: the root and any adapter segments,
        i.e. `segments` without a trailing format segment.
        """
        if self._format_segment is None:
            return self.segments
        return self.segments[:-1]

    @property
    def _zarr_format(self) -> ZarrFormat | None:
        """The format selected by a trailing `zarr2:`/`zarr3:` segment, or None
        (also for a trailing `zarr:` segment, which detects the format)."""
        format_segment = self._format_segment
        if format_segment is None:
            return None
        return _FORMAT_SCHEMES[format_segment.scheme]

    @property
    def _node_path(self) -> str:
        """
        The node path given as the body of a trailing format segment,
        normalized. Empty when there is no format segment or its body is empty.
        """
        format_segment = self._format_segment
        if format_segment is None:
            return ""
        return normalize_path(format_segment.body)

    def _resolve_zarr_format(self, zarr_format: ZarrFormat | None) -> ZarrFormat | None:
        """
        Combine the format selected by this pipeline with a caller-supplied one.

        Parameters
        ----------
        zarr_format : ZarrFormat | None
            The format requested by the caller, or None when unspecified.

        Returns
        -------
        ZarrFormat | None
            The pipeline's format when the caller did not specify one, otherwise
            the caller's format. None when neither is set.

        Raises
        ------
        ValueError
            If the caller's format differs from the one selected by the pipeline.
        """
        pipeline_format = self._zarr_format
        if pipeline_format is None:
            return zarr_format
        if zarr_format is not None and zarr_format != pipeline_format:
            raise ValueError(
                f"zarr_format={zarr_format} conflicts with the 'zarr{pipeline_format}:' "
                f"segment of the URL pipeline {str(self)!r}; pass zarr_format=None to use "
                "the pipeline's format"
            )
        return pipeline_format

    async def _resolve_store_path(
        self,
        *,
        mode: AccessModeLiteral | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> StorePath:
        """
        Resolve the pipeline into a store and a residual path.

        The residual path joins the path returned by the adapters with the
        node path of a trailing format segment (`...|zarr3:path/to/node`).
        The format itself is not part of the result; read it from
        `_zarr_format`.

        Parameters
        ----------
        mode : AccessModeLiteral | None
            The caller's access mode. `"r"` requires a read-only store; the
            resolver enforces this on whatever the final adapter returns.
        storage_options : dict | None
            Options forwarded to the root sub-URL's store (and visible to
            adapters via the context).

        Raises
        ------
        URLPipelineError
            If a segment names a scheme with no registered adapter, names an
            adapter entry point that fails to import, or the root sub-URL
            cannot be resolved.
        TypeError
            If `storage_options` are passed to a root that does not accept
            them (`file:` and `memory:`), as for non-pipeline stores.
        OSError
            Errors from opening the root resource (e.g. a missing local
            directory in mode `"r"`) propagate unchanged, as for non-pipeline
            stores.
        """
        from zarr.storage._common import StorePath  # circular import

        resolved = await _resolve(self._store_segments, mode=mode, storage_options=storage_options)
        if not self._node_path:
            return resolved
        # both paths are already normalized: StorePath normalizes the adapter's,
        # and __post_init__ validated the format segment's
        return StorePath(resolved.store, _join_paths([resolved.path, self._node_path]))

    def __str__(self) -> str:
        return "|".join(segment.raw for segment in self.segments)


def _root_routes_to_adapter(scheme: str) -> bool:
    """
    Whether a root sub-URL with this scheme is dispatched to a registered
    root adapter. Schemes zarr resolves natively (`file:`, `memory:`) are
    excluded; the registry check inspects entry-point names only — no
    adapter code is imported here.
    """
    return scheme not in _NATIVE_ROOT_SCHEMES and scheme in _list_url_adapter_schemes()


async def _resolve(
    segments: tuple[PipelineSegment, ...],
    *,
    mode: AccessModeLiteral | None,
    storage_options: dict[str, Any] | None,
) -> StorePath:
    from zarr.storage._common import StorePath  # circular import

    if len(segments) == 1 and not _root_routes_to_adapter(segments[0].scheme):
        return StorePath(
            await _resolve_root(segments[0], mode=mode, storage_options=storage_options)
        )

    *preceding, last = segments
    adapter_cls = _get_url_adapter(last.scheme)

    context = PipelineContext(
        preceding=tuple(preceding),
        mode=mode,
        storage_options=storage_options,
    )
    resolved = await adapter_cls.open_pipeline_segment(last, context)
    if mode == "r" and not resolved.store.read_only:
        # The caller required read-only; enforce it rather than trusting
        # the adapter to have honored context.read_only.
        try:
            read_only_store = resolved.store.with_read_only(True)
            await read_only_store._ensure_open()
        except NotImplementedError as exc:
            resolved.store.close()
            raise URLPipelineError(
                f"adapter {last.scheme!r} returned a writable store for mode 'r', "
                "and the store does not support read-only conversion via "
                ".with_read_only()"
            ) from exc
        except Exception:
            resolved.store.close()
            raise
        resolved = StorePath(read_only_store, resolved.path)
    return resolved


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
    paths); every other scheme (e.g. fsspec URLs) delegates to the existing
    `StoreLike` machinery unchanged.
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
            if body.startswith("//"):
                # the spec's file-path-absolute grammar forbids an empty first
                # path component
                raise URLPipelineError(
                    f"'file:' pipeline root paths must not start with '//': {segment.raw!r}"
                )
        if sys.platform == "win32":
            # a file URL spells a Windows drive path as /C:/...; strip the
            # leading slash so the local-path machinery sees the drive
            if body.startswith("/") and _WINDOWS_DRIVE_RE.match(body[1:]):
                body = body[1:]
            is_absolute = body.startswith("/") or bool(_WINDOWS_DRIVE_RE.match(body))
        else:
            is_absolute = body.startswith("/")
        if not is_absolute:
            raise URLPipelineError(
                f"'file:' pipeline roots must carry an absolute path: {segment.raw!r}"
            )
        _reject_dot_segments(body, segment)
        return await make_store(Path(body), mode=mode, storage_options=storage_options)

    # fsspec matches protocols case-sensitively; schemes are case-insensitive
    url = segment.scheme + segment.raw[len(segment.scheme) :]
    try:
        return await make_store(url, mode=mode, storage_options=storage_options)
    except ValueError as exc:
        raise URLPipelineError(
            f"could not resolve the pipeline root {segment.raw!r}: {exc}"
        ) from exc


def _reject_dot_segments(path: str, segment: PipelineSegment) -> None:
    """Reject `.` and `..` path components in a `file:` or `memory:` root."""
    if any(part in (".", "..") for part in re.split(r"[/\\]", path)):
        raise URLPipelineError(
            f"'{segment.scheme}:' pipeline root paths must not contain '.' or '..' "
            f"components: {segment.raw!r}"
        )


# The single in-memory store every `memory:` pipeline root addresses.
_PIPELINE_MEMORY_STORE_NAME = ""


def _resolve_memory_root(
    segment: PipelineSegment,
    *,
    mode: AccessModeLiteral | None,
    storage_options: dict[str, Any] | None,
) -> Store:
    """
    Resolve a `memory:` pipeline root per the spec: there is a single
    in-memory store, and the body is a path within it. Spellings follow the
    spec's equivalences: `memory:` ≡ `memory:/` ≡ `memory://`, and
    `memory:a/b` ≡ `memory:/a/b` ≡ `memory://a/b`.
    """
    if segment.query is not None:
        raise URLPipelineError(f"'memory:' pipeline roots do not accept a query: {segment.raw!r}")
    if storage_options is not None:
        raise TypeError(
            "'storage_options' was provided but unused. "
            "'storage_options' is only used when the store is passed as an FSSpec URI string.",
        )
    path = segment.body.lstrip("/")
    _reject_dot_segments(path, segment)
    return ManagedMemoryStore(name=_PIPELINE_MEMORY_STORE_NAME, path=path, read_only=mode == "r")
