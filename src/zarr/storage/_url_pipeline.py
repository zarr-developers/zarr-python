"""
Parsing and resolution of URL pipelines (https://github.com/jbms/url-pipeline).

Only [`URLPipeline.from_url`][zarr.storage.URLPipeline.from_url] reads a string
as a pipeline; plain string store specifications keep their meaning. Adapters
registered through the `zarr.url_adapters` entry-point group are imported only
when a pipeline naming their scheme is resolved. No percent-escape is decoded.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

from zarr.errors import URLPipelineError
from zarr.registry import _URL_SCHEME_RE, _get_url_adapter, _list_url_adapter_schemes
from zarr.storage._memory import ManagedMemoryStore
from zarr.storage._utils import _join_paths, normalize_path

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.common import AccessModeLiteral, ZarrFormat
    from zarr.storage._common import StorePath

__all__ = ["PipelineContext", "PipelineSegment", "URLPipeline"]

# Format segments are read by zarr-python itself, not by an adapter: they
# select the zarr format (`zarr:` detects it), and their body is a node path.
_FORMAT_SCHEMES: dict[str, ZarrFormat | None] = {"zarr": None, "zarr2": 2, "zarr3": 3}

# Roots zarr resolves itself; an adapter can never claim them.
_NATIVE_ROOT_SCHEMES = frozenset({"file", "memory"})

# More segments than any real pipeline needs; bounds resolution recursion.
_MAX_SEGMENTS = 32

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_DRIVE_RE = re.compile(r"[A-Za-z]:/")


def _redact(text: str) -> str:
    """Hide query strings and URL userinfo, which may carry credentials."""
    text = re.sub(r"//[^/@|?]*@", "//…@", text)
    return re.sub(r"\?[^|]*", "?…", text)


def _check_scheme(scheme: str, raw: str) -> None:
    if not _URL_SCHEME_RE.fullmatch(scheme):
        raise URLPipelineError(f"invalid URL scheme {scheme!r} in the sub-URL {_redact(raw)!r}")


@dataclasses.dataclass(frozen=True)
class PipelineSegment:
    """
    One `|`-delimited sub-URL of a URL pipeline.

    Attributes
    ----------
    scheme : str
        The lowercased URL scheme.
    body : str
        The text after `scheme:` and before any `?`, verbatim (case-significant
        content such as Icechunk snapshot IDs is preserved).
    query : str | None
        The raw query string after `?`, or None.
    raw : str
        The original sub-URL text.
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
        _check_scheme(self.scheme, self.raw)
        if self.scheme != self.scheme.lower():
            raise URLPipelineError(f"PipelineSegment.scheme must be lowercase: {self.scheme!r}")

    def __repr__(self) -> str:
        return f"PipelineSegment({_redact(self.raw)!r})"


@dataclasses.dataclass(frozen=True, eq=False)
class PipelineContext:
    """
    Context handed to a [`URLPipelineAdapter`][zarr.abc.url_pipeline.URLPipelineAdapter]
    describing the pipeline to the left of its segment.

    Instances are immutable and compare by identity. Derive a context with
    other values with `dataclasses.replace`.

    Attributes
    ----------
    preceding : tuple[PipelineSegment, ...]
        The sub-URLs to the left of the adapter's segment, outer to inner.
        Empty when the adapter's segment is the pipeline root.
    mode : AccessModeLiteral | None
        The caller's access mode, or None. Adapters for read-only resources
        should raise for every mode except `"r"`.
    storage_options : dict[str, Any] | None
        The caller's options. They configure the root sub-URL (e.g. fsspec
        options), and every adapter in the pipeline sees all of them: an
        adapter should consume only keys namespaced for it (e.g.
        `myscheme_token`), and strip them before resolving the rest of the
        pipeline.
    """

    preceding: tuple[PipelineSegment, ...]
    mode: AccessModeLiteral | None
    storage_options: dict[str, Any] | None = dataclasses.field(repr=False)

    @property
    def read_only(self) -> bool:
        """
        True when the caller requires a read-only store (`mode == "r"`).
        The resolver enforces this on the store the adapter returns.
        """
        return self.mode == "r"

    @property
    def preceding_url(self) -> str:
        """
        The pipeline to the left of this segment, reconstructed exactly.

        An adapter that consumes this string instead of calling
        `resolve_preceding` must validate every preceding segment itself,
        and raise `URLPipelineError` for any it does not understand.
        """
        return "|".join(segment.raw for segment in self.preceding)

    async def resolve_preceding(self) -> StorePath:
        """
        Resolve the preceding pipeline, with this context's `mode` and
        `storage_options`, into a store and a residual path.

        Wrapper adapters (e.g. `zip:`) that only read the preceding resource
        should resolve it read-only:
        `await dataclasses.replace(context, mode="r").resolve_preceding()`.
        A local *file* root can currently only be resolved in mode `"r"`.
        """
        if not self.preceding:
            raise URLPipelineError(
                "this adapter segment is at the pipeline root; "
                "there is no preceding sub-URL to resolve"
            )
        return await _resolve(self.preceding, mode=self.mode, storage_options=self.storage_options)


def _split_query(sub_url: str) -> tuple[str, str | None]:
    if "#" in sub_url:
        raise URLPipelineError(
            f"URL pipeline sub-URLs do not support fragments ('#'): {_redact(sub_url)!r}"
        )
    body, sep, query = sub_url.partition("?")
    return body, query if sep else None


def _parse_pipeline(url: str) -> tuple[PipelineSegment, ...]:
    """
    Split a URL pipeline into segments. Adapter segments are `scheme:body`,
    where the colon is optional when the body is empty (`zip` ≡ `zip:`).
    """
    if not isinstance(url, str):
        raise TypeError(f"a URL pipeline must be a str, not {type(url).__name__}")
    if (match := _CONTROL_CHAR_RE.search(url)) is not None:
        raise URLPipelineError(f"URL pipeline contains the control character {match.group()!r}")
    parts = url.split("|")
    if len(parts) > _MAX_SEGMENTS:
        raise URLPipelineError(f"URL pipeline has more than {_MAX_SEGMENTS} segments")
    if any(not part for part in parts):
        raise URLPipelineError(f"URL pipeline contains an empty sub-URL: {_redact(url)!r}")

    segments = []
    for index, part in enumerate(parts):
        if part != part.strip():
            raise URLPipelineError(f"sub-URL {_redact(part)!r} has leading or trailing whitespace")
        body_and_scheme, query = _split_query(part)
        scheme, sep, body = body_and_scheme.partition(":")
        if index == 0:
            if not sep or not scheme or "/" in scheme or "\\" in scheme:
                raise URLPipelineError(
                    f"the root sub-URL {_redact(part)!r} has no URL scheme. Spell a local path "
                    "as an absolute 'file:' URL; a local path containing '|' cannot be "
                    "addressed by a URL pipeline, so open it with a plain path instead."
                )
            _check_scheme(scheme, part)
            if len(scheme) == 1:
                raise URLPipelineError(
                    f"the root sub-URL {_redact(part)!r} looks like a Windows drive path; "
                    "spell it as a 'file:' URL (e.g. 'file:/C:/data')"
                )
        else:
            _check_scheme(scheme, part)
        segments.append(PipelineSegment(scheme=scheme.lower(), body=body, query=query, raw=part))
    return tuple(segments)


@dataclasses.dataclass(frozen=True)
class URLPipeline:
    """
    A parsed URL pipeline, accepted wherever a `StoreLike` is.

    Build one with [`from_url`][zarr.storage.URLPipeline.from_url] and pass it to
    any `StoreLike`-taking function, e.g.
    `zarr.open_group(URLPipeline.from_url("s3://bucket/data.zip|zip:|zarr3:"))`.
    Instances are immutable and hashable values; nothing is resolved until the
    pipeline is opened.

    Attributes
    ----------
    segments : tuple[PipelineSegment, ...]
        The sub-URLs in order: the root, any adapter segments, and an optional
        trailing format segment (`zarr:`, `zarr2:` or `zarr3:`).

    Raises
    ------
    TypeError
        If `segments` is not a tuple of `PipelineSegment`.
    URLPipelineError
        If there are no segments or more than 32, if a format segment is
        anywhere but last (including the root), or if a format segment carries
        a query or a node path with `.` or `..` components.
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
        if not 1 <= len(self.segments) <= _MAX_SEGMENTS:
            raise URLPipelineError(f"a URL pipeline needs 1 to {_MAX_SEGMENTS} segments")
        if self.segments[0].scheme in _FORMAT_SCHEMES:
            raise URLPipelineError(f"a format segment cannot be the root of {self!r}")
        if any(segment.scheme in _FORMAT_SCHEMES for segment in self.segments[1:-1]):
            raise URLPipelineError(f"a format segment must be the last segment of {self!r}")
        format_segment = self._format_segment
        if format_segment is not None:
            if format_segment.query is not None:
                raise URLPipelineError(f"format segments do not accept a query: {self!r}")
            try:
                normalize_path(format_segment.body)
            except ValueError:
                raise URLPipelineError(
                    f"the node path of a format segment must not contain '.' or '..': {self!r}"
                ) from None

    @classmethod
    def from_url(cls, url: str) -> URLPipeline:
        """
        Parse a URL pipeline string. A URL without `|` is a pipeline of its
        root alone. Parsing loads no adapter.

        Raises
        ------
        TypeError
            If `url` is not a `str`.
        URLPipelineError
            If the string cannot be parsed: an empty sub-URL, a root without a
            URL scheme (including a local or Windows drive path), an invalid
            scheme, a fragment, an ASCII control character, or whitespace at
            either end of a sub-URL; or if the pipeline is invalid (see the
            class docstring).
        """
        return cls(_parse_pipeline(url))

    def __str__(self) -> str:
        return "|".join(segment.raw for segment in self.segments)

    def __repr__(self) -> str:
        return f"URLPipeline({_redact(str(self))!r})"

    @property
    def _format_segment(self) -> PipelineSegment | None:
        last = self.segments[-1]
        if len(self.segments) > 1 and last.scheme in _FORMAT_SCHEMES:
            return last
        return None

    async def _resolve_store_path(
        self,
        *,
        mode: AccessModeLiteral | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> StorePath:
        """
        Resolve the pipeline into a store and a path: the adapters' residual
        path joined with the node path of a trailing format segment.
        """
        from zarr.storage._common import StorePath  # circular import

        format_segment = self._format_segment
        if format_segment is None:
            return await _resolve(self.segments, mode=mode, storage_options=storage_options)
        resolved = await _resolve(self.segments[:-1], mode=mode, storage_options=storage_options)
        node_path = normalize_path(format_segment.body)
        return StorePath(resolved.store, _join_paths([resolved.path, node_path]))


async def _resolve(
    segments: tuple[PipelineSegment, ...],
    *,
    mode: AccessModeLiteral | None,
    storage_options: dict[str, Any] | None,
) -> StorePath:
    from zarr.storage._common import StorePath  # circular import

    *preceding, last = segments
    if not preceding and (
        last.scheme in _NATIVE_ROOT_SCHEMES or last.scheme not in _list_url_adapter_schemes()
    ):
        return StorePath(await _resolve_root(last, mode=mode, storage_options=storage_options))

    adapter_cls = _get_url_adapter(last.scheme)
    context = PipelineContext(
        preceding=tuple(preceding), mode=mode, storage_options=storage_options
    )
    resolved = await adapter_cls.open_pipeline_segment(last, context)
    if mode == "r" and not resolved.store.read_only:
        # enforce read-only rather than trusting the adapter
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


def _reject_dot_segments(path: str, segment: PipelineSegment) -> None:
    if any(part in (".", "..") for part in re.split(r"[/\\]", path)):
        raise URLPipelineError(
            f"'{segment.scheme}:' pipeline root paths must not contain '.' or '..' "
            f"components: {_redact(segment.raw)!r}"
        )


async def _resolve_root(
    segment: PipelineSegment,
    *,
    mode: AccessModeLiteral | None,
    storage_options: dict[str, Any] | None,
) -> Store:
    """
    Resolve a root sub-URL. `memory:` and `file:` follow the URL pipeline
    spec; every other scheme goes to the plain-string machinery (fsspec).
    """
    from zarr.storage._common import make_store  # circular import

    if segment.scheme in _NATIVE_ROOT_SCHEMES and segment.query is not None:
        raise URLPipelineError(
            f"'{segment.scheme}:' pipeline roots do not accept a query: {_redact(segment.raw)!r}"
        )

    if segment.scheme == "memory":
        # one in-memory store; the body is a path within it
        # (memory:a ≡ memory:/a ≡ memory://a)
        if storage_options is not None:
            raise TypeError(
                "'storage_options' was provided but unused. "
                "'storage_options' is only used when the store is passed as an FSSpec URI string.",
            )
        path = segment.body.lstrip("/")
        _reject_dot_segments(path, segment)
        return ManagedMemoryStore(name="", path=path, read_only=mode == "r")

    if segment.scheme == "file":
        # file://localhost/p ≡ file:///p ≡ file:/p; other authorities and
        # relative paths are rejected. On Windows '\' is a separator too, so
        # check the '/'-spelled path (this also rejects UNC paths like /\host).
        body = segment.body.replace("\\", "/") if sys.platform == "win32" else segment.body
        if body.startswith("//"):
            authority, sep, rest = body[2:].partition("/")
            if authority.lower() not in ("", "localhost"):
                raise URLPipelineError(
                    f"unsupported authority in the 'file:' pipeline root {_redact(segment.raw)!r}; "
                    "only an empty authority or 'localhost' is allowed"
                )
            body = f"/{rest}" if sep else ""
            if body.startswith("//"):
                raise URLPipelineError(
                    f"'file:' pipeline root paths must not start with '//': "
                    f"{_redact(segment.raw)!r}"
                )
        if sys.platform == "win32":
            # a file URL spells a drive path /C:/...; drop the leading slash
            if body.startswith("/") and _WINDOWS_DRIVE_RE.match(body[1:]):
                body = body[1:]
            drive = PureWindowsPath(body).drive
            is_absolute = (body.startswith("/") and not drive) or bool(
                _WINDOWS_DRIVE_RE.match(body)
            )
            if ":" in body[len(drive) :]:
                # elsewhere ':' names an NTFS alternate data stream
                raise URLPipelineError(
                    f"':' is only allowed in the drive of a 'file:' pipeline root on "
                    f"Windows: {_redact(segment.raw)!r}"
                )
        else:
            is_absolute = body.startswith("/")
        if not is_absolute:
            raise URLPipelineError(
                f"'file:' pipeline roots must carry an absolute local path: "
                f"{_redact(segment.raw)!r}"
            )
        _reject_dot_segments(body, segment)
        return await make_store(Path(body), mode=mode, storage_options=storage_options)

    # fsspec matches protocols case-sensitively; URL schemes are not
    url = segment.scheme + segment.raw[len(segment.scheme) :]
    try:
        return await make_store(url, mode=mode, storage_options=storage_options)
    except ValueError as exc:
        raise URLPipelineError(
            f"could not resolve the pipeline root {_redact(segment.raw)!r}: {_redact(str(exc))}"
        ) from exc
