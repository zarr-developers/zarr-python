"""
Abstract base class and data model for URL pipeline adapters.

A URL pipeline is a `|`-separated chain of sub-URLs, read outer-to-inner,
as specified by https://github.com/jbms/url-pipeline. The first sub-URL (the
*root*) locates a resource using a conventional URL, and each subsequent
sub-URL names an *adapter* that reinterprets everything to its left:

    s3://bucket/data.zip|zip:path/inside|zarr3:

Third-party packages provide adapters by subclassing
[`URLPipelineAdapter`][zarr.abc.url_pipeline.URLPipelineAdapter] and
registering the class under the `zarr.url_adapters` entry-point group,
using the URL scheme as the entry-point name.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zarr.errors import URLPipelineError

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.common import AccessModeLiteral, ZarrFormat

__all__ = [
    "AdapterResolution",
    "PipelineContext",
    "PipelineSegment",
    "URLPipelineAdapter",
]


class _Unset(enum.Enum):
    token = 0


_UNSET = _Unset.token


@dataclass(frozen=True)
class PipelineSegment:
    """
    One `|`-delimited sub-URL of a URL pipeline.

    Attributes
    ----------
    scheme : str
        The lowercased URL scheme. Empty string only for a schemeless root
        (a bare local path), which is treated as opaque text.
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

    def __str__(self) -> str:
        return self.raw


@dataclass(frozen=True)
class AdapterResolution:
    """
    The result of resolving a URL pipeline (or a prefix of one).

    Attributes
    ----------
    store : Store
        The resolved store.
    path : str
        Residual path *within* the store that the pipeline addresses
        (e.g. `"path/to/node"` for `...|icechunk://tag.v1/path/to/node`).
        Empty string when the pipeline addresses the store root.
    zarr_format : ZarrFormat | None
        Zarr format selected by a format segment (`zarr2:`/`zarr3:`),
        or None if unspecified. A wrapper adapter that re-wraps a preceding
        resolution must carry every field it does not change forward —
        prefer `dataclasses.replace(preceding, store=..., path=...)` over
        reconstructing, so fields added later are never silently dropped.
    """

    store: Store
    path: str = ""
    zarr_format: ZarrFormat | None = None


@dataclass(frozen=True)
class PipelineContext:
    """
    Context handed to a [`URLPipelineAdapter`][zarr.abc.url_pipeline.URLPipelineAdapter]
    describing the pipeline to the left of its segment.

    Attributes
    ----------
    preceding : tuple[PipelineSegment, ...]
        The parsed sub-URLs to the left of the adapter's segment, outer to
        inner. Empty when the adapter's segment is the pipeline root.
    mode : AccessModeLiteral | None
        The access mode requested by the caller (e.g. `zarr.open(mode=...)`),
        or None when unspecified. Adapters for read-only resources should
        raise for unambiguous write modes (`"w"`, `"w-"`, `"r+"`) and
        open read-only otherwise. `"a"` (the `zarr.open` default) means
        open-or-create: read-only adapters serve the "open" half, and any
        subsequent write fails at the store level.
    storage_options : dict[str, Any] | None
        Options passed by the caller. By convention these configure the
        *root* sub-URL (e.g. fsspec options); adapters may consume
        adapter-specific keys, and should namespace them (e.g.
        `myscheme_credentials`) to avoid collisions with other segments'
        backends. An adapter that consumes keys should strip them before
        resolving the rest of the pipeline, by passing the reduced mapping
        to [`resolve_preceding`][zarr.abc.url_pipeline.PipelineContext.resolve_preceding].
        Non-dict forms of the caller-facing `storage_options` argument are
        reserved for future per-segment configuration (one mapping per
        pipeline segment); this attribute will remain a single mapping —
        the one addressed to this adapter's segment.
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
        [`resolve_preceding`][zarr.abc.url_pipeline.PipelineContext.resolve_preceding]
        takes ownership of the *entire* preceding pipeline: it must
        validate every preceding segment itself and raise
        [`URLPipelineError`][zarr.errors.URLPipelineError] for segments it
        does not understand, so that no segment is ever silently ignored.
        """
        return "|".join(segment.raw for segment in self.preceding)

    async def resolve_preceding(
        self,
        *,
        mode: AccessModeLiteral | _Unset | None = _UNSET,
        storage_options: dict[str, Any] | _Unset | None = _UNSET,
    ) -> AdapterResolution:
        """
        Resolve the preceding pipeline into a store.

        This is the entry point for *wrapper* adapters (e.g. `zip:`) that
        operate on the resource produced by the segments to their left. It
        composes with any preceding adapters, because each segment is
        resolved by its own adapter. Adapters backed by their own I/O
        machinery (e.g. `icechunk:`) may instead consume
        [`preceding_url`][zarr.abc.url_pipeline.PipelineContext.preceding_url]
        and never materialize the intermediate store — subject to the
        ownership contract documented there.

        Parameters
        ----------
        mode : AccessModeLiteral | None, optional
            Override the mode used to resolve the preceding pipeline.
            Wrapper adapters that only read the preceding resource should
            pass `mode="r"` so the root is opened read-only and without
            create-on-open side effects, regardless of the caller's mode.
            When omitted, the caller's mode is used.
        storage_options : dict | None, optional
            Override the options forwarded to the preceding pipeline. An
            adapter that consumed adapter-specific keys should pass the
            remaining mapping here (or None when nothing remains), so the
            root store never sees keys that were not addressed to it.
            When omitted, the caller's options are forwarded unchanged.
        """
        from zarr.storage._url_pipeline import _resolve

        if not self.preceding:
            raise URLPipelineError(
                "this adapter segment is at the pipeline root; "
                "there is no preceding sub-URL to resolve"
            )
        return await _resolve(
            self.preceding,
            mode=self.mode if isinstance(mode, _Unset) else mode,
            storage_options=(
                self.storage_options if isinstance(storage_options, _Unset) else storage_options
            ),
        )


class URLPipelineAdapter(ABC):
    """
    Handler for one URL pipeline scheme.

    Subclasses implement a single classmethod,
    [`open_pipeline_segment`][zarr.abc.url_pipeline.URLPipelineAdapter.open_pipeline_segment],
    and are registered under the `zarr.url_adapters` entry-point group with
    the URL scheme as the entry-point name:

        [project.entry-points."zarr.url_adapters"]
        mypackage.myscheme = "mypackage.zarr_adapter:MyAdapter"

    Nonstandard schemes should be vendor-prefixed (`vendor.scheme`) per the
    URL pipeline specification.

    An adapter is used in two positions:

    - as an *adapter segment*: `s3://bucket/repo|icechunk://tag.v1` — the
      context carries the preceding sub-URLs;
    - as a *root scheme*: `gh://org/repo` — `context.preceding` is empty.
    """

    @classmethod
    @abstractmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> AdapterResolution:
        """
        Resolve `segment` (in the context of the pipeline to its left)
        into a store and an optional residual path within that store.

        The returned store must already be open and must honor
        `context.read_only` (the resolver additionally enforces it by
        downgrading — or rejecting — a writable store when the caller
        required read-only).

        This coroutine runs on zarr's internal I/O event loop. It must not
        block (do I/O through async APIs or a thread executor) and must not
        call zarr's synchronous API (`zarr.open`, `Group.open`, or anything
        else that uses `zarr.core.sync.sync`) — doing so raises
        `SyncError`. To open the preceding pipeline, use
        [`resolve_preceding`][zarr.abc.url_pipeline.PipelineContext.resolve_preceding].
        """
        ...
