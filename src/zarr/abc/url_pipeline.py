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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from zarr.abc.store import Store
    from zarr.core.common import AccessModeLiteral

__all__ = [
    "AdapterResolution",
    "PipelineContext",
    "PipelineSegment",
    "URLPipelineAdapter",
]


@dataclass(frozen=True)
class PipelineSegment:
    """
    One `|`-delimited sub-URL of a URL pipeline.

    Attributes
    ----------
    scheme : str
        The lowercased URL scheme. Empty string only for a schemeless root
        (a bare local path).
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
    """

    store: Store
    path: str = ""


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
    read_only : bool
        True when the caller requires a read-only store (`mode == "r"`).
        Adapters must construct their store read-only when this is set;
        when it is False, they may construct a writable store if the
        underlying resource supports writing.
    storage_options : dict[str, Any] | None
        Options passed by the caller. By convention these configure the
        *root* sub-URL (e.g. fsspec options), but adapters may consume
        adapter-specific keys.
    """

    preceding: tuple[PipelineSegment, ...]
    mode: AccessModeLiteral | None
    read_only: bool
    storage_options: dict[str, Any] | None
    _resolver: Callable[[tuple[PipelineSegment, ...]], Awaitable[AdapterResolution]] = field(
        repr=False
    )

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

    async def resolve_preceding(self) -> AdapterResolution:
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
        """
        return await self._resolver(self.preceding)


class URLPipelineAdapter(ABC):
    """
    Handler for one URL pipeline scheme.

    Subclasses implement a single classmethod,
    [`open_pipeline_segment`][zarr.abc.url_pipeline.URLPipelineAdapter.open_pipeline_segment],
    and are registered under the `zarr.url_adapters` entry-point group with
    the URL scheme as the entry-point name:

        [project.entry-points."zarr.url_adapters"]
        myscheme = "mypackage.zarr_adapter:MyAdapter"

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
        `context.read_only`.
        """
        ...
