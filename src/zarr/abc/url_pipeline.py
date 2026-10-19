"""
Abstract base class for URL pipeline adapters.

A URL pipeline is a `|`-separated chain of sub-URLs, read outer-to-inner,
as specified by https://github.com/jbms/url-pipeline. The first sub-URL (the
*root*) locates a resource using a conventional URL, and each subsequent
sub-URL names an *adapter* that reinterprets everything to its left:

    s3://bucket/data.zip|zip:path/inside|zarr3:

Third-party packages provide adapters by subclassing
[`URLPipelineAdapter`][zarr.abc.url_pipeline.URLPipelineAdapter] and
registering the class under the `zarr.url_adapters` entry-point group,
using the URL scheme as the entry-point name. The values an adapter receives
and returns, [`PipelineSegment`][zarr.storage.PipelineSegment],
[`PipelineContext`][zarr.storage.PipelineContext] and
[`StorePath`][zarr.storage.StorePath], live in `zarr.storage`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.storage import PipelineContext, PipelineSegment, StorePath

__all__ = ["URLPipelineAdapter"]


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
    ) -> StorePath:
        """
        Resolve `segment` (in the context of the pipeline to its left)
        into a store and a residual path within that store.

        The returned store must already be open and must honor
        `context.read_only` (the resolver additionally enforces it by
        downgrading — or rejecting — a writable store when the caller
        required read-only). The path of the returned
        [`StorePath`][zarr.storage.StorePath] is the location the pipeline
        addresses inside the store; it is empty for the store root.

        This coroutine runs on zarr's internal I/O event loop. It must not
        block (do I/O through async APIs or a thread executor) and must not
        call zarr's synchronous API (`zarr.open`, `Group.open`, or anything
        else that uses `zarr.core.sync.sync`) — doing so raises
        `SyncError`. To open the preceding pipeline, use
        [`resolve_preceding`][zarr.storage.PipelineContext.resolve_preceding].
        """
        ...
