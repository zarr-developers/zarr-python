"""
Abstract base class for URL pipeline adapters
(https://github.com/jbms/url-pipeline). See
[`URLPipelineAdapter`][zarr.abc.url_pipeline.URLPipelineAdapter].
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

    In a pipeline such as `s3://bucket/data.zip|zip:inner|zarr3:`, each sub-URL
    after the first names an adapter that reinterprets everything to its left.
    Subclasses implement
    [`open_pipeline_segment`][zarr.abc.url_pipeline.URLPipelineAdapter.open_pipeline_segment]
    and are registered under the `zarr.url_adapters` entry-point group, with
    the URL scheme as the entry-point name:

        [project.entry-points."zarr.url_adapters"]
        "mypackage.myscheme" = "mypackage.zarr_adapter:MyAdapter"

    Nonstandard schemes should be vendor-prefixed (`vendor.scheme`), per the
    URL pipeline specification. An adapter is used either as an *adapter
    segment* (`s3://bucket/repo|icechunk://tag.v1`), where the context carries
    the preceding sub-URLs, or as a *root scheme* (`gh://org/repo`), where
    `context.preceding` is empty.
    """

    @classmethod
    @abstractmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> StorePath:
        """
        Resolve `segment`, in the context of the pipeline to its left, into an
        open store and the path the pipeline addresses inside it (empty for
        the store root).

        The store must honor `context.read_only`; the resolver also enforces
        it. This coroutine runs on zarr's I/O event loop, so it must not
        block, and must not call zarr's synchronous API (which raises
        `SyncError`). Open the preceding pipeline with
        [`resolve_preceding`][zarr.storage.PipelineContext.resolve_preceding].
        """
        ...
