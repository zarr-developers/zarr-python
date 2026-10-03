"""
The `zarr:`, `zarr2:` and `zarr3:` URL pipeline adapters.

Per the URL pipeline specification (`schemes/zarr.md`), these address a Zarr
array or group within the `directory` resource to their left:

    file:///path/to/data.zarr/|zarr3:path/to/node

The segment body is a path *within* the preceding resource. `zarr2:` and
`zarr3:` select the Zarr format; `zarr:` leaves it to auto-detection.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar

from zarr.abc.url_pipeline import AdapterResolution, URLPipelineAdapter
from zarr.errors import URLPipelineError
from zarr.storage._utils import _join_paths, normalize_path

if TYPE_CHECKING:
    from zarr.abc.url_pipeline import PipelineContext, PipelineSegment
    from zarr.core.common import ZarrFormat

__all__ = ["Zarr2Adapter", "Zarr3Adapter", "ZarrAdapter"]


class _ZarrFormatAdapter(URLPipelineAdapter):
    """
    Shared implementation of the format adapters.

    Resolves the preceding pipeline with the caller's mode and options, joins
    the segment body onto the preceding residual path, and records the Zarr
    format. Every other field of the preceding resolution is carried forward
    unchanged via `dataclasses.replace`.
    """

    zarr_format: ClassVar[ZarrFormat | None]

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> AdapterResolution:
        """
        Resolve a `zarr:`/`zarr2:`/`zarr3:` segment.

        Parameters
        ----------
        segment : PipelineSegment
            The format segment. Its body is a path within the preceding resource.
        context : PipelineContext
            The pipeline to the left of the segment.

        Returns
        -------
        AdapterResolution
            The preceding resolution with the segment path appended and the
            Zarr format recorded.

        Raises
        ------
        URLPipelineError
            If the segment carries a query, has an invalid path, is the pipeline
            root, or selects a format that conflicts with an earlier format segment.
        """
        if segment.query is not None:
            raise URLPipelineError(
                f"'{segment.scheme}:' pipeline segments do not accept a query: {segment.raw!r}"
            )
        try:
            subpath = normalize_path(segment.body)
        except ValueError as exc:
            raise URLPipelineError(
                f"invalid path in pipeline segment {segment.raw!r}: {exc}"
            ) from exc

        preceding = await context.resolve_preceding()

        zarr_format = cls.zarr_format
        if preceding.zarr_format is not None:
            if zarr_format is None:
                # `zarr:` (auto-detect) never discards a format pinned earlier
                zarr_format = preceding.zarr_format
            elif zarr_format != preceding.zarr_format:
                preceding.store.close()
                raise URLPipelineError(
                    f"pipeline segment {segment.raw!r} selects Zarr format {zarr_format}, "
                    f"but an earlier segment selected Zarr format {preceding.zarr_format}"
                )

        path = _join_paths([normalize_path(preceding.path), subpath])
        return dataclasses.replace(preceding, path=path, zarr_format=zarr_format)


class ZarrAdapter(_ZarrFormatAdapter):
    """
    The `zarr:` adapter: a Zarr node whose format is auto-detected.
    """

    zarr_format = None


class Zarr2Adapter(_ZarrFormatAdapter):
    """
    The `zarr2:` adapter: a Zarr format 2 node.
    """

    zarr_format = 2


class Zarr3Adapter(_ZarrFormatAdapter):
    """
    The `zarr3:` adapter: a Zarr format 3 node.
    """

    zarr_format = 3
