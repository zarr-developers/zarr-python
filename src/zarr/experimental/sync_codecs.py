"""Backwards-compatible alias for SyncCodecPipeline.

The synchronous codec optimizations (inline per-chunk codec chains, thread pool
parallelism, fully synchronous read/write bypass) have been merged into
``BatchedCodecPipeline``. That pipeline now automatically selects the optimal
strategy based on codec and store capabilities — no configuration needed.

``SyncCodecPipeline`` is retained as a subclass alias so that existing config
references (``codec_pipeline.path: zarr.experimental.sync_codecs.SyncCodecPipeline``)
and imports continue to work.
"""

from __future__ import annotations

from dataclasses import dataclass

from zarr.core.codec_pipeline import (
    BatchedCodecPipeline,
    _CODEC_DECODE_NS_PER_BYTE,  # noqa: F401
    _CODEC_ENCODE_NS_PER_BYTE,  # noqa: F401
    _choose_workers,  # noqa: F401
    _estimate_chunk_work_ns,  # noqa: F401
)
from zarr.registry import register_pipeline

__all__ = ["SyncCodecPipeline"]


@dataclass(frozen=True)
class SyncCodecPipeline(BatchedCodecPipeline):
    """Backwards-compatible alias for BatchedCodecPipeline.

    All synchronous codec optimizations are now built into
    ``BatchedCodecPipeline``. This subclass exists only so that
    existing ``codec_pipeline.path`` config values and imports
    continue to work.
    """


register_pipeline(SyncCodecPipeline)
