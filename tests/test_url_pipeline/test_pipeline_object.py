from __future__ import annotations

import pytest

from zarr.errors import URLPipelineError
from zarr.storage import PipelineSegment, URLPipeline
from zarr.storage._url_pipeline import _parse_pipeline


@pytest.mark.parametrize(
    "url",
    [
        "s3://bucket/data.zip|zip:inner|zarr3:",
        "file:///data/node.zarr|zarr2",
        "file:///repo|icechunk:|zarr3:/path/to/array/",
        "memory://x|wrap:a",
        "s3://bucket/key",
    ],
)
def test_from_url(url: str) -> None:
    """`from_url` holds every parsed segment; `str()` reproduces the URL exactly."""
    pipeline = URLPipeline.from_url(url)
    assert pipeline.segments == _parse_pipeline(url)
    assert str(pipeline) == url
    assert URLPipeline(pipeline.segments) == pipeline


def test_format_segment_rejects_query() -> None:
    """A `zarr2:`/`zarr3:` segment takes a node path only; a query is an error."""
    with pytest.raises(URLPipelineError, match="do not accept a query"):
        URLPipeline.from_url("file:/data|zarr3:x?opt=1")


def test_direct_construction_is_validated() -> None:
    """Validation lives in the constructor, so a hand-built pipeline is checked too."""
    segments = _parse_pipeline("file:/data|zarr3:|zip:")
    with pytest.raises(URLPipelineError, match="must be the last segment"):
        URLPipeline(segments)


def test_empty_pipeline_rejected() -> None:
    """A pipeline needs at least a root segment."""
    with pytest.raises(URLPipelineError, match="1 to 32 segments"):
        URLPipeline(())


@pytest.mark.parametrize(
    "segments",
    ["s3://b|zip:", ["s3://b"], ("s3://b",), [PipelineSegment("s3", "//b", None, "s3://b")]],
)
def test_direct_construction_rejects_non_segments(segments: object) -> None:
    """A string or a list is not a pipeline; `from_url` is the way to parse one."""
    with pytest.raises(TypeError, match="tuple of PipelineSegment"):
        URLPipeline(segments)  # type: ignore[arg-type]


@pytest.mark.parametrize("url", [None, b"s3://b", URLPipeline.from_url("s3://b")])
def test_from_url_rejects_non_strings(url: object) -> None:
    with pytest.raises(TypeError, match="must be a str"):
        URLPipeline.from_url(url)  # type: ignore[arg-type]


def test_hashable() -> None:
    """Pipelines are values: equal pipelines hash equal and can key a dict."""
    a = URLPipeline.from_url("s3://b/d.zip|zip:|zarr3:")
    b = URLPipeline.from_url("s3://b/d.zip|zip:|zarr3:")
    assert a == b
    assert hash(a) == hash(b)
    assert {a: 1}[b] == 1
