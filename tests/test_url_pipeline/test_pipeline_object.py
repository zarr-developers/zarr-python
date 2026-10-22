from __future__ import annotations

import pytest

from zarr.errors import URLPipelineError
from zarr.storage import PipelineSegment, URLPipeline
from zarr.storage._url_pipeline import parse_pipeline


@pytest.mark.parametrize(
    ("url", "schemes", "store_schemes", "zarr_format", "path"),
    [
        (
            "s3://bucket/data.zip|zip:inner|zarr3:",
            ("s3", "zip", "zarr3"),
            ("s3", "zip"),
            3,
            "",
        ),
        # the trailing colon of a format segment is optional
        ("file:///data/node.zarr|zarr2", ("file", "zarr2"), ("file",), 2, ""),
        # the body of a format segment is a node path, normalized
        (
            "file:///repo|icechunk:|zarr3:/path/to/array/",
            ("file", "icechunk", "zarr3"),
            ("file", "icechunk"),
            3,
            "path/to/array",
        ),
        # no format segment
        ("memory://x|wrap:a", ("memory", "wrap"), ("memory", "wrap"), None, ""),
        # a single URL is a trivial pipeline
        ("s3://bucket/key", ("s3",), ("s3",), None, ""),
    ],
)
def test_from_url(
    url: str,
    schemes: tuple[str, ...],
    store_schemes: tuple[str, ...],
    zarr_format: int | None,
    path: str,
) -> None:
    """
    `URLPipeline.from_url` holds every parsed segment of the URL, `str()`
    reproduces the URL exactly, and the zarr format, node path and
    store-addressing segments are derived from a trailing `zarr2:`/`zarr3:`
    segment.
    """
    pipeline = URLPipeline.from_url(url)
    assert pipeline.segments == parse_pipeline(url)
    assert tuple(segment.scheme for segment in pipeline.segments) == schemes
    assert tuple(segment.scheme for segment in pipeline._store_segments) == store_schemes
    assert pipeline._zarr_format == zarr_format
    assert pipeline._node_path == path
    assert str(pipeline) == url
    assert URLPipeline(pipeline.segments) == pipeline


def test_format_segment_must_be_last() -> None:
    """A `zarr2:`/`zarr3:` segment anywhere but the end of the pipeline is rejected."""
    with pytest.raises(URLPipelineError, match="must be the last segment"):
        URLPipeline.from_url("file:/data|zarr3:|zip:")


def test_format_segment_rejects_query() -> None:
    """A `zarr2:`/`zarr3:` segment takes a node path only; a query is an error."""
    with pytest.raises(URLPipelineError, match="do not accept a query"):
        URLPipeline.from_url("file:/data|zarr3:x?opt=1")


def test_direct_construction_is_validated() -> None:
    """Validation lives in the constructor, so a hand-built pipeline is checked too."""
    segments = parse_pipeline("file:/data|zarr3:|zip:")
    with pytest.raises(URLPipelineError, match="must be the last segment"):
        URLPipeline(segments)


def test_empty_pipeline_rejected() -> None:
    """A pipeline needs at least a root segment."""
    with pytest.raises(URLPipelineError, match="at least one segment"):
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


@pytest.mark.parametrize(
    ("pipeline_format", "requested", "expected"),
    [(None, None, None), (None, 3, 3), (2, None, 2), (2, 2, 2)],
)
def test_resolve_zarr_format(
    pipeline_format: int | None, requested: int | None, expected: int | None
) -> None:
    """
    `resolve_zarr_format` returns the pipeline's format when the caller did not
    specify one, and the caller's format otherwise.
    """
    suffix = "" if pipeline_format is None else f"|zarr{pipeline_format}:"
    pipeline = URLPipeline.from_url(f"memory://x{suffix}")
    assert pipeline._resolve_zarr_format(requested) == expected  # type: ignore[arg-type]


def test_resolve_zarr_format_conflict_raises() -> None:
    """A caller-supplied format that differs from the pipeline's segment is an error."""
    pipeline = URLPipeline.from_url("memory://x|zarr2:")
    with pytest.raises(ValueError, match="conflicts with the 'zarr2:' segment"):
        pipeline._resolve_zarr_format(3)
