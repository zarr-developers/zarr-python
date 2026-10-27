from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zarr.errors import URLPipelineError
from zarr.storage import PipelineSegment, URLPipeline
from zarr.storage._url_pipeline import _parse_pipeline


@pytest.mark.parametrize(
    ("url", "canonical"),
    [
        ("s3://bucket/data.zip|zip:inner|zarr3:", "s3://bucket/data.zip|zip:inner|zarr3:"),
        # the optional colon is always written, and schemes are lowercased
        ("FILE:///data/node.zarr|zarr2", "file:///data/node.zarr|zarr2:"),
        (
            "file:///repo|icechunk:|zarr3:/path/to/array/",
            "file:///repo|icechunk:|zarr3:/path/to/array/",
        ),
        ("memory://x|wrap:a", "memory://x|wrap:a"),
        ("s3://bucket/key", "s3://bucket/key"),
    ],
)
def test_from_url(url: str, canonical: str) -> None:
    """`from_url` holds every parsed segment; `str()` is the canonical URL, which re-parses equal."""
    pipeline = URLPipeline.from_url(url)
    assert pipeline.segments == _parse_pipeline(url)
    assert str(pipeline) == canonical
    assert URLPipeline.from_url(str(pipeline)) == pipeline
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
    [
        "s3://b|zip:",
        ["s3://b"],
        ("s3://b",),
        [PipelineSegment(scheme="s3", body="//b", query=None)],
    ],
)
def test_direct_construction_rejects_non_segments(segments: object) -> None:
    """Only a tuple of `PipelineSegment` is accepted; `from_url` parses strings."""
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


# --- round-trip properties ---------------------------------------------
#
# Every pipeline, hand-built or parsed, must satisfy
# `URLPipeline.from_url(str(p)) == p`: otherwise `str()` could name a different
# pipeline, e.g. a hand-built body "/tmp/a|evil:" re-parsing as an 'evil'
# adapter. Parts are drawn with the characters the parser treats specially.

_NASTY = st.characters(min_codepoint=0, max_codepoint=0x7F) | st.sampled_from(
    ["|", "#", "?", ":", "/", "\\", " ", "\t", "\n", "\u212a", "\u0131"]
)
_SCHEME = st.from_regex(r"[a-z][a-z0-9+.\-]{0,7}", fullmatch=True) | st.text(_NASTY, max_size=4)
_PART = st.text(_NASTY, max_size=12)


@st.composite
def _hand_built(draw: st.DrawFn) -> URLPipeline:
    segments = []
    for _ in range(draw(st.integers(1, 4))):
        try:
            segments.append(
                PipelineSegment(
                    scheme=draw(_SCHEME), body=draw(_PART), query=draw(st.none() | _PART)
                )
            )
        except URLPipelineError:
            continue
    try:
        return URLPipeline(tuple(segments))
    except URLPipelineError:
        return draw(st.nothing())


@given(_hand_built())
def test_property_hand_built_pipeline_round_trips(pipeline: URLPipeline) -> None:
    assert URLPipeline.from_url(str(pipeline)) == pipeline


@given(st.text(_NASTY, max_size=40))
def test_property_parsed_pipeline_round_trips(url: str) -> None:
    try:
        pipeline = URLPipeline.from_url(url)
    except URLPipelineError:
        return
    assert URLPipeline.from_url(str(pipeline)) == pipeline
