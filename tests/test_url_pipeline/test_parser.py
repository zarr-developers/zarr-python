from __future__ import annotations

import pytest

from zarr.errors import URLPipelineError
from zarr.storage._url_pipeline import parse_pipeline


def test_single_root_url() -> None:
    (segment,) = parse_pipeline("s3://bucket/key")
    assert segment.scheme == "s3"
    assert segment.body == "//bucket/key"
    assert segment.query is None
    assert segment.raw == "s3://bucket/key"
    assert str(segment) == "s3://bucket/key"


def test_schemeless_root() -> None:
    (segment,) = parse_pipeline("/local/path")
    assert segment.scheme == ""
    assert segment.body == "/local/path"
    assert segment.raw == "/local/path"


def test_adapter_chain() -> None:
    segments = parse_pipeline("s3://bucket/data.zip|zip:inner/path|zarr3:")
    assert [s.scheme for s in segments] == ["s3", "zip", "zarr3"]
    assert segments[1].body == "inner/path"
    assert segments[2].body == ""


def test_trailing_colon_optional() -> None:
    with_colon = parse_pipeline("file:/tmp/x.zip|zip:")
    without_colon = parse_pipeline("file:/tmp/x.zip|zip")
    assert with_colon[1].scheme == without_colon[1].scheme == "zip"
    assert with_colon[1].body == without_colon[1].body == ""


def test_scheme_case_insensitive() -> None:
    segments = parse_pipeline("FILE:/tmp/x.zip|ZIP:Inner/Path")
    assert segments[0].scheme == "file"
    assert segments[1].scheme == "zip"
    # bodies are case-preserved
    assert segments[1].body == "Inner/Path"


def test_case_preserved_in_raw() -> None:
    # e.g. icechunk snapshot IDs are case-significant
    segments = parse_pipeline("file:/tmp/repo|icechunk://ABCDEFGH12345678ABCD/x")
    assert segments[1].raw == "icechunk://ABCDEFGH12345678ABCD/x"
    assert segments[1].body == "//ABCDEFGH12345678ABCD/x"


def test_vendor_prefixed_scheme() -> None:
    segments = parse_pipeline("file:/data|vendor-1.custom+adapter:sub/path")
    assert segments[1].scheme == "vendor-1.custom+adapter"
    assert segments[1].body == "sub/path"


def test_query_strings() -> None:
    segments = parse_pipeline("https://example.com/d.zip?token=abc|zip:x?opt=1")
    assert segments[0].query == "token=abc"
    assert segments[0].body == "//example.com/d.zip"
    assert segments[1].query == "opt=1"
    assert segments[1].body == "x"


def test_empty_query() -> None:
    (segment,) = parse_pipeline("https://example.com/d?")
    assert segment.query == ""


def test_windows_drive_path_is_not_a_scheme() -> None:
    # On Windows parse_store_url treats C:\... as a local path; elsewhere the
    # single-letter scheme is preserved but must not crash the parser.
    (segment,) = parse_pipeline(r"C:\data\store")
    assert segment.raw == r"C:\data\store"


@pytest.mark.parametrize("url", ["a||b:", "|zip:", "file:/tmp|", ""])
def test_empty_sub_url_rejected(url: str) -> None:
    with pytest.raises(URLPipelineError, match="empty sub-URL"):
        parse_pipeline(url)


def test_fragment_rejected() -> None:
    with pytest.raises(URLPipelineError, match="fragment"):
        parse_pipeline("file:/tmp/x.zip|zip:inner#frag")


@pytest.mark.parametrize("segment", ["1zip:", "zi p:x", "zip@:x"])
def test_invalid_adapter_scheme_rejected(segment: str) -> None:
    with pytest.raises(URLPipelineError, match="invalid adapter scheme"):
        parse_pipeline(f"file:/tmp/x|{segment}")


def test_round_trip() -> None:
    url = "s3://bucket/a.zip?v=2|zip:b/inner.zip|zip:c|zarr3:"
    segments = parse_pipeline(url)
    assert "|".join(s.raw for s in segments) == url
