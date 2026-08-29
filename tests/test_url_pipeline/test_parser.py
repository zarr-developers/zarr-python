from __future__ import annotations

import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

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
    if sys.platform == "win32":
        assert segment.scheme == ""
        assert segment.body == r"C:\data\store"
    else:
        assert segment.scheme == "c"


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


def test_single_letter_scheme_with_exotic_authority() -> None:
    # single-letter candidates are delegated to parse_store_url, which may
    # reject an exotic authority; the raw scheme is used as the fallback
    segments = parse_pipeline("x://[authority]/path|zip:")
    assert segments[0].scheme in ("", "x")
    assert segments[0].raw == "x://[authority]/path"


def test_fsspec_chained_root_is_opaque() -> None:
    # fsspec's ``scheme::`` chaining is not pipeline syntax; such roots are
    # schemeless/opaque and flow through the ordinary store machinery
    (segment,) = parse_pipeline("zip::file:///tmp/data.zip")
    assert segment.scheme == ""
    assert segment.body == "zip::file:///tmp/data.zip"


def test_whitespace_prefixed_root_is_opaque() -> None:
    # urlparse strips leading whitespace when sniffing a scheme; the parser
    # must not, or the detected scheme desyncs from the body slice
    segments = parse_pipeline("\tfile:/tmp/x|zip:")
    assert segments[0].scheme == ""
    assert segments[0].body == "\tfile:/tmp/x"


def test_schemeless_root_keeps_query_and_fragment_chars() -> None:
    # ? and # are ordinary filename characters in a bare local path
    segments = parse_pipeline("/tmp/d?v=1|zip:")
    assert segments[0].scheme == ""
    assert segments[0].body == "/tmp/d?v=1"
    assert segments[0].query is None
    segments = parse_pipeline("/tmp/d#frag|zip:")
    assert segments[0].body == "/tmp/d#frag"


def test_round_trip() -> None:
    url = "s3://bucket/a.zip?v=2|zip:b/inner.zip|zip:c|zarr3:"
    segments = parse_pipeline(url)
    assert "|".join(s.raw for s in segments) == url


# --- property-based tests -----------------------------------------------
#
# These exercise the parser's *splitting invariants*, not URL pipeline spec
# validity: parse_pipeline is a permissive segment splitter (bodies and
# queries are opaque text to it), and semantic validation is left to the
# resolver and the adapters. Rather than enumerating registered schemes,
# the strategies sample the full scheme grammar the parser accepts
# (RFC 3986 plus "." for vendor prefixes), so every valid root/adapter
# scheme is reachable. Bodies and queries draw from printable ASCII minus
# the characters the parser itself splits on ("|", "#", and "?" for
# bodies); the generated text is not required to be a well-formed URI.

_ADAPTER_SCHEME = st.from_regex(r"[a-zA-Z][a-zA-Z0-9+.\-]{0,15}", fullmatch=True)
# two or more characters, so Windows drive-letter handling cannot reclassify
# the root scheme as a local path on one platform but not another
_ROOT_SCHEME = st.from_regex(r"[a-zA-Z]{2}[a-zA-Z0-9+.\-]{0,14}", fullmatch=True)
_BODY = st.text(st.characters(min_codepoint=32, max_codepoint=126, exclude_characters="|#?"))
# a root body starting with ":" would spell fsspec's chained-URL syntax
# ("scheme::..."), which the parser deliberately treats as an opaque root
_ROOT_BODY = _BODY.filter(lambda body: not body.startswith(":"))
_QUERY = st.text(st.characters(min_codepoint=32, max_codepoint=126, exclude_characters="|#"))


@st.composite
def pipelines(
    draw: st.DrawFn, min_depth: int = 1
) -> tuple[str, list[tuple[str, str, str | None, str]]]:
    """A parseable pipeline URL up to depth 8, with its expected split."""
    depth = draw(st.integers(min_value=min_depth, max_value=8))
    expected = []
    parts = []
    for index in range(depth):
        scheme = draw(_ROOT_SCHEME if index == 0 else _ADAPTER_SCHEME)
        body = draw(_ROOT_BODY if index == 0 else _BODY)
        query = draw(st.none() | _QUERY)
        raw = f"{scheme}:{body}" + (f"?{query}" if query is not None else "")
        parts.append(raw)
        expected.append((scheme.lower(), body, query, raw))
    return "|".join(parts), expected


@given(pipelines())
def test_valid_pipeline_invariants(
    case: tuple[str, list[tuple[str, str, str | None, str]]],
) -> None:
    url, expected = case
    segments = parse_pipeline(url)
    # schemes lowercased; bodies and queries preserved verbatim
    assert [(s.scheme, s.body, s.query, s.raw) for s in segments] == expected
    # lossless round trip
    assert "|".join(s.raw for s in segments) == url


@given(pipelines(), st.data())
def test_property_empty_sub_url_rejected(
    case: tuple[str, list[tuple[str, str, str | None, str]]], data: st.DataObject
) -> None:
    url, _ = case
    parts = url.split("|")
    parts.insert(data.draw(st.integers(0, len(parts))), "")
    with pytest.raises(URLPipelineError, match="empty sub-URL"):
        parse_pipeline("|".join(parts))


@given(pipelines(min_depth=2), st.data())
def test_property_invalid_adapter_scheme_rejected(
    case: tuple[str, list[tuple[str, str, str | None, str]]], data: st.DataObject
) -> None:
    url, _ = case
    parts = url.split("|")
    # corrupt one adapter segment's scheme with a leading character the
    # scheme grammar forbids
    index = data.draw(st.integers(1, len(parts) - 1))
    parts[index] = data.draw(st.sampled_from(["1", " ", "@", "~"])) + parts[index]
    with pytest.raises(URLPipelineError, match="invalid adapter scheme"):
        parse_pipeline("|".join(parts))


@given(pipelines(), st.data())
def test_property_fragment_rejected(
    case: tuple[str, list[tuple[str, str, str | None, str]]], data: st.DataObject
) -> None:
    url, _ = case
    parts = url.split("|")
    parts[data.draw(st.integers(0, len(parts) - 1))] += "#frag"
    with pytest.raises(URLPipelineError, match="fragment"):
        parse_pipeline("|".join(parts))
