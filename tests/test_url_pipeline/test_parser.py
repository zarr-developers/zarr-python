from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zarr.errors import URLPipelineError
from zarr.storage._url_pipeline import _parse_pipeline


def test_single_root_url() -> None:
    (segment,) = _parse_pipeline("s3://bucket/key")
    assert segment.scheme == "s3"
    assert segment.body == "//bucket/key"
    assert segment.query is None
    assert segment.raw == "s3://bucket/key"


@pytest.mark.parametrize("url", ["/local/path", "/local/path|zip:", "relative/path|zip:"])
def test_schemeless_root_rejected(url: str) -> None:
    """The root sub-URL must carry a URL scheme; a bare path (or anything urlparse would not read as a scheme) is an error."""
    with pytest.raises(URLPipelineError, match="URL scheme"):
        _parse_pipeline(url)


def test_adapter_chain() -> None:
    segments = _parse_pipeline("s3://bucket/data.zip|zip:inner/path|zarr3:")
    assert [s.scheme for s in segments] == ["s3", "zip", "zarr3"]
    assert segments[1].body == "inner/path"
    assert segments[2].body == ""


def test_trailing_colon_optional() -> None:
    with_colon = _parse_pipeline("file:/tmp/x.zip|zip:")
    without_colon = _parse_pipeline("file:/tmp/x.zip|zip")
    assert with_colon[1].scheme == without_colon[1].scheme == "zip"
    assert with_colon[1].body == without_colon[1].body == ""


def test_scheme_case_insensitive() -> None:
    segments = _parse_pipeline("FILE:/tmp/x.zip|ZIP:Inner/Path")
    assert segments[0].scheme == "file"
    assert segments[1].scheme == "zip"
    # bodies are case-preserved
    assert segments[1].body == "Inner/Path"


def test_case_preserved_in_raw() -> None:
    # e.g. icechunk snapshot IDs are case-significant
    segments = _parse_pipeline("file:/tmp/repo|icechunk://ABCDEFGH12345678ABCD/x")
    assert segments[1].raw == "icechunk://ABCDEFGH12345678ABCD/x"
    assert segments[1].body == "//ABCDEFGH12345678ABCD/x"


def test_vendor_prefixed_scheme() -> None:
    segments = _parse_pipeline("file:/data|vendor-1.custom+adapter:sub/path")
    assert segments[1].scheme == "vendor-1.custom+adapter"
    assert segments[1].body == "sub/path"


def test_query_strings() -> None:
    segments = _parse_pipeline("https://example.com/d.zip?token=abc|zip:x?opt=1")
    assert segments[0].query == "token=abc"
    assert segments[0].body == "//example.com/d.zip"
    assert segments[1].query == "opt=1"
    assert segments[1].body == "x"


def test_empty_query() -> None:
    (segment,) = _parse_pipeline("https://example.com/d?")
    assert segment.query == ""


@pytest.mark.parametrize(
    "url", [r"C:\data\store", "C:/data/store|zip:", "c:\\x", "c:data", "C:..\\x", "x://host/p"]
)
def test_single_letter_root_scheme_rejected(url: str) -> None:
    """A one-letter root scheme reads as a drive path on Windows, so it is rejected everywhere."""
    with pytest.raises(URLPipelineError, match="Windows drive path"):
        _parse_pipeline(url)


@pytest.mark.parametrize("scheme", ["\u212aelvin", "s\u0131", "zi\u0440"])
def test_non_ascii_scheme_rejected(scheme: str) -> None:
    """Validation happens before lowercasing, so a look-alike cannot alias an ASCII scheme."""
    with pytest.raises(URLPipelineError, match="invalid URL scheme"):
        _parse_pipeline(f"{scheme}://x")
    with pytest.raises(URLPipelineError, match="invalid URL scheme"):
        _parse_pipeline(f"s3://b|{scheme}:x")


def test_segment_limit() -> None:
    _parse_pipeline("memory:" + "|w" * 31)
    with pytest.raises(URLPipelineError, match="more than 32 segments"):
        _parse_pipeline("memory:" + "|w" * 32)


@pytest.mark.parametrize("url", [None, 42, b"s3://b|zip:", ["s3://b"]])
def test_non_string_rejected(url: object) -> None:
    with pytest.raises(TypeError, match="must be a str"):
        _parse_pipeline(url)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "url",
    ["s3://b/\x00a", "s3://b/a\n|zip:", "s3://b|zip:\r\n", "s3://b|zip:\x7f", "s3://b\t|zip:"],
)
def test_control_characters_rejected(url: str) -> None:
    with pytest.raises(URLPipelineError, match="control character"):
        _parse_pipeline(url)


@pytest.mark.parametrize("url", [" s3://b", "s3://b |zip:", "s3://b| zip:", "s3://b|zip:x\u00a0"])
def test_surrounding_whitespace_rejected(url: str) -> None:
    with pytest.raises(URLPipelineError, match="leading or trailing whitespace"):
        _parse_pipeline(url)


def test_interior_whitespace_is_preserved() -> None:
    """zarr does not percent-decode, so a key with a space can only be spelled with one."""
    segments = _parse_pipeline("s3://b/My Data.zip|zip:a b")
    assert segments[0].body == "//b/My Data.zip"
    assert segments[1].body == "a b"


@pytest.mark.parametrize("url", ["a||b:", "|zip:", "file:/tmp|", ""])
def test_empty_sub_url_rejected(url: str) -> None:
    with pytest.raises(URLPipelineError, match="empty sub-URL"):
        _parse_pipeline(url)


def test_fragment_rejected() -> None:
    with pytest.raises(URLPipelineError, match="fragment"):
        _parse_pipeline("file:/tmp/x.zip|zip:inner#frag")


@pytest.mark.parametrize("segment", ["1zip:", "zi p:x", "zip@:x"])
def test_invalid_adapter_scheme_rejected(segment: str) -> None:
    with pytest.raises(URLPipelineError, match="invalid URL scheme"):
        _parse_pipeline(f"file:/tmp/x|{segment}")


def test_scheme_detection_ignores_authority_syntax() -> None:
    """Scheme detection works on the raw text, so an authority urlparse would reject is no obstacle."""
    segments = _parse_pipeline("xy://[authority]/path|zip:")
    assert segments[0].scheme == "xy"
    assert segments[0].raw == "xy://[authority]/path"


def test_round_trip() -> None:
    url = "s3://bucket/a.zip?v=2|zip:b/inner.zip|zip:c|zarr3:"
    segments = _parse_pipeline(url)
    assert "|".join(s.raw for s in segments) == url


# --- property-based tests -----------------------------------------------
#
# These exercise the parser's *splitting invariants*, not URL pipeline spec
# validity: _parse_pipeline is a permissive segment splitter (bodies and
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
# no trailing space: whitespace at either end of a sub-URL is rejected
_BODY = st.text(
    st.characters(min_codepoint=32, max_codepoint=126, exclude_characters="|#?")
).filter(lambda t: not t.endswith(" "))
_QUERY = st.text(
    st.characters(min_codepoint=32, max_codepoint=126, exclude_characters="|#")
).filter(lambda t: not t.endswith(" "))


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
        body = draw(_BODY)
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
    segments = _parse_pipeline(url)
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
        _parse_pipeline("|".join(parts))


@given(pipelines(min_depth=2), st.data())
def test_property_invalid_adapter_scheme_rejected(
    case: tuple[str, list[tuple[str, str, str | None, str]]], data: st.DataObject
) -> None:
    url, _ = case
    parts = url.split("|")
    # corrupt one adapter segment's scheme with a leading character the
    # scheme grammar forbids
    index = data.draw(st.integers(1, len(parts) - 1))
    parts[index] = data.draw(st.sampled_from(["1", "@", "~"])) + parts[index]
    with pytest.raises(URLPipelineError, match="invalid URL scheme"):
        _parse_pipeline("|".join(parts))


@given(pipelines(), st.data())
def test_property_fragment_rejected(
    case: tuple[str, list[tuple[str, str, str | None, str]]], data: st.DataObject
) -> None:
    url, _ = case
    parts = url.split("|")
    parts[data.draw(st.integers(0, len(parts) - 1))] += "#frag"
    with pytest.raises(URLPipelineError, match="fragment"):
        _parse_pipeline("|".join(parts))
