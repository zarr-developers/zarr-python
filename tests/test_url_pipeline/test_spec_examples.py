"""
Conformance tests against the URL pipeline specification.

Every entry in `SPEC_EXAMPLES` is an example URL from the specification
repository (https://github.com/jbms/url-pipeline @ 1a01ce6), extracted
with the same rule as the spec's own linter (`scripts/lint.py`): backticked
examples on bullet lines, skipping spans without a `:` or `|`. The spec
validates each example (and an uppercased-scheme variant) against its ABNF
grammar, so this corpus is grammar-valid by construction.

The parser must accept every example — including schemes zarr-python has no
adapter for, since parsing is independent of adapter availability — split it
into the expected sub-URL schemes, and preserve the text losslessly.

To regenerate after a spec update: extract examples per the rule above and
recompute the expected scheme tuples with `parse_pipeline`, reviewing any
changes against the spec diff.
"""

from __future__ import annotations

import re

import pytest

from zarr.storage._url_pipeline import parse_pipeline

SPEC_EXAMPLES: list[tuple[str, tuple[str, ...]]] = [
    # README.md
    ("s3://bucket/path/to/archive.zip|zip:path/within/zip.zarr/|zarr3:", ("s3", "zip", "zarr3")),
    (
        "file:///tmp/dataset.ocdbt/|ocdbt://2025-01-01T01:23:45.678Z/path/within/database",
        ("file", "ocdbt"),
    ),
    (
        "s3+https://example.com/path/to/database.icechunk/|icechunk://tag.v5/path/to/node/|zarr3:",
        ("s3+https", "icechunk", "zarr3"),
    ),
    (
        "vendor1-2.custom-1+proto.ext://[authority]/path?query/part?x",
        ("vendor1-2.custom-1+proto.ext",),
    ),
    (
        "http://example.com/file|vendor1-2.custom+adapter:/path/within/adapter",
        ("http", "vendor1-2.custom+adapter"),
    ),
    ("a.b:?", ("a.b",)),
    ("http://somehost/downloads/somefile.zip|zip:", ("http", "zip")),
    ("http://example.com/archive.jar|zip:path/to/file.txt", ("http", "zip")),
    ("https://host/archive.zip|zip:path/in/outer.zip|zip:path/in/inner", ("https", "zip", "zip")),
    ("file:///path/to/archive.zip|zip:path/within/archive", ("file", "zip")),
    # avif.md
    ("file:/path/to/image.avif|avif:", ("file", "avif")),
    ("file:/path/to/image.avif|avif", ("file", "avif")),
    # bmp.md
    ("file:/path/to/image.bmp|bmp:", ("file", "bmp")),
    ("file:/path/to/image.bmp|bmp", ("file", "bmp")),
    # byte-range.md
    ("file:/path/to/data|byte-range:1000-2000", ("file", "byte-range")),
    ("file:/path/to/data|byte-range:0-1", ("file", "byte-range")),
    # file.md
    ("file:/", ("file",)),
    ("file:/a", ("file",)),
    ("file:/path/to/file.txt", ("file",)),
    ("file://localhost/path/to/file.txt", ("file",)),
    ("file://LOCALHOST/path/to/file.txt", ("file",)),
    ("file:///path/to/file.txt", ("file",)),
    ("file://somehost/sharename/path/to/file.txt", ("file",)),
    # gs.md
    ("gs://bucket", ("gs",)),
    ("gs://bucket/", ("gs",)),
    ("gs://bucket/path/within/bucket", ("gs",)),
    ("gs://aaa", ("gs",)),
    (
        "gs://label1-is-sixty-two-characters-long-xxxxxxxxxxxxxxxxxxxxxxxxxx.label2-is-sixty-two-characters-long-yyyyyyyyyyyyyyyyyyyyyyyyyy.label3-is-sixty-two-characters-long-zzzzzzzzzzzzzzzzzzzzzzzzzz.label4-is-thirty-one-characters",
        ("gs",),
    ),
    # gzip.md
    ("file:/path/to/data.gz|gzip:", ("file", "gzip")),
    ("file:/path/to/data.gz|gzip", ("file", "gzip")),
    # hdf5.md
    ("s3://bucket/path/to/file.h5|hdf5:/path/to/dataset", ("s3", "hdf5")),
    ("s3://bucket/path/to/file.h5|hdf5:a", ("s3", "hdf5")),
    ("file:///path/to/file.h5|hdf5:", ("file", "hdf5")),
    ("file:///path/to/file.h5|hdf5", ("file", "hdf5")),
    # http.md
    ("https://example.com/path/to/resource%20name", ("https",)),
    ("https://example.com/path/to/resource?query=value", ("https",)),
    ("https://example.com/path/to/?query=value", ("https",)),
    ("https://example.com/path/with:colon", ("https",)),
    ("https://server.example.com:1234/path/to/array", ("https",)),
    ("http://a", ("http",)),
    ("http://example.com", ("http",)),
    ("http://local%68ost", ("http",)),
    ("http://example.com:", ("http",)),
    ("http://192.168.10.1:1234", ("http",)),
    ("http://[::1]", ("http",)),
    ("http://[::ffff:127.0.0.1]", ("http",)),
    ("http://example.com/", ("http",)),
    # icechunk.md
    ("file:///path/to/repo.zarr.icechunk/|icechunk:", ("file", "icechunk")),
    ("file:///path/to/repo.zarr.icechunk/|icechunk", ("file", "icechunk")),
    ("file:///path/to/repo.zarr.icechunk/|icechunk://branch.main/", ("file", "icechunk")),
    ("file:///path/to/repo.zarr.icechunk/|icechunk:path/to/node/", ("file", "icechunk")),
    ("file:///path/to/repo.zarr.icechunk/|icechunk:a", ("file", "icechunk")),
    ("file:///path/to/repo.zarr.icechunk/|icechunk:/path/to/node/", ("file", "icechunk")),
    ("file:///path/to/repo.zarr.icechunk/|icechunk:path/to/node/zarr.json", ("file", "icechunk")),
    ("file:///path/to/repo.zarr.icechunk/|icechunk:path/to/node/c/0/0/1", ("file", "icechunk")),
    (
        "file:///path/to/repo.zarr.icechunk/|icechunk://branch.mybranch/path/to/node/",
        ("file", "icechunk"),
    ),
    ("file:///path/to/repo.zarr.icechunk/|icechunk://branch.a", ("file", "icechunk")),
    (
        "file:///path/to/repo.zarr.icechunk/|icechunk://tag.mytag/path/to/node/",
        ("file", "icechunk"),
    ),
    ("file:///path/to/repo.zarr.icechunk/|icechunk://tag.a", ("file", "icechunk")),
    (
        "file:///path/to/repo.zarr.icechunk/|icechunk://FWWFQGAW742XMX0F5MF0/path/to/node/",
        ("file", "icechunk"),
    ),
    (
        "file:///path/to/repo.zarr.icechunk/|icechunk://ABCDEFGHJKMNPQRSTVWX/path/to/node/",
        ("file", "icechunk"),
    ),
    (
        "file:///path/to/repo.zarr.icechunk/|icechunk:|zarr3:path/to/array/",
        ("file", "icechunk", "zarr3"),
    ),
    (
        "file:///path/to/repo.zarr.icechunk/|icechunk://branch.other/|zarr3:path/to/array/",
        ("file", "icechunk", "zarr3"),
    ),
    (
        "file:///path/to/repo.zarr.icechunk/|icechunk://tag.v5/|zarr3:path/to/array/",
        ("file", "icechunk", "zarr3"),
    ),
    (
        "file:///path/to/repo.zarr.icechunk/|icechunk://4N0217AZA4VNPYD0HR0G/|zarr3:path/to/array/",
        ("file", "icechunk", "zarr3"),
    ),
    # jpeg.md
    ("file:/path/to/image.jpeg|jpeg:", ("file", "jpeg")),
    ("file:/path/to/image.jpeg|jpeg", ("file", "jpeg")),
    # json.md
    ("file:/path/to/data.json|json:", ("file", "json")),
    ("file:/path/to/data.json|json", ("file", "json")),
    ("file:/path/to/data.json|json:/path/to/node", ("file", "json")),
    ("file:/path/to/data.json|json:/abc~0def", ("file", "json")),
    ("file:/path/to/data.json|json:/abc~1def", ("file", "json")),
    ("file:/path/to/data.json|json:/", ("file", "json")),
    # memory.md
    ("memory:", ("memory",)),
    ("memory:/", ("memory",)),
    ("memory://", ("memory",)),
    ("memory:path/to/resource", ("memory",)),
    ("memory:a", ("memory",)),
    ("memory:another@path+with,lots&of(special;characters)*_!-$'", ("memory",)),
    ("memory:/path/to/resource", ("memory",)),
    ("memory://path/to/resource", ("memory",)),
    # n5.md
    ("file:///tmp/data.n5/|n5:path/to/array", ("file", "n5")),
    ("file:///tmp/data.n5/|n5:a", ("file", "n5")),
    ("file:///tmp/data.n5/|n5:/path/to/array", ("file", "n5")),
    ("file:///tmp/data.n5/|n5:", ("file", "n5")),
    ("file:///tmp/data.n5/|n5", ("file", "n5")),
    # neuroglancer-precomputed.md
    (
        "file:///tmp/dataset.precomputed/|neuroglancer-precomputed:",
        ("file", "neuroglancer-precomputed"),
    ),
    (
        "file:///tmp/dataset.precomputed/|neuroglancer-precomputed",
        ("file", "neuroglancer-precomputed"),
    ),
    # ocdbt.md
    ("file:///path/to/repo.ocdbt/|ocdbt:", ("file", "ocdbt")),
    ("file:///path/to/repo.ocdbt/|ocdbt", ("file", "ocdbt")),
    ("file:///path/to/repo.ocdbt/|ocdbt:path/within/repo/", ("file", "ocdbt")),
    ("file:///path/to/repo.ocdbt/|ocdbt:path/within/repo", ("file", "ocdbt")),
    ("file:///path/to/repo.ocdbt/|ocdbt:a", ("file", "ocdbt")),
    ("file:///path/to/repo.ocdbt/|ocdbt:/path/within/repo", ("file", "ocdbt")),
    ("file:///path/to/repo.ocdbt/|ocdbt://v123/", ("file", "ocdbt")),
    ("file:///path/to/repo.ocdbt/|ocdbt://v1", ("file", "ocdbt")),
    ("file:///path/to/repo.ocdbt/|ocdbt://v123/path/within/repo", ("file", "ocdbt")),
    (
        "file:///path/to/repo.ocdbt/|ocdbt://2025-01-01T01:23:45.678Z/path/within/database",
        ("file", "ocdbt"),
    ),
    ("file:///path/to/repo.ocdbt/|ocdbt://2025-01-01T01:23:45Z", ("file", "ocdbt")),
    ("file:///path/to/repo.ocdbt/|ocdbt://2025-01-01T01:23:45.1Z", ("file", "ocdbt")),
    ("file:///path/to/repo.ocdbt/|ocdbt://2025-01-01T01:23:45.123456789Z", ("file", "ocdbt")),
    # png.md
    ("file:/path/to/image.png|png:", ("file", "png")),
    ("file:/path/to/image.png|png", ("file", "png")),
    # s3+http.md
    ("s3+https://endpoint/path/within/bucket", ("s3+https",)),
    ("s3+https://endpoint/bucket/path/within/bucket", ("s3+https",)),
    ("s3+https://mybucket.s3.amazonaws.com/path/to/file", ("s3+https",)),
    ("s3+https://s3.amazonaws.com/mybucket/path/to/file", ("s3+https",)),
    ("s3+http://example.com", ("s3+http",)),
    ("s3+http://example.com/", ("s3+http",)),
    # s3.md
    ("s3://bucket", ("s3",)),
    ("s3://bucket/", ("s3",)),
    ("s3://bucket/path/within/bucket", ("s3",)),
    ("s3://aaa", ("s3",)),
    (
        "s3://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ("s3",),
    ),
    # tiff.md
    ("file:/path/to/image.tiff|tiff:", ("file", "tiff")),
    ("file:/path/to/image.tiff|tiff", ("file", "tiff")),
    # webp.md
    ("file:/path/to/image.webp|webp:", ("file", "webp")),
    ("file:/path/to/image.webp|webp", ("file", "webp")),
    # zarr.md
    ("file:///path/to/node.zarr/|zarr3:", ("file", "zarr3")),
    ("file:///path/to/node.zarr/|zarr3", ("file", "zarr3")),
    ("file:///path/to/node.zarr/|zarr2:", ("file", "zarr2")),
    ("file:///path/to/node.zarr/|zarr2", ("file", "zarr2")),
    ("file:///path/to/node.zarr/|zarr:", ("file", "zarr")),
    ("file:///path/to/node.zarr/|zarr", ("file", "zarr")),
    # zip.md
    ("file:/path/to/archive.zip|zip:path/to/file.txt", ("file", "zip")),
    ("file:/path/to/archive.zip|zip", ("file", "zip")),
    ("file:/path/to/archive.zip|zip:/path/to/file.txt", ("file", "zip")),
    ("file:/path/to/outer.zip|zip:path/to/inner.zip|zip:path/to/file.txt", ("file", "zip", "zip")),
    # zstd.md
    ("file:/path/to/data.zstd|zstd:", ("file", "zstd")),
    ("file:/path/to/data.zstd|zstd", ("file", "zstd")),
]


def _uppercase_schemes(example: str) -> str:
    """Uppercase every sub-URL scheme, as the spec's linter does."""

    def repl(match: re.Match[str]) -> str:
        return match.group(1) + match.group(2).upper()

    return re.sub(r"((?:^|\|)\s*)([a-zA-Z][a-zA-Z0-9+.-]*)", repl, example)


@pytest.mark.parametrize(("url", "expected_schemes"), SPEC_EXAMPLES, ids=lambda v: str(v))
def test_spec_example_parses(url: str, expected_schemes: tuple[str, ...]) -> None:
    segments = parse_pipeline(url)
    assert tuple(segment.scheme for segment in segments) == expected_schemes
    # lossless round-trip of the original text
    assert "|".join(segment.raw for segment in segments) == url


@pytest.mark.parametrize(("url", "expected_schemes"), SPEC_EXAMPLES, ids=lambda v: str(v))
def test_spec_example_schemes_case_insensitive(url: str, expected_schemes: tuple[str, ...]) -> None:
    upper = _uppercase_schemes(url)
    segments = parse_pipeline(upper)
    assert tuple(segment.scheme for segment in segments) == expected_schemes
