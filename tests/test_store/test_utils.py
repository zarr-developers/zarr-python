from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from tests.conftest import Expect, ExpectFail
from zarr.storage._utils import ParsedStoreUrl, _dereference_path, parse_store_url


class TestParseStoreUrl:
    """Tests for parse_store_url."""

    def test_memory_url(self) -> None:
        result = parse_store_url("memory://mystore")
        assert result == ParsedStoreUrl(
            scheme="memory", name="mystore", path="", raw="memory://mystore"
        )

    def test_memory_url_with_path(self) -> None:
        result = parse_store_url("memory://mystore/path/to/data")
        assert result == ParsedStoreUrl(
            scheme="memory",
            name="mystore",
            path="path/to/data",
            raw="memory://mystore/path/to/data",
        )

    def test_memory_url_no_name(self) -> None:
        result = parse_store_url("memory://")
        assert result.scheme == "memory"
        assert result.name is None

    def test_s3_url(self) -> None:
        result = parse_store_url("s3://bucket/key")
        assert result == ParsedStoreUrl(
            scheme="s3", name="bucket", path="key", raw="s3://bucket/key"
        )

    def test_file_url(self) -> None:
        result = parse_store_url("file:///tmp/test")
        assert result.scheme == "file"

    def test_local_absolute_path(self) -> None:
        result = parse_store_url("/local/path")
        assert result == ParsedStoreUrl(scheme="", name=None, path="/local/path", raw="/local/path")

    def test_local_relative_path(self) -> None:
        result = parse_store_url("relative/path")
        assert result == ParsedStoreUrl(
            scheme="", name=None, path="relative/path", raw="relative/path"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "C:\\Users\\foo",
            "C:/Users/foo",
            "D:/data/zarr",
            "c:/test",
        ],
    )
    def test_windows_drive_letter(self, url: str) -> None:
        """On Windows, bare drive-letter paths must be treated as local paths."""
        with patch.object(sys, "platform", "win32"):
            result = parse_store_url(url)
        assert result.scheme == ""
        assert result.name is None
        assert result.path == url
        assert result.raw == url

    @pytest.mark.parametrize(
        "url",
        [
            "file:///C:/Users/foo",
            "file://C:/Users/foo",
        ],
    )
    def test_file_url_with_drive_letter_on_windows(self, url: str) -> None:
        """file:// URLs with drive letters are not treated as bare paths."""
        with patch.object(sys, "platform", "win32"):
            result = parse_store_url(url)
        assert result.scheme == "file"

    @pytest.mark.parametrize(
        "url",
        [
            "C:\\Users\\foo",
            "C:/Users/foo",
        ],
    )
    def test_drive_letter_not_special_on_non_windows(self, url: str) -> None:
        """On non-Windows platforms, drive-letter paths go through urlparse."""
        with patch.object(sys, "platform", "linux"):
            result = parse_store_url(url)
        # urlparse interprets the drive letter as a scheme
        assert result.scheme == "c"


@pytest.mark.parametrize(
    "case",
    [
        Expect(input=("root", "path"), output="root/path", id="basic"),
        Expect(input=("root", ""), output="root", id="empty_path"),
        Expect(input=("", "path"), output="path", id="empty_root"),
        Expect(input=("", ""), output="", id="both_empty"),
        Expect(input=("root/", "path"), output="root/path", id="root_trailing_slash"),
        Expect(input=("root//", "path"), output="root/path", id="root_multiple_trailing_slashes"),
        Expect(input=("root", "path/"), output="root/path", id="path_trailing_slash"),
        Expect(input=("a/b", "c/d"), output="a/b/c/d", id="nested"),
        Expect(input=("memory://store", "key"), output="memory://store/key", id="url_root"),
    ],
    ids=lambda case: case.id,
)
def test_dereference_path(case: Expect[tuple[str, str], str]) -> None:
    """
    Test the normal behavior of _dereference_path
    """
    root, path = case.input
    assert _dereference_path(root, path) == case.output


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            input=(1, "path"), exception=TypeError, msg="root=.*not a string", id="root_not_str"
        ),
        ExpectFail(
            input=("root", 1), exception=TypeError, msg="path=.*not a string", id="path_not_str"
        ),
    ],
    ids=lambda case: case.id,
)
def test_dereference_path_errors(case: ExpectFail[tuple[object, object]]) -> None:
    """
    Test that _dereference_path raises TypeError for non-string inputs.
    """
    root, path = case.input
    with pytest.raises(case.exception, match=case.msg):
        _dereference_path(root, path)  # type: ignore[arg-type]
