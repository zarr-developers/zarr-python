from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from zarr.abc.store import SuffixByteRequest
from zarr.core.buffer.core import default_buffer_prototype
from zarr.storage._utils import ParsedStoreUrl, _normalize_byte_range_index, parse_store_url

if TYPE_CHECKING:
    from pathlib import Path


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

    @pytest.mark.parametrize(
        ("url", "expected_path"),
        [
            ("file:///tmp/test", "/tmp/test"),
            ("file:///a/b", "/a/b"),
            ("file:///a/b/", "/a/b"),
            ("file:///", "/"),
        ],
    )
    def test_absolute_file_url(self, url: str, expected_path: str) -> None:
        """file:///a/b (empty netloc) is an absolute path."""
        result = parse_store_url(url)
        assert result == ParsedStoreUrl(scheme="file", name=None, path=expected_path, raw=url)

    @pytest.mark.parametrize(
        ("url", "relative_path"),
        [
            ("file://a/b", "a/b"),
            ("file://a/b/", "a/b"),
            ("file://a", "a"),
            ("file://./a/b", "a/b"),
            ("file://", ""),
        ],
    )
    def test_relative_file_url(
        self, url: str, relative_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """file://a/b is a path relative to the cwd, not a host named "a"."""
        monkeypatch.chdir(tmp_path)
        result = parse_store_url(url)
        assert result == ParsedStoreUrl(
            scheme="file", name=None, path=str(tmp_path / relative_path), raw=url
        )

    @pytest.mark.parametrize(
        "url",
        [
            "file:///a/b?query=1",
            "file:///a/b#fragment",
        ],
    )
    def test_file_url_rejects_params_query_fragment(self, url: str) -> None:
        with pytest.raises(ValueError, match="not supported in file:// URLs"):
            parse_store_url(url)

    def test_local_absolute_path(self) -> None:
        result = parse_store_url("/local/path")
        assert result == ParsedStoreUrl(scheme="", name=None, path="/local/path", raw="/local/path")

    def test_local_relative_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = parse_store_url("relative/path")
        assert result == ParsedStoreUrl(
            scheme="", name=None, path=str(tmp_path / "relative/path"), raw="relative/path"
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


class TestNormalizeByteRangeIndex:
    """Tests for _normalize_byte_range_index."""

    def test_suffix_larger_than_data_returns_all_bytes(self) -> None:
        """Regression: SuffixByteRequest with suffix > len(data) must not produce a
        negative start index that causes numpy to return fewer bytes than available."""
        prototype = default_buffer_prototype()
        data = prototype.buffer.from_bytes(b"hello")  # 5 bytes
        byte_range = SuffixByteRequest(suffix=7)
        start, stop = _normalize_byte_range_index(data, byte_range)
        assert start == 0, f"start should be 0 (clamped), got {start}"
        result = data[start:stop]
        assert len(result) == 5, f"expected all 5 bytes, got {len(result)}"

    def test_suffix_exact_length(self) -> None:
        """SuffixByteRequest with suffix == len(data) returns all bytes."""
        prototype = default_buffer_prototype()
        data = prototype.buffer.from_bytes(b"hello")
        start, _stop = _normalize_byte_range_index(data, SuffixByteRequest(suffix=5))
        assert start == 0

    def test_suffix_shorter_than_data(self) -> None:
        """SuffixByteRequest with suffix < len(data) returns the last n bytes."""
        prototype = default_buffer_prototype()
        data = prototype.buffer.from_bytes(b"hello")
        start, _stop = _normalize_byte_range_index(data, SuffixByteRequest(suffix=3))
        assert start == 2
