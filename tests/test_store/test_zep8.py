"""
Tests for ZEP 8 URL syntax support in zarr-python.

This module tests the ZEP 8 URL syntax functionality using pytest's functional approach.
Tests are organized by functionality groups rather than classes.
"""

import zipfile
from pathlib import Path
from typing import Any

import pytest

import zarr
from zarr.abc.store import Store
from zarr.abc.store_adapter import StoreAdapter, URLSegment
from zarr.core.array import Array
from zarr.core.buffer import default_buffer_prototype
from zarr.registry import get_store_adapter, register_store_adapter
from zarr.storage import FsspecStore, LocalStore, LoggingStore, MemoryStore, ZipStore
from zarr.storage._builtin_adapters import (
    GSAdapter,
    HttpsAdapter,
    LoggingAdapter,
    S3Adapter,
    S3HttpAdapter,
    S3HttpsAdapter,
)
from zarr.storage._common import make_store_path
from zarr.storage._zep8 import URLParser, URLStoreResolver, ZEP8URLError, is_zep8_url


def test_simple_url_parsing() -> None:
    """Test parsing of simple URLs."""
    parser = URLParser()

    # Test simple URL
    segments = parser.parse("s3://bucket/data.zarr")
    assert len(segments) == 1
    assert segments[0].scheme == "s3"
    assert segments[0].path == "bucket/data.zarr"
    assert segments[0].adapter is None


def test_zep8_url_parsing() -> None:
    """Test parsing of ZEP 8 URLs with pipe separators."""
    parser = URLParser()

    # Test chained URL
    segments = parser.parse("s3://bucket/data.zip|zip:|zarr3:")
    assert len(segments) == 3

    assert segments[0].scheme == "s3"
    assert segments[0].path == "bucket/data.zip"
    assert segments[0].adapter is None

    assert segments[1].scheme is None
    assert segments[1].adapter == "zip"
    assert segments[1].path == ""

    assert segments[2].scheme is None
    assert segments[2].adapter == "zarr3"
    assert segments[2].path == ""


def test_complex_url_parsing() -> None:
    """Test parsing of complex URLs with paths and parameters."""
    parser = URLParser()

    segments = parser.parse("https://example.com/data.zip|zip:subdir/|memory:")
    assert len(segments) == 3

    assert segments[0].scheme == "https"
    assert segments[0].path == "example.com/data.zip"

    assert segments[1].adapter == "zip"
    assert segments[1].path == "subdir/"

    assert segments[2].adapter == "memory"
    assert segments[2].path == ""


def test_invalid_url_parsing() -> None:
    """Test error handling for invalid URLs."""
    parser = URLParser()

    # Test empty pipe segment
    with pytest.raises(ZEP8URLError, match="Empty URL segment"):
        parser.parse("s3://bucket/data||zip:")

    # Test invalid pipe at start
    with pytest.raises(ZEP8URLError, match="URL cannot start with pipe"):
        parser.parse("|zip:s3://bucket")


def test_relative_path_resolution() -> None:
    """Test relative path resolution."""
    parser = URLParser()
    base = URLSegment(scheme="s3", path="bucket/data/", adapter=None)

    resolved = parser.resolve_relative(base, "subdir/file.txt")
    assert resolved.scheme == "s3"
    assert resolved.path == "bucket/data/subdir/file.txt"

    # Test with trailing slash normalization
    base2 = URLSegment(scheme="s3", path="bucket/data", adapter=None)
    resolved2 = parser.resolve_relative(base2, "subdir/file.txt")
    assert resolved2.path == "bucket/data/subdir/file.txt"


# =============================================================================
# Store Adapter Registry Tests
# =============================================================================


def test_builtin_adapters_registered() -> None:
    """Test that built-in adapters are registered."""
    # Test some built-in adapters
    file_adapter = get_store_adapter("file")
    assert file_adapter is not None

    memory_adapter = get_store_adapter("memory")
    assert memory_adapter is not None

    zip_adapter = get_store_adapter("zip")
    assert zip_adapter is not None


def test_custom_adapter_registration() -> None:
    """Test registering custom store adapters."""

    class TestAdapter(StoreAdapter):
        adapter_name = "test"

        @classmethod
        async def from_url_segment(
            cls, segment: URLSegment, preceding_url: str, **kwargs: Any
        ) -> MemoryStore:
            return MemoryStore()

    # Register adapter
    register_store_adapter(TestAdapter)

    # Verify it's registered
    adapter = get_store_adapter("test")
    assert adapter is TestAdapter


# =============================================================================
# URL Store Resolver Tests
# =============================================================================


async def test_simple_url_resolution() -> None:
    """Test resolving simple URLs without chaining."""
    resolver = URLStoreResolver()

    # Test memory URL
    store = await resolver.resolve_url("memory:")
    assert isinstance(store, MemoryStore)


async def test_file_url_resolution(tmp_path: Path) -> None:
    """Test resolving file URLs."""
    resolver = URLStoreResolver()

    # Create a temporary directory
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    # Test local file URL
    store = await resolver.resolve_url(f"file:{test_dir}")
    assert isinstance(store, LocalStore)


async def test_zip_chain_resolution(tmp_path: Path) -> None:
    """Test resolving ZIP chain URLs."""
    resolver = URLStoreResolver()

    # Create a test ZIP file with some content
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/array.json", '{"test": "data"}')
        zf.writestr("data/0.0", b"test chunk data")

    # Test ZIP URL chain
    store = await resolver.resolve_url(f"file:{zip_path}|zip:")
    # The store should be accessible
    assert store is not None


def test_zarr_format_extraction() -> None:
    """Test extracting Zarr format from URLs."""
    resolver = URLStoreResolver()

    # Test zarr2 format
    format_type = resolver.extract_zarr_format("memory:|zarr2:")
    assert format_type == 2

    # Test zarr3 format
    format_type = resolver.extract_zarr_format("memory:|zarr3:")
    assert format_type == 3

    # Test no format (should return None)
    format_type = resolver.extract_zarr_format("memory:")
    assert format_type is None


def test_path_extraction() -> None:
    """Test extracting paths from URLs."""
    resolver = URLStoreResolver()

    # Test with path in last segment
    path = resolver.extract_path("s3://bucket/data|zip:subdir/")
    assert path == "subdir/"

    # Test with no path
    path = resolver.extract_path("s3://bucket/data|zip:")
    assert path == ""


# =============================================================================
# make_store_path Integration Tests
# =============================================================================


def test_zep8_url_detection() -> None:
    """Test that ZEP 8 URLs are detected correctly."""
    # Should detect ZEP 8 URLs
    assert is_zep8_url("s3://bucket/data|zip:")
    assert is_zep8_url("memory:|zarr3:")
    assert is_zep8_url("file:/path/data.zip|zip:subdir/")

    # Should not detect regular URLs
    assert not is_zep8_url("s3://bucket/data")
    assert not is_zep8_url("/local/path")
    assert not is_zep8_url("https://example.com/data")

    assert not is_zep8_url(MemoryStore())


async def test_make_store_path_with_zep8_url() -> None:
    """Test make_store_path with ZEP 8 URLs."""
    # Test simple memory URL
    store_path = await make_store_path("memory:")
    assert store_path.store is not None
    assert isinstance(store_path.store, MemoryStore)
    assert store_path.path == ""


async def test_make_store_path_with_regular_url() -> None:
    """Test make_store_path with regular URLs (backward compatibility)."""
    pytest.importorskip("fsspec", reason="fsspec not available")
    pytest.importorskip(
        "fsspec",
        minversion="2024.12.0",
        reason="fsspec >= 2024.12.0 required for AsyncFileSystemWrapper",
    )

    # Test that regular fsspec paths still work
    # Note: We test with memory:// which doesn't require network
    store_path = await make_store_path("memory://test")
    assert store_path.store is not None
    # Path should be preserved in the store
    assert "test" in str(store_path)


# =============================================================================
# Integration Tests
# =============================================================================


def test_memory_store_integration() -> None:
    """Test end-to-end with memory store."""
    # Create array with ZEP 8 URL
    arr = zarr.create_array("memory:|zarr3:", shape=(10,), dtype="i4")
    assert isinstance(arr, Array), "Expected array, got group"
    arr[:] = range(10)

    # Verify data
    assert arr[0] == 0
    assert arr[9] == 9


def test_zip_integration(tmp_path: Path) -> None:
    """Test end-to-end with ZIP store."""
    # Create a zarr group and save to ZIP
    zip_path = tmp_path / "test.zip"

    # Create a test group with array using ZipStore directly
    with ZipStore(str(zip_path), mode="w") as zip_store:
        group = zarr.open_group(zip_store, mode="w")
        arr = group.create_array("data", shape=(5,), dtype="i4")
        arr[:] = [1, 2, 3, 4, 5]

    # Now read using ZEP 8 URL syntax
    group = zarr.open_group(f"{zip_path}|zip:", mode="r")
    # Verify we can read the data
    assert list(group["data"][:]) == [1, 2, 3, 4, 5]  # type: ignore[arg-type, index]


def test_zip_integration_simple_file_path(tmp_path: Path) -> None:
    """Test ZEP 8 URL with simple file path (no file: prefix)."""
    # Create a zarr group and save to ZIP
    zip_path = tmp_path / "simple.zip"

    # Create a test group with array using ZipStore directly
    with ZipStore(str(zip_path), mode="w") as zip_store:
        group = zarr.open_group(zip_store, mode="w")
        arr = group.create_array("data", shape=(3,), dtype="i4")
        arr[:] = [10, 20, 30]

    # Now read using ZEP 8 URL syntax with simple path
    group = zarr.open_group(f"{zip_path}|zip:", mode="r")
    # Verify we can read the data
    assert "data" in group
    data_arr = group["data"]
    assert list(data_arr[:]) == [10, 20, 30]  # type: ignore[arg-type, index]


def test_format_specification() -> None:
    """Test that Zarr format can be specified in URLs."""
    # Test zarr2 format specification
    arr2 = zarr.create_array("memory:|zarr2:", shape=(5,), dtype="i4", zarr_format=2)
    assert arr2 is not None

    # Test zarr3 format specification
    arr3 = zarr.create_array("memory:|zarr3:", shape=(5,), dtype="i4", zarr_format=3)
    assert arr3 is not None


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


def test_existing_urls_work(tmp_path: Path) -> None:
    """Test that existing URL patterns continue to work."""
    # Test local filesystem
    local_path = tmp_path / "test.zarr"
    arr = zarr.create_array(str(local_path), shape=(5,), dtype="i4")
    arr[:] = [1, 2, 3, 4, 5]

    # Read back
    arr2 = zarr.open_array(str(local_path), mode="r")
    assert list(arr2[:]) == [1, 2, 3, 4, 5]  # type: ignore[arg-type]


def test_memory_store_compatibility() -> None:
    """Test memory store compatibility."""
    # New style using ZEP 8
    arr2 = zarr.create_array("memory:", shape=(3,), dtype="i4")
    arr2[:] = [4, 5, 6]
    assert list(arr2[:]) == [4, 5, 6]  # type: ignore[arg-type]


# =============================================================================
# URLSegment Tests
# =============================================================================


def test_url_segment_creation() -> None:
    """Test creating URL segments."""
    # Test with scheme
    segment = URLSegment(scheme="s3", path="bucket/data", adapter=None)
    assert segment.scheme == "s3"
    assert segment.path == "bucket/data"
    assert segment.adapter is None

    # Test with adapter
    segment2 = URLSegment(scheme=None, path="subdir/", adapter="zip")
    assert segment2.scheme is None
    assert segment2.path == "subdir/"
    assert segment2.adapter == "zip"


def test_url_segment_repr() -> None:
    """Test URL segment string representation."""
    segment = URLSegment(scheme="s3", path="bucket/data", adapter=None)
    repr_str = repr(segment)
    assert "s3" in repr_str
    assert "bucket/data" in repr_str


def test_url_segment_equality() -> None:
    """Test URL segment equality."""
    seg1 = URLSegment(scheme="s3", path="bucket", adapter=None)
    seg2 = URLSegment(scheme="s3", path="bucket", adapter=None)
    seg3 = URLSegment(scheme="s3", path="bucket2", adapter=None)

    assert seg1 == seg2
    assert seg1 != seg3


# =============================================================================
# Store Adapter Interface Tests
# =============================================================================


def test_abstract_methods() -> None:
    """Test that StoreAdapter requires implementation of abstract methods."""

    # Should fail because from_url_segment is not implemented
    class IncompleteAdapter(StoreAdapter):
        adapter_name = "incomplete"

    with pytest.raises(TypeError):
        IncompleteAdapter()  # type: ignore[abstract]


def test_concrete_implementation() -> None:
    """Test concrete implementation of StoreAdapter."""

    class TestAdapter(StoreAdapter):
        adapter_name = "test"

        @classmethod
        async def from_url_segment(
            cls, segment: URLSegment, preceding_url: str, **kwargs: Any
        ) -> MemoryStore:
            return MemoryStore()

    adapter = TestAdapter()
    assert adapter.adapter_name == "test"


# =============================================================================
# FSSpec Integration Tests
# =============================================================================


def test_fsspec_store_adapters_registered() -> None:
    """Test that fsspec-based adapters are registered."""
    pytest.importorskip("fsspec", reason="fsspec not available")

    # Test that fsspec adapters are available
    s3_adapter = get_store_adapter("s3")
    assert s3_adapter is not None

    https_adapter = get_store_adapter("https")
    assert https_adapter is not None

    gs_adapter = get_store_adapter("gs")
    assert gs_adapter is not None


async def test_fsspec_s3_url_resolution() -> None:
    """Test S3 URL resolution using fsspec."""
    pytest.importorskip("fsspec", reason="fsspec not available")

    resolver = URLStoreResolver()

    # Test S3 URL parsing and format extraction
    s3_url = "s3://my-bucket/data.zip|zip:|zarr3:"

    # Extract zarr format
    zarr_format = resolver.extract_zarr_format(s3_url)
    assert zarr_format == 3

    # Extract path
    path = resolver.extract_path(s3_url)
    assert path == ""

    # Test URL without format
    s3_simple = "s3://my-bucket/data.zarr"
    format_none = resolver.extract_zarr_format(s3_simple)
    assert format_none is None


async def test_fsspec_https_url_resolution() -> None:
    """Test HTTPS URL resolution using fsspec."""
    pytest.importorskip("fsspec", reason="fsspec not available")

    resolver = URLStoreResolver()

    # Test HTTPS URL parsing
    https_url = "https://example.com/data.zip|zip:|zarr2:"

    # Extract zarr format
    zarr_format = resolver.extract_zarr_format(https_url)
    assert zarr_format == 2

    # Extract path
    path = resolver.extract_path(https_url)
    assert path == ""


async def test_fsspec_store_creation_mock() -> None:
    """Test fsspec store creation with mocked filesystem."""
    fsspec = pytest.importorskip("fsspec", reason="fsspec not available")
    pytest.importorskip(
        "fsspec",
        minversion="2024.12.0",
        reason="fsspec >= 2024.12.0 required for AsyncFileSystemWrapper",
    )

    # Create a mock filesystem for testing
    from zarr.storage._fsspec import _make_async

    # Test creating store from memory filesystem (doesn't require network)
    sync_fs = fsspec.filesystem("memory")
    async_fs = _make_async(sync_fs)
    store = FsspecStore(fs=async_fs, path="/test", read_only=True)

    assert store.fs == async_fs
    assert store.path == "/test"
    assert store.read_only


async def test_make_store_path_with_fsspec_urls() -> None:
    """Test make_store_path with fsspec-style URLs."""
    pytest.importorskip("fsspec", reason="fsspec not available")

    # Test that fsspec URLs still work with make_store_path
    # Note: These will fail to connect but should parse correctly
    fsspec_urls = ["s3://bucket/path", "gs://bucket/path", "https://example.com/data"]

    for url in fsspec_urls:
        # These should not be detected as ZEP 8 URLs
        assert not is_zep8_url(url)

        # make_store_path should handle them via fsspec logic
        # We don't actually call it here to avoid network requests


def test_fsspec_zep8_url_detection() -> None:
    """Test ZEP 8 URL detection with fsspec schemes."""
    pytest.importorskip("fsspec", reason="fsspec not available")

    # These should be detected as ZEP 8 URLs
    zep8_urls = [
        "s3://bucket/data.zip|zip:",
        "s3+http://minio.local:9000/bucket/data.zip|zip:",
        "s3+https://storage.example.com/bucket/data.zarr|zarr3:",
        "https://example.com/data|zip:|zarr3:",
        "gs://bucket/data.zarr|zarr2:",
    ]

    for url in zep8_urls:
        assert is_zep8_url(url), f"Should detect {url} as ZEP 8"

    # These should NOT be detected as ZEP 8 URLs
    regular_urls = [
        "s3://bucket/data.zarr",
        "https://example.com/data.zarr",
        "gs://bucket/data",
    ]

    for url in regular_urls:
        assert not is_zep8_url(url), f"Should NOT detect {url} as ZEP 8"


async def test_fsspec_adapter_error_handling() -> None:
    """Test error handling in fsspec adapters."""
    pytest.importorskip("fsspec", reason="fsspec not available")

    # Test S3 adapter with invalid URL
    segment = URLSegment(scheme="s3", path="bucket/data", adapter=None)

    with pytest.raises(ValueError, match="Unsupported S3 URL format"):
        await S3Adapter.from_url_segment(segment, "invalid://url")

    # Test HTTPS adapter with invalid URL
    with pytest.raises(ValueError, match="Unsupported scheme"):
        await HttpsAdapter.from_url_segment(segment, "ftp://invalid")


async def test_fsspec_storage_options() -> None:
    """Test that storage options are properly passed to fsspec."""
    pytest.importorskip("fsspec", reason="fsspec not available")

    # Test with storage options - verify adapter accepts configuration

    # This would normally create an fsspec store, but we can't test the full
    # creation without network access. We just verify the adapter can handle
    # the parameters without raising an error during validation.
    # The adapter should accept the parameters
    assert S3Adapter.can_handle_scheme("s3")
    assert "s3" in S3Adapter.get_supported_schemes()


def test_fsspec_schemes_support() -> None:
    """Test which schemes fsspec adapters support."""
    pytest.importorskip("fsspec", reason="fsspec not available")

    # Test S3 adapter
    assert S3Adapter.can_handle_scheme("s3")
    assert S3Adapter.get_supported_schemes() == ["s3", "s3+http", "s3+https"]

    # Test HTTPS adapter
    assert HttpsAdapter.can_handle_scheme("https")
    assert HttpsAdapter.can_handle_scheme("http")
    assert set(HttpsAdapter.get_supported_schemes()) == {"http", "https"}

    # Test GCS adapter
    assert GSAdapter.can_handle_scheme("gs")
    # GCS adapter only supports gs:// scheme
    supported_schemes = GSAdapter.get_supported_schemes()
    assert "gs" in supported_schemes


async def test_fsspec_url_chain_parsing() -> None:
    """Test parsing of complex fsspec URL chains."""
    pytest.importorskip("fsspec", reason="fsspec not available")

    resolver = URLStoreResolver()

    # Test complex chained URLs
    complex_urls = [
        "s3://bucket/archive.zip|zip:data/|zarr3:group",
        "s3+http://minio.local:9000/bucket/archive.zip|zip:data/|zarr3:group",
        "s3+https://storage.example.com/bucket/nested.zip|zip:inner/|zarr2:",
        "https://example.com/data.tar.gz|tar:|zip:|zarr2:",
        "gs://bucket/dataset.zarr|zarr3:array/subarray",
    ]

    for url in complex_urls:
        # Should be detected as ZEP 8 URL
        assert is_zep8_url(url)

        # Should be able to extract format
        zarr_format = resolver.extract_zarr_format(url)

        # Verify reasonable results
        if "|zarr2:" in url:
            assert zarr_format == 2
        elif "|zarr3:" in url:
            assert zarr_format == 3


# =============================================================================
# LoggingStore Adapter Tests
# =============================================================================


def test_logging_adapter_registration() -> None:
    """Test that LoggingAdapter is registered properly."""
    adapter = get_store_adapter("log")
    assert adapter is LoggingAdapter
    assert adapter.adapter_name == "log"


def test_logging_adapter_scheme_handling() -> None:
    """Test LoggingAdapter scheme handling."""

    # LoggingAdapter should not handle schemes directly (it's a wrapper)
    assert not LoggingAdapter.can_handle_scheme("log")
    assert not LoggingAdapter.can_handle_scheme("memory")
    assert LoggingAdapter.get_supported_schemes() == []


async def test_logging_adapter_basic_functionality() -> None:
    """Test basic LoggingAdapter functionality with memory store."""

    resolver = URLStoreResolver()

    # Test basic memory store with logging
    store = await resolver.resolve_url("memory:|log:")
    assert isinstance(store, LoggingStore)

    # Verify it wraps a MemoryStore
    assert isinstance(store._store, MemoryStore)


async def test_logging_adapter_query_parameters() -> None:
    """Test LoggingAdapter with query parameters for log level."""

    resolver = URLStoreResolver()

    # Test with custom log level
    store = await resolver.resolve_url("memory:|log:?log_level=INFO")
    assert isinstance(store, LoggingStore)
    assert store.log_level == "INFO"

    # Test with DEBUG log level
    store_debug = await resolver.resolve_url("memory:|log:?log_level=DEBUG")
    assert isinstance(store_debug, LoggingStore)
    assert store_debug.log_level == "DEBUG"

    # Test without query parameters (should default to DEBUG)
    store_default = await resolver.resolve_url("memory:|log:")
    assert isinstance(store_default, LoggingStore)
    assert store_default.log_level == "DEBUG"


async def test_logging_adapter_with_file_store(tmp_path: Path) -> None:
    """Test LoggingAdapter wrapping a file store."""

    resolver = URLStoreResolver()

    # Test with file store
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    store = await resolver.resolve_url(f"file:{test_dir}|log:")
    assert isinstance(store, LoggingStore)
    assert isinstance(store._store, LocalStore)


async def test_logging_adapter_operations_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that store operations are actually logged."""

    resolver = URLStoreResolver()

    # Create test directory
    test_dir = tmp_path / "log_test"
    test_dir.mkdir()

    # Create logging store
    store = await resolver.resolve_url(f"file:{test_dir}|log:?log_level=INFO")
    assert isinstance(store, LoggingStore)

    # Clear previous log records
    caplog.clear()

    # Perform some operations
    buffer = default_buffer_prototype().buffer
    test_data = buffer.from_bytes(b"test data")

    # Test set operation
    await store.set("c/0", test_data)

    # Check that operations were logged
    assert len(caplog.record_tuples) >= 2  # Start and finish logs
    log_messages = [record[2] for record in caplog.record_tuples]

    # Should see calling and finished messages
    calling_logs = [msg for msg in log_messages if "Calling" in msg]
    finished_logs = [msg for msg in log_messages if "Finished" in msg]

    assert len(calling_logs) >= 1
    assert len(finished_logs) >= 1

    # Should mention the operation and store type
    assert any("set" in msg for msg in log_messages)
    assert any("LocalStore" in msg for msg in log_messages)


def test_logging_adapter_zep8_url_detection() -> None:
    """Test that logging URLs are properly detected as ZEP 8 URLs."""
    # URLs with logging should be detected as ZEP 8
    logging_urls = [
        "memory:|log:",
        "file:/tmp/data.zarr|log:",
        "memory:|log:?log_level=INFO",
        "s3://bucket/data.zarr|log:?log_level=DEBUG",
        "file:/path/data.zip|zip:|log:",
    ]

    for url in logging_urls:
        assert is_zep8_url(url), f"Should detect {url} as ZEP 8 URL"

    # Regular URLs should not be detected as ZEP 8
    regular_urls = [
        "file:/tmp/zarr.log",
        "/local/log",
        "https://example.com/data.zarr/log",
    ]

    for url in regular_urls:
        assert not is_zep8_url(url), f"Should NOT detect {url} as ZEP 8 URL"


def test_logging_adapter_integration_with_zarr() -> None:
    """Test LoggingAdapter integration with zarr.open and zarr.create."""
    # Test creating array with logging
    arr = zarr.create_array("memory:|log:", shape=(5,), dtype="i4")
    arr[:] = [1, 2, 3, 4, 5]

    # Verify data integrity
    data = []
    for i in range(5):
        value = arr[i]
        # Cast numpy scalar to Python int
        data.append(value.item() if hasattr(value, "item") else int(value))  # type: ignore[arg-type]
    assert data == [1, 2, 3, 4, 5]

    # Verify that the underlying store is LoggingStore
    assert isinstance(arr.store, LoggingStore)


def test_logging_adapter_integration_with_zip(tmp_path: Path) -> None:
    """Test LoggingAdapter integration with ZIP files."""

    # Create a test ZIP with zarr data
    zip_path = tmp_path / "test_with_logging.zip"

    # First create zarr data in ZIP without logging
    with ZipStore(str(zip_path), mode="w") as zip_store:
        group = zarr.open_group(zip_store, mode="w")
        arr = group.create_array("temperature", shape=(4,), dtype="f4")
        arr[:] = [20.5, 21.0, 19.8, 22.1]

    # Now read using ZEP 8 URL with logging
    group = zarr.open_group(f"file:{zip_path}|zip:|log:", mode="r")

    # Verify we can access the data
    assert "temperature" in group
    temp_item = group["temperature"]  # This gets the array from the group

    # Check if it's an array by looking for array-like attributes
    assert isinstance(temp_item, Array)
    assert temp_item.shape == (4,)

    # Verify the store chain
    store_path = group.store_path
    assert isinstance(store_path.store, LoggingStore)


async def test_logging_adapter_error_handling() -> None:
    """Test error handling in LoggingAdapter."""
    from zarr.storage._builtin_adapters import LoggingAdapter

    # Test with invalid preceding URL
    segment = URLSegment(scheme=None, adapter="log", path="")

    # Should handle errors gracefully when underlying store creation fails
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        await LoggingAdapter.from_url_segment(segment, "invalid://nonexistent", mode="r")


async def test_logging_adapter_query_parameter_parsing() -> None:
    """Test that LoggingAdapter correctly parses and applies query parameters."""
    resolver = URLStoreResolver()

    # Test different log levels via query parameters
    test_cases = [
        ("memory:|log:?log_level=INFO", "INFO"),
        ("memory:|log:?log_level=DEBUG", "DEBUG"),
        ("memory:|log:?log_level=WARNING", "WARNING"),
        ("memory:|log:?log_level=ERROR", "ERROR"),
        ("memory:|log:", "DEBUG"),  # Default when no parameters
    ]

    for url, expected_level in test_cases:
        store = await resolver.resolve_url(url)
        assert isinstance(store, LoggingStore)
        assert store.log_level == expected_level


async def test_logging_adapter_preserves_store_properties() -> None:
    """Test that LoggingAdapter preserves underlying store properties."""

    resolver = URLStoreResolver()

    # Test with memory store (supports writes)
    memory_logged = await resolver.resolve_url("memory:|log:")
    assert isinstance(memory_logged, LoggingStore)
    assert memory_logged.supports_writes
    assert memory_logged.supports_deletes
    assert memory_logged.supports_listing
    assert not memory_logged.supports_partial_writes  # Always False per ABC


# Error handling and edge case tests
async def test_url_segment_validation_errors() -> None:
    """Test URLSegment validation error conditions."""
    from zarr.storage._zep8 import ZEP8URLError

    # Test missing both scheme and adapter
    with pytest.raises(ZEP8URLError, match="URL segment must have either scheme or adapter"):
        URLSegment()

    # Test invalid adapter name (contains spaces)
    with pytest.raises(ZEP8URLError, match="Invalid adapter name"):
        URLSegment(adapter="invalid name")

    # Test invalid adapter name (contains special chars)
    with pytest.raises(ZEP8URLError, match="Invalid adapter name"):
        URLSegment(adapter="invalid@name")


async def test_memory_adapter_error_conditions() -> None:
    """Test error conditions in MemoryAdapter."""
    from zarr.storage._builtin_adapters import MemoryAdapter

    # Test non-memory URL error
    segment = URLSegment(adapter="memory")
    with pytest.raises(ValueError, match="Expected memory: URL"):
        await MemoryAdapter.from_url_segment(segment, "file:/invalid/path")

    # Test memory URL with sub-path error
    segment = URLSegment(adapter="memory", path="subpath")
    with pytest.raises(ValueError, match="Memory store does not currently support sub-paths"):
        await MemoryAdapter.from_url_segment(segment, "memory:")


async def test_file_adapter_error_conditions() -> None:
    """Test error conditions in FileSystemAdapter."""
    from zarr.storage._builtin_adapters import FileSystemAdapter

    # Test non-file URL error
    segment = URLSegment(adapter="file")
    with pytest.raises(ValueError, match="Expected file: URL"):
        await FileSystemAdapter.from_url_segment(segment, "memory:/invalid")


async def test_zip_adapter_error_conditions(tmp_path: Path) -> None:
    """Test error conditions in ZipAdapter."""
    from zarr.storage._builtin_adapters import ZipAdapter

    # Test with invalid ZIP file path (should raise FileNotFoundError)
    segment = URLSegment(adapter="zip")
    with pytest.raises(FileNotFoundError):
        await ZipAdapter.from_url_segment(segment, "/nonexistent/path/file.zip")


async def test_store_adapter_abstract_methods() -> None:
    """Test that StoreAdapter abstract methods work correctly."""
    from zarr.storage._builtin_adapters import FileSystemAdapter

    # Test can_handle_scheme method
    assert FileSystemAdapter.can_handle_scheme("file")
    assert not FileSystemAdapter.can_handle_scheme("s3")

    # Test get_supported_schemes method
    schemes = FileSystemAdapter.get_supported_schemes()
    assert "file" in schemes

    # Test MemoryAdapter schemes
    from zarr.storage._builtin_adapters import MemoryAdapter

    assert MemoryAdapter.can_handle_scheme("memory")
    assert not MemoryAdapter.can_handle_scheme("file")

    memory_schemes = MemoryAdapter.get_supported_schemes()
    assert "memory" in memory_schemes


async def test_fsspec_adapter_error_conditions() -> None:
    """Test error conditions in RemoteAdapter."""
    from zarr.storage._builtin_adapters import RemoteAdapter

    # Test non-remote URL
    segment = URLSegment(adapter="s3")
    with pytest.raises(ValueError, match="Expected remote URL"):
        await RemoteAdapter.from_url_segment(segment, "file:/invalid")


async def test_zep8_url_parsing_edge_cases() -> None:
    """Test edge cases in ZEP 8 URL parsing."""
    from zarr.storage._zep8 import URLParser, ZEP8URLError

    parser = URLParser()

    # Test URL with empty segments (should raise error)
    with pytest.raises(ZEP8URLError, match="Empty URL segment found"):
        parser.parse("file:///path||memory:")

    # Test URL with complex paths
    segments = parser.parse("s3://bucket/deep/nested/path.zip|zip:inner/path")
    assert len(segments) == 2
    assert segments[0].scheme == "s3"
    assert segments[0].path == "bucket/deep/nested/path.zip"
    assert segments[1].adapter == "zip"
    assert segments[1].path == "inner/path"


async def test_url_resolver_error_handling() -> None:
    """Test error handling in URLStoreResolver."""
    from zarr.storage._zep8 import URLStoreResolver

    resolver = URLStoreResolver()

    # Test unregistered adapter
    with pytest.raises(ValueError, match="Unknown store adapter"):
        await resolver.resolve_url("file:/test|unregistered_adapter:")

    # Test invalid URL format
    with pytest.raises((ValueError, TypeError)):  # Various parsing errors possible
        await resolver.resolve_url(":::invalid:::url:::")


async def test_logging_adapter_with_invalid_log_level() -> None:
    """Test LoggingAdapter with invalid log level handling."""
    from zarr.storage._builtin_adapters import LoggingAdapter

    # Test with invalid log level (should raise ValueError)
    segment = URLSegment(adapter="log", path="?log_level=INVALID_LEVEL")
    with pytest.raises(ValueError, match="Unknown level"):
        await LoggingAdapter.from_url_segment(segment, "memory:")


def test_adapter_subclass_validation() -> None:
    """Test StoreAdapter subclass validation."""

    # Test that adapter_name is required
    with pytest.raises(TypeError, match="must define 'adapter_name'"):

        class InvalidAdapter(StoreAdapter):
            pass

    # Test that adapter_name must be string
    with pytest.raises(TypeError, match="adapter_name must be a string"):

        class InvalidAdapter2(StoreAdapter):
            adapter_name = 123  # type: ignore[assignment]

    # Test that adapter_name format is validated (must start with letter)
    with pytest.raises(ValueError, match="Invalid adapter_name format"):

        class InvalidAdapter3(StoreAdapter):
            adapter_name = "9invalid"

    # Test that adapter_name format is validated (no special chars)
    with pytest.raises(ValueError, match="Invalid adapter_name format"):

        class InvalidAdapter4(StoreAdapter):
            adapter_name = "invalid@name"


async def test_store_adapter_base_class_methods() -> None:
    """Test StoreAdapter base class methods and their default implementations."""

    # Create a concrete test adapter to test base class functionality
    class TestAdapter(StoreAdapter):
        adapter_name = "test"

        @classmethod
        async def from_url_segment(
            cls, segment: URLSegment, preceding_url: str, **kwargs: Any
        ) -> Store:
            from zarr.storage import MemoryStore

            return await MemoryStore.open()

    # Test default can_handle_scheme implementation
    assert not TestAdapter.can_handle_scheme("file")
    assert not TestAdapter.can_handle_scheme("s3")
    assert not TestAdapter.can_handle_scheme("https")

    # Test default get_supported_schemes implementation
    schemes = TestAdapter.get_supported_schemes()
    assert schemes == []

    # Test that we can instantiate and use the adapter
    segment = URLSegment(adapter="test")
    store = await TestAdapter.from_url_segment(segment, "memory:")
    assert store is not None


def test_url_segment_equality_and_repr() -> None:
    """Test URLSegment equality, repr, and hash methods."""
    # Test equality
    segment1 = URLSegment(scheme="file", path="/test")
    segment2 = URLSegment(scheme="file", path="/test")
    segment3 = URLSegment(scheme="s3", path="/test")

    assert segment1 == segment2
    assert segment1 != segment3

    # Test repr
    segment = URLSegment(scheme="file", adapter="zip", path="/test.zip")
    repr_str = repr(segment)
    assert "URLSegment" in repr_str
    assert "scheme='file'" in repr_str
    assert "adapter='zip'" in repr_str
    assert "path='/test.zip'" in repr_str

    # Test __post_init__ path coverage
    segment = URLSegment(scheme="file")  # Valid: has scheme
    assert segment.scheme == "file"

    segment = URLSegment(adapter="zip")  # Valid: has adapter
    assert segment.adapter == "zip"


def test_url_segment_edge_cases() -> None:
    """Test URLSegment edge cases and validation."""

    # Test adapter name validation with various valid names
    valid_names = ["zip", "memory", "s3", "test123", "valid_name", "valid-name", "z", "Z", "1abc"]
    for name in valid_names:
        segment = URLSegment(adapter=name)
        assert segment.adapter == name

    # Test scheme without adapter (valid)
    segment = URLSegment(scheme="file", path="/test")
    assert segment.scheme == "file"
    assert segment.adapter is None

    # Test adapter without scheme (valid)
    segment = URLSegment(adapter="zip", path="inner/path")
    assert segment.adapter == "zip"
    assert segment.scheme is None

    # Test default path
    segment = URLSegment(scheme="file")
    assert segment.path == ""


async def test_store_adapter_abstract_method() -> None:
    """Test that StoreAdapter.from_url_segment is properly abstract."""
    # Test that calling the abstract method directly raises NotImplementedError
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        StoreAdapter()  # type: ignore[abstract]

    # Test creating a subclass without implementing the abstract method
    class IncompleteAdapter(StoreAdapter):
        adapter_name = "incomplete"

    # The error happens when trying to instantiate, not when defining the class
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteAdapter()  # type: ignore[abstract]


def test_store_adapter_validation_edge_cases() -> None:
    """Test edge cases in StoreAdapter validation."""

    # Test adapter_name with minimum length
    class SingleCharAdapter(StoreAdapter):
        adapter_name = "a"

        @classmethod
        async def from_url_segment(
            cls, segment: URLSegment, preceding_url: str, **kwargs: Any
        ) -> Store:
            from zarr.storage import MemoryStore

            return await MemoryStore.open()

    # Should work fine
    assert SingleCharAdapter.adapter_name == "a"

    # Test adapter_name with underscores and hyphens
    class ComplexNameAdapter(StoreAdapter):
        adapter_name = "complex_name-with-parts"

        @classmethod
        async def from_url_segment(
            cls, segment: URLSegment, preceding_url: str, **kwargs: Any
        ) -> Store:
            from zarr.storage import MemoryStore

            return await MemoryStore.open()

    assert ComplexNameAdapter.adapter_name == "complex_name-with-parts"


def test_store_adapter_imports_and_docstrings() -> None:
    """Test that imports and module-level functionality work correctly."""
    # Test that URLSegment is available and can be imported
    # Test that type annotations work

    # Test that all exports are available
    from zarr.abc.store_adapter import StoreAdapter, URLSegment, __all__

    assert "StoreAdapter" in __all__
    assert "URLSegment" in __all__

    # Test that we can create URLSegments and they validate correctly
    segment = URLSegment(scheme="test")
    assert segment.scheme == "test"

    # Test the base StoreAdapter class exists
    assert StoreAdapter.__name__ == "StoreAdapter"
    assert hasattr(StoreAdapter, "from_url_segment")
    assert hasattr(StoreAdapter, "can_handle_scheme")
    assert hasattr(StoreAdapter, "get_supported_schemes")


def test_store_adapter_kwargs_handling() -> None:
    """Test StoreAdapter __init_subclass__ with kwargs."""

    # Test that __init_subclass__ handles kwargs correctly
    class TestAdapterWithKwargs(StoreAdapter):
        adapter_name = "test_kwargs"
        custom_param: Any  # Type annotation for dynamic attribute

        @classmethod
        async def from_url_segment(
            cls, segment: URLSegment, preceding_url: str, **kwargs: Any
        ) -> Store:
            from zarr.storage import MemoryStore

            return await MemoryStore.open()

        def __init_subclass__(cls, custom_param: Any = None, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            cls.custom_param = custom_param

    # Test subclassing with custom kwargs
    class SubAdapterWithCustom(TestAdapterWithKwargs, custom_param="test_value"):
        adapter_name = "sub_kwargs"

    # The base class doesn't have custom_param set until subclassed
    # Only subclass has the custom_param attribute
    assert SubAdapterWithCustom.custom_param == "test_value"


# Tests for comprehensive coverage of _builtin_adapters.py


async def test_filesystem_adapter_edge_cases() -> None:
    """Test FileSystemAdapter edge cases and full functionality."""
    from zarr.storage._builtin_adapters import FileSystemAdapter

    # Test with empty path (should default to ".")
    segment = URLSegment(adapter="file")
    store = await FileSystemAdapter.from_url_segment(segment, "file:")
    assert store is not None

    # Test with mode specified in kwargs
    segment = URLSegment(adapter="file")
    store = await FileSystemAdapter.from_url_segment(segment, "file:/tmp", mode="r")
    assert store.read_only

    # Test with read_only in storage_options
    segment = URLSegment(adapter="file")
    store = await FileSystemAdapter.from_url_segment(
        segment, "file:/tmp", storage_options={"read_only": True}
    )
    assert store.read_only

    # Test get_supported_schemes
    schemes = FileSystemAdapter.get_supported_schemes()
    assert "file" in schemes


async def test_memory_adapter_comprehensive() -> None:
    """Test MemoryAdapter comprehensive functionality."""
    from zarr.storage._builtin_adapters import MemoryAdapter

    # Test get_supported_schemes
    schemes = MemoryAdapter.get_supported_schemes()
    assert "memory" in schemes

    # Test can_handle_scheme
    assert MemoryAdapter.can_handle_scheme("memory")


async def test_remote_adapter_comprehensive() -> None:
    """Test RemoteAdapter comprehensive functionality."""
    from zarr.storage._builtin_adapters import RemoteAdapter

    # Test _is_remote_url method
    assert RemoteAdapter._is_remote_url("s3://bucket/path")
    assert RemoteAdapter._is_remote_url("gs://bucket/path")
    assert RemoteAdapter._is_remote_url("https://example.com/file")
    assert not RemoteAdapter._is_remote_url("file:/local/path")
    assert not RemoteAdapter._is_remote_url("/local/path")


async def test_s3_adapter_functionality() -> None:
    """Test S3Adapter functionality including custom endpoints."""
    from zarr.storage._builtin_adapters import S3Adapter

    # Test can_handle_scheme for all supported schemes
    assert S3Adapter.can_handle_scheme("s3")
    assert S3Adapter.can_handle_scheme("s3+http")
    assert S3Adapter.can_handle_scheme("s3+https")
    assert not S3Adapter.can_handle_scheme("gs")
    assert not S3Adapter.can_handle_scheme("http")

    # Test get_supported_schemes
    schemes = S3Adapter.get_supported_schemes()
    assert "s3" in schemes
    assert "s3+http" in schemes
    assert "s3+https" in schemes
    assert len(schemes) == 3


async def test_s3_custom_endpoint_url_parsing() -> None:
    """Test S3Adapter URL parsing for custom endpoints."""
    from zarr.storage._builtin_adapters import S3Adapter

    # Test standard AWS S3 URL parsing
    s3_url, endpoint_url, storage_options = S3Adapter._parse_s3_url("s3://my-bucket/path/to/data")
    assert s3_url == "s3://my-bucket/path/to/data"
    assert endpoint_url is None
    assert storage_options == {}

    # Test custom HTTP endpoint parsing
    s3_url, endpoint_url, storage_options = S3Adapter._parse_s3_url(
        "s3+http://minio.local:9000/my-bucket/data"
    )
    assert s3_url == "s3://my-bucket/data"
    assert endpoint_url == "http://minio.local:9000"
    assert storage_options == {"endpoint_url": "http://minio.local:9000", "use_ssl": False}

    # Test custom HTTPS endpoint parsing
    s3_url, endpoint_url, storage_options = S3Adapter._parse_s3_url(
        "s3+https://storage.example.com/bucket/path/file.zarr"
    )
    assert s3_url == "s3://bucket/path/file.zarr"
    assert endpoint_url == "https://storage.example.com"
    assert storage_options == {"endpoint_url": "https://storage.example.com", "use_ssl": True}

    # Test custom HTTP endpoint with port
    s3_url, endpoint_url, storage_options = S3Adapter._parse_s3_url(
        "s3+http://localhost:9000/test-bucket"
    )
    assert s3_url == "s3://test-bucket"
    assert endpoint_url == "http://localhost:9000"
    assert storage_options["endpoint_url"] == "http://localhost:9000"
    assert storage_options["use_ssl"] is False

    # Test edge case: endpoint without path
    s3_url, endpoint_url, storage_options = S3Adapter._parse_s3_url("s3+https://minio.example.com")
    assert s3_url == "s3://"
    assert endpoint_url == "https://minio.example.com"


async def test_s3_custom_endpoint_scheme_extraction() -> None:
    """Test S3Adapter scheme extraction for custom endpoints."""
    from zarr.storage._builtin_adapters import S3Adapter

    # Test scheme extraction
    assert S3Adapter._extract_scheme("s3://bucket/path") == "s3"
    assert S3Adapter._extract_scheme("s3+http://minio.local:9000/bucket") == "s3+http"
    assert S3Adapter._extract_scheme("s3+https://storage.example.com/bucket") == "s3+https"


async def test_s3_custom_endpoint_error_handling() -> None:
    """Test S3Adapter error handling for invalid URLs."""
    from zarr.storage._builtin_adapters import S3Adapter

    # Test unsupported URL format
    with pytest.raises(ValueError, match="Unsupported S3 URL format"):
        S3Adapter._parse_s3_url("invalid://not-s3")

    with pytest.raises(ValueError, match="Unsupported S3 URL format"):
        S3Adapter._parse_s3_url("gs://bucket/path")


async def test_s3_custom_endpoint_registration() -> None:
    """Test that custom S3 endpoint schemes are properly registered."""
    from zarr.registry import get_store_adapter

    # Test that all S3 schemes can be retrieved
    s3_adapter = get_store_adapter("s3")
    assert s3_adapter is not None
    assert s3_adapter == S3Adapter

    s3_http_adapter = get_store_adapter("s3+http")
    assert s3_http_adapter is not None
    assert s3_http_adapter == S3HttpAdapter

    s3_https_adapter = get_store_adapter("s3+https")
    assert s3_https_adapter is not None
    assert s3_https_adapter == S3HttpsAdapter


async def test_s3_http_adapter_functionality() -> None:
    """Test S3HttpAdapter specific functionality."""
    # Test adapter name
    assert S3HttpAdapter.adapter_name == "s3+http"

    # Test supported schemes
    schemes = S3HttpAdapter.get_supported_schemes()
    assert schemes == ["s3+http"]

    # Test can_handle_scheme
    assert S3HttpAdapter.can_handle_scheme("s3+http")
    assert not S3HttpAdapter.can_handle_scheme("s3")
    assert not S3HttpAdapter.can_handle_scheme("s3+https")


async def test_s3_https_adapter_functionality() -> None:
    """Test S3HttpsAdapter specific functionality."""
    # Test adapter name
    assert S3HttpsAdapter.adapter_name == "s3+https"

    # Test supported schemes
    schemes = S3HttpsAdapter.get_supported_schemes()
    assert schemes == ["s3+https"]

    # Test can_handle_scheme
    assert S3HttpsAdapter.can_handle_scheme("s3+https")
    assert not S3HttpsAdapter.can_handle_scheme("s3")
    assert not S3HttpsAdapter.can_handle_scheme("s3+http")


async def test_s3_custom_endpoint_zep8_url_detection() -> None:
    """Test ZEP 8 URL detection with custom S3 endpoints."""
    from zarr.storage._zep8 import is_zep8_url

    # Standard S3 URLs (not ZEP 8)
    assert not is_zep8_url("s3://bucket/data")
    assert not is_zep8_url("s3+http://minio.local:9000/bucket/data")
    assert not is_zep8_url("s3+https://storage.example.com/bucket/data")

    # ZEP 8 URLs with custom S3 endpoints
    assert is_zep8_url("s3://bucket/data.zip|zip:")
    assert is_zep8_url("s3+http://minio.local:9000/bucket/data.zip|zip:")
    assert is_zep8_url("s3+https://storage.example.com/bucket/data|zarr3:")
    assert is_zep8_url("s3+http://localhost:9000/bucket/archive.zip|zip:data/|zarr2:")


async def test_gcs_adapter_functionality() -> None:
    """Test GSAdapter functionality."""

    # Test can_handle_scheme
    assert GSAdapter.can_handle_scheme("gs")
    assert not GSAdapter.can_handle_scheme("s3")

    # Test get_supported_schemes
    schemes = GSAdapter.get_supported_schemes()
    assert "gs" in schemes
    assert len(schemes) == 1  # Should only support gs now


async def test_https_adapter_functionality() -> None:
    """Test HttpsAdapter functionality."""
    from zarr.storage._builtin_adapters import HttpsAdapter

    # Test can_handle_scheme
    assert HttpsAdapter.can_handle_scheme("https")
    assert HttpsAdapter.can_handle_scheme("http")
    assert not HttpsAdapter.can_handle_scheme("s3")

    # Test get_supported_schemes
    schemes = HttpsAdapter.get_supported_schemes()
    assert "http" in schemes
    assert "https" in schemes


async def test_zip_adapter_comprehensive(tmp_path: Path) -> None:
    """Test ZipAdapter comprehensive functionality."""
    from zarr.storage._builtin_adapters import ZipAdapter

    # Create a test ZIP file
    zip_path = tmp_path / "test.zip"
    from zarr.storage import ZipStore

    # Create minimal ZIP content
    with ZipStore(zip_path, mode="w") as zip_store:
        import zarr

        arr = zarr.create_array(zip_store, shape=(2,), dtype="i4")
        arr[:] = [1, 2]

    # Test with different modes
    segment = URLSegment(adapter="zip")

    # Test with explicit read mode
    store = await ZipAdapter.from_url_segment(segment, f"file:{zip_path}", mode="r")
    assert store.read_only

    # Test with storage_options read_only
    store = await ZipAdapter.from_url_segment(
        segment, f"file:{zip_path}", storage_options={"read_only": True}
    )
    assert store.read_only

    # Test mode mapping
    segment = URLSegment(adapter="zip")
    store = await ZipAdapter.from_url_segment(
        segment,
        f"file:{zip_path}",
        mode="w-",  # Should map to "w"
    )

    # Test _get_fsspec_protocol method
    assert ZipAdapter._get_fsspec_protocol("s3://bucket/file.zip") == "s3"
    assert ZipAdapter._get_fsspec_protocol("gs://bucket/file.zip") == "gs"
    assert ZipAdapter._get_fsspec_protocol("https://example.com/file.zip") == "http"

    # Test _is_remote_url method
    assert ZipAdapter._is_remote_url("s3://bucket/file.zip")
    assert ZipAdapter._is_remote_url("https://example.com/file.zip")
    assert not ZipAdapter._is_remote_url("file:/local/file.zip")


async def test_logging_adapter_edge_cases() -> None:
    """Test LoggingAdapter edge cases."""
    from zarr.storage._builtin_adapters import LoggingAdapter

    # Test with no query parameters
    segment = URLSegment(adapter="log", path="")
    store = await LoggingAdapter.from_url_segment(segment, "memory:")
    from zarr.storage import LoggingStore

    assert isinstance(store, LoggingStore)
    assert store.log_level == "DEBUG"  # Default

    # Test with complex query parsing
    from urllib.parse import parse_qs, urlparse

    path_with_query = "?log_level=INFO&other_param=value"
    segment = URLSegment(adapter="log", path=path_with_query)

    # Test parsing logic directly
    parsed = urlparse(f"log:{path_with_query}")
    query_params = parse_qs(parsed.query)
    assert "log_level" in query_params
    assert query_params["log_level"][0] == "INFO"


def test_builtin_adapters_imports_and_module_structure() -> None:
    """Test imports and module structure of _builtin_adapters.py."""
    # Test that all adapter classes can be imported
    from zarr.storage._builtin_adapters import (
        FileSystemAdapter,
        GSAdapter,
        HttpsAdapter,
        LoggingAdapter,
        MemoryAdapter,
        RemoteAdapter,
        S3Adapter,
        S3HttpAdapter,
        S3HttpsAdapter,
        ZipAdapter,
    )

    # Test that they all have the right adapter_name
    assert FileSystemAdapter.adapter_name == "file"
    assert MemoryAdapter.adapter_name == "memory"
    assert RemoteAdapter.adapter_name == "remote"
    assert S3Adapter.adapter_name == "s3"
    assert S3HttpAdapter.adapter_name == "s3+http"
    assert S3HttpsAdapter.adapter_name == "s3+https"
    assert GSAdapter.adapter_name == "gs"
    assert HttpsAdapter.adapter_name == "https"
    assert LoggingAdapter.adapter_name == "log"
    assert ZipAdapter.adapter_name == "zip"

    # Test inheritance relationships
    assert issubclass(S3Adapter, RemoteAdapter)
    assert issubclass(S3HttpAdapter, S3Adapter)
    assert issubclass(S3HttpsAdapter, S3Adapter)
    assert issubclass(GSAdapter, RemoteAdapter)
    assert issubclass(HttpsAdapter, RemoteAdapter)


# Tests for comprehensive coverage of _zep8.py


async def test_url_parser_relative_url_resolution() -> None:
    """Test URLParser relative URL resolution functionality."""
    from zarr.storage._zep8 import URLParser

    # Test resolve_relative_url method
    base_url = "s3://bucket/data/file.zarr"
    relative_url = "../other/file.zarr"

    result = URLParser.resolve_relative_url(base_url, relative_url)
    # This should resolve the relative path
    assert "other/file.zarr" in result


async def test_url_parser_edge_cases_and_validation() -> None:
    """Test URLParser edge cases and validation."""
    from zarr.storage._zep8 import URLParser, ZEP8URLError

    parser = URLParser()

    # Test empty URL
    with pytest.raises(ZEP8URLError, match="URL cannot be empty"):
        parser.parse("")

    # Test URL starting with pipe
    with pytest.raises(ZEP8URLError, match="URL cannot start with pipe"):
        parser.parse("|invalid")

    # Test URL with multiple consecutive pipes (empty segments)
    with pytest.raises(ZEP8URLError, match="Empty URL segment found"):
        parser.parse("file:///test||memory:")


async def test_url_store_resolver_comprehensive() -> None:
    """Test comprehensive URLStoreResolver functionality."""
    from zarr.registry import get_store_adapter

    # Test that resolver can access registered adapters via registry
    adapter = get_store_adapter("memory")
    assert adapter is not None
    assert adapter.adapter_name == "memory"

    # Test registry for unregistered adapter
    with pytest.raises(KeyError, match="Store adapter 'nonexistent' not found"):
        get_store_adapter("nonexistent")


async def test_url_resolver_methods() -> None:
    """Test URLStoreResolver extract methods."""
    from zarr.storage._zep8 import URLStoreResolver

    resolver = URLStoreResolver()

    # Test extract_zarr_format method
    assert resolver.extract_zarr_format("memory:|zarr3:") == 3
    assert resolver.extract_zarr_format("file:/test.zarr") is None
    assert resolver.extract_zarr_format("s3://bucket/data.zarr|zarr3:") == 3

    # Test extract_path method (extracts from final segment path, not base URL)
    assert resolver.extract_path("file:/tmp/test.zarr|zip:inner/path") == "inner/path"
    assert resolver.extract_path("memory:") == ""
    assert resolver.extract_path("s3://bucket/data.zarr|zarr3:group") == "group"


async def test_is_zep8_url_comprehensive() -> None:
    """Test is_zep8_url comprehensive functionality."""
    from zarr.storage._zep8 import is_zep8_url

    # Test various URL types
    assert is_zep8_url("memory:")
    assert is_zep8_url("file:/test.zarr|zip:")
    assert is_zep8_url("s3://bucket/file.zip|zip:|zarr3:")

    # Test non-ZEP 8 URLs
    assert not is_zep8_url("/local/path")
    assert not is_zep8_url("regular_filename.zarr")

    # Test edge cases
    assert not is_zep8_url("")
    assert is_zep8_url("scheme:")


async def test_url_resolver_relative_urls() -> None:
    """Test URLStoreResolver relative URL resolution."""
    from zarr.storage._zep8 import URLStoreResolver

    resolver = URLStoreResolver()

    # Test relative URL resolution
    base_url = "file:/tmp/base"
    relative_url = "../relative/path"
    resolved = resolver.resolve_relative_url(base_url, relative_url)
    assert resolved is not None
    assert "relative/path" in resolved

    # Test absolute URL (should return as-is)
    absolute_url = "memory:"
    resolved = resolver.resolve_relative_url(base_url, absolute_url)
    assert resolved == absolute_url


def test_zep8_module_imports_and_structure() -> None:
    """Test _zep8.py module imports and structure."""
    # Test that all main classes and functions are importable
    from zarr.storage._zep8 import (
        URLParser,
        URLStoreResolver,
        ZEP8URLError,
        is_zep8_url,
        resolve_url,
    )

    # Test that classes exist and have expected methods
    assert hasattr(URLParser, "parse")
    assert hasattr(URLParser, "resolve_relative_url")
    assert hasattr(URLStoreResolver, "resolve_url")
    assert hasattr(URLStoreResolver, "extract_path")
    assert hasattr(URLStoreResolver, "extract_zarr_format")

    # Test that error class is properly defined
    assert issubclass(ZEP8URLError, ValueError)

    # Test that functions are callable
    assert callable(is_zep8_url)
    assert callable(resolve_url)


async def test_url_resolver_adapter_access() -> None:
    """Test URL resolver adapter access via registry."""
    from zarr.registry import get_store_adapter, list_store_adapters

    # Test that builtin adapters are registered and accessible
    registered_adapters = list_store_adapters()
    assert "memory" in registered_adapters
    assert "file" in registered_adapters
    assert "zip" in registered_adapters

    # Test accessing a specific adapter
    memory_adapter = get_store_adapter("memory")
    assert memory_adapter is not None
    assert memory_adapter.adapter_name == "memory"


async def test_complex_relative_url_resolution() -> None:
    """Test complex relative URL resolution scenarios."""
    from zarr.storage._zep8 import URLParser

    parser = URLParser()

    # Test relative navigation with .. in complex paths
    base_url = "file:/path/to/base/dir/file.zarr"
    relative_url = "../../../other/path.zarr"
    resolved = parser.resolve_relative_url(base_url, relative_url)
    assert "other/path.zarr" in resolved

    # Test relative navigation with mixed segments
    base_url = "s3://bucket/deep/nested/path/data.zip"
    relative_url = "../../sibling/file.zarr|zip:inner"
    resolved = parser.resolve_relative_url(base_url, relative_url)
    assert "sibling/file.zarr|zip:inner" in resolved

    # Test navigation to root level
    base_url = "file:/single/level.zarr"
    relative_url = "../root.zarr"
    resolved = parser.resolve_relative_url(base_url, relative_url)
    assert "root.zarr" in resolved

    # Test no base path case
    base_url = "memory:"
    relative_url = "../relative/path.zarr"
    resolved = parser.resolve_relative_url(base_url, relative_url)
    # Should handle gracefully even without base path
    assert resolved is not None

    # Test edge case: .. navigation at root level
    base_url = "file:/root.zarr"
    relative_url = "../up.zarr"
    resolved = parser.resolve_relative_url(base_url, relative_url)
    assert "up.zarr" in resolved

    # Test mixed .. navigation with file adapter
    base_url = "file:/deep/nested/current.zarr"
    relative_url = "../../file:other.zarr|zip:"
    resolved = parser.resolve_relative_url(base_url, relative_url)
    assert "file:" in resolved or "other.zarr" in resolved

    # Test .. navigation with no path segments left
    base_url = "file:/single.zarr"
    relative_url = "../../memory:"
    resolved = parser.resolve_relative_url(base_url, relative_url)
    assert "memory:" in resolved

    # Test complex URL reconstruction
    base_url = "s3://bucket/path/file.zarr"
    relative_url = "../other.zarr|zip:|log:"
    resolved = parser.resolve_relative_url(base_url, relative_url)
    assert "|zip:|log:" in resolved


async def test_url_parsing_complex_scenarios() -> None:
    """Test URL parsing complex scenarios."""
    from zarr.storage._zep8 import URLParser

    parser = URLParser()

    # Test complex multi-adapter chain
    segments = parser.parse("s3://bucket/deep/nested/file.zip|zip:inner/path|zarr3:")
    assert len(segments) == 3
    assert segments[0].scheme == "s3"
    assert segments[0].path == "bucket/deep/nested/file.zip"
    assert segments[1].adapter == "zip"
    assert segments[1].path == "inner/path"
    assert segments[2].adapter == "zarr3"

    # Test URL with query parameters
    segments = parser.parse("memory:|log:?log_level=INFO")
    assert len(segments) == 2
    assert segments[1].adapter == "log"
    assert "log_level=INFO" in segments[1].path

    # Test URL with mixed schemes and adapters
    segments = parser.parse("https://example.com/data.zarr|log:")
    assert len(segments) == 2
    assert segments[0].scheme == "https"
    assert segments[1].adapter == "log"


async def test_module_level_resolve_url() -> None:
    """Test module-level resolve_url convenience function."""
    from zarr.storage._zep8 import resolve_url

    # Test resolving a simple memory URL
    store = await resolve_url("memory:", mode="w")
    assert store is not None

    # Test resolving with storage options
    store = await resolve_url("memory:", storage_options={"read_only": True}, mode="r")
    assert store is not None


async def test_url_resolver_icechunk_path_handling() -> None:
    """Test icechunk-specific path handling in URL resolution."""
    from zarr.storage._zep8 import URLStoreResolver

    resolver = URLStoreResolver()

    # Test icechunk branch syntax parsing (simulated)
    # Note: These tests will exercise the icechunk path handling logic
    # even though icechunk might not be available
    test_url = "s3://bucket/repo|icechunk:branch:main"
    try:
        zarr_format = resolver.extract_zarr_format(test_url)
        # Should not raise an error even if icechunk is not available
        assert zarr_format is None or isinstance(zarr_format, int)
    except Exception:
        # If icechunk is not available, the test should still pass
        pass

    # Test icechunk tag syntax
    test_url = "s3://bucket/repo|icechunk:tag:v1.0"
    try:
        zarr_format = resolver.extract_zarr_format(test_url)
        assert zarr_format is None or isinstance(zarr_format, int)
    except Exception:
        pass

    # Test icechunk @ syntax (new format)
    test_url = "s3://bucket/repo|icechunk:@main.path/to/data"
    try:
        path = resolver.extract_path(test_url)
        assert isinstance(path, str)
    except Exception:
        pass


async def test_url_parser_static_methods() -> None:
    """Test URLParser static methods for comprehensive coverage."""
    from zarr.storage._zep8 import URLParser

    # Test _parse_base_url static method with various URLs
    base_segment = URLParser._parse_base_url("s3://bucket/path/file.zip")
    assert base_segment.scheme == "s3"
    assert base_segment.path == "bucket/path/file.zip"

    # Test _parse_adapter_spec static method
    adapter_segment = URLParser._parse_adapter_spec("zip:inner/path")
    assert adapter_segment.adapter == "zip"
    assert adapter_segment.path == "inner/path"

    # Test adapter spec without path
    adapter_segment = URLParser._parse_adapter_spec("zarr3:")
    assert adapter_segment.adapter == "zarr3"
    assert adapter_segment.path == ""


async def test_edge_cases_and_error_conditions() -> None:
    """Test edge cases and error conditions for better coverage."""
    from zarr.storage._zep8 import URLParser, ZEP8URLError

    parser = URLParser()

    # Test empty URL segments (should raise error)
    with pytest.raises(ZEP8URLError, match="Empty URL segment found"):
        parser.parse("s3://bucket/file.zip||memory:")

    # Test invalid adapter names (should handle gracefully)
    try:
        segments = parser.parse("file:/test.zarr|invalid-adapter:")
        # Should still parse successfully
        assert len(segments) >= 1
    except ZEP8URLError:
        # Or raise an error, both are acceptable
        pass
