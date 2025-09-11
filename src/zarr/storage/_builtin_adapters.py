"""
Built-in store adapters for ZEP 8 URL syntax.

This module provides store adapters for common store types that are
built into zarr-python.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import parse_qs, urlparse

from zarr.abc.store_adapter import StoreAdapter
from zarr.storage._local import LocalStore
from zarr.storage._logging import LoggingStore
from zarr.storage._memory import MemoryStore
from zarr.storage._zep8 import URLStoreResolver
from zarr.storage._zip import ZipStore

if TYPE_CHECKING:
    from typing import Any

    from zarr.abc.store import Store
    from zarr.abc.store_adapter import URLSegment

__all__ = [
    "FileSystemAdapter",
    "HttpsAdapter",
    "LoggingAdapter",
    "MemoryAdapter",
    "RemoteAdapter",
    "S3Adapter",
    "ZipAdapter",
]


class FileSystemAdapter(StoreAdapter):
    """Store adapter for local filesystem access."""

    adapter_name = "file"

    @classmethod
    async def from_url_segment(
        cls,
        segment: URLSegment,
        preceding_url: str,
        **kwargs: Any,
    ) -> Store:
        """Create a LocalStore from a file URL segment."""
        # For file scheme, the preceding_url should be the full file: URL
        if not preceding_url.startswith("file:"):
            raise ValueError(f"Expected file: URL, got: {preceding_url}")

        # Extract path from preceding URL
        path = preceding_url[5:]  # Remove 'file:' prefix
        if not path:
            path = "."

        # Determine read-only mode
        read_only = kwargs.get("storage_options", {}).get("read_only", False)
        if "mode" in kwargs:
            mode = kwargs["mode"]
            read_only = mode == "r"

        return await LocalStore.open(root=Path(path), read_only=read_only)

    @classmethod
    def can_handle_scheme(cls, scheme: str) -> bool:
        return scheme == "file"

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        return ["file"]


class MemoryAdapter(StoreAdapter):
    """Store adapter for in-memory storage."""

    adapter_name = "memory"

    @classmethod
    async def from_url_segment(
        cls,
        segment: URLSegment,
        preceding_url: str,
        **kwargs: Any,
    ) -> Store:
        """Create a MemoryStore from a memory URL segment."""
        # For memory scheme, the preceding_url should start with 'memory:'
        if not preceding_url.startswith("memory:"):
            raise ValueError(f"Expected memory: URL, got: {preceding_url}")

        if segment.path:
            raise ValueError("Memory store does not currently support sub-paths")

        # Determine read-only mode
        read_only = kwargs.get("storage_options", {}).get("read_only", False)
        if "mode" in kwargs:
            mode = kwargs["mode"]
            read_only = mode == "r"

        return await MemoryStore.open(read_only=read_only)

    @classmethod
    def can_handle_scheme(cls, scheme: str) -> bool:
        return scheme == "memory"

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        return ["memory"]


class RemoteAdapter(StoreAdapter):
    """Universal store adapter for remote URLs using fsspec.

    Supports any URL scheme that fsspec can handle, including:
    - S3: s3://bucket/path
    - Google Cloud Storage: gs://bucket/path
    - HTTP(S): https://example.com/file.zip, http://example.com/file.zip
    - Azure: abfs://container/path, az://container/path
    - And many more via fsspec ecosystem
    """

    adapter_name = "remote"

    @classmethod
    async def from_url_segment(
        cls,
        segment: URLSegment,
        preceding_url: str,
        **kwargs: Any,
    ) -> Store:
        """Create an FsspecStore for any remote URL."""
        from zarr.storage._fsspec import FsspecStore

        # Validate that it's a remote URL
        if not cls._is_remote_url(preceding_url):
            raise ValueError(f"Expected remote URL, got: {preceding_url}")

        # Extract and validate scheme
        scheme = cls._extract_scheme(preceding_url)
        cls._validate_scheme(scheme)

        # Process storage options
        storage_options = kwargs.get("storage_options", {})
        read_only = cls._determine_read_only_mode(preceding_url, **kwargs)

        # Apply scheme-specific URL normalization
        normalized_url = cls._normalize_url(preceding_url, scheme)

        return FsspecStore.from_url(
            normalized_url, storage_options=storage_options, read_only=read_only
        )

    @classmethod
    def _is_remote_url(cls, url: str) -> bool:
        """Check if URL is a supported remote URL."""
        return "://" in url and not url.startswith("file:")

    @classmethod
    def _extract_scheme(cls, url: str) -> str:
        """Extract scheme from URL."""
        return url.split("://", 1)[0]

    @classmethod
    def _validate_scheme(cls, scheme: str) -> None:
        """Validate that the scheme is supported."""
        supported = cls.get_supported_schemes()
        if scheme not in supported:
            raise ValueError(
                f"Unsupported scheme '{scheme}'. Supported schemes: {', '.join(supported)}"
            )

    @classmethod
    def _determine_read_only_mode(cls, url: str, **kwargs: Any) -> bool:
        """Determine read-only mode with scheme-specific defaults."""
        # Check explicit mode in kwargs
        if "mode" in kwargs:
            return bool(kwargs["mode"] == "r")

        # Check storage_options
        storage_options = kwargs.get("storage_options", {})
        if "read_only" in storage_options:
            return bool(storage_options["read_only"])

        # Scheme-specific defaults
        scheme = cls._extract_scheme(url)
        return scheme in ("http", "https")  # HTTP(S) typically read-only, others can be writable

    @classmethod
    def _normalize_url(cls, url: str, scheme: str) -> str:
        """Apply scheme-specific URL normalization."""
        return url

    @classmethod
    def can_handle_scheme(cls, scheme: str) -> bool:
        return scheme in cls.get_supported_schemes()

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        return [
            "s3",
            "gs",
            "http",
            "https",  # HTTP(S)
            "abfs",
            "az",  # Azure Blob Storage
            "ftp",
            "ftps",  # FTP
            # Could be extended with more fsspec-supported schemes
        ]


class HttpsAdapter(RemoteAdapter):
    """Store adapter for HTTP(S) URLs using fsspec."""

    adapter_name = "https"

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        return ["http", "https"]


class S3Adapter(RemoteAdapter):
    """Store adapter for S3 URLs using fsspec."""

    adapter_name = "s3"

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        return ["s3"]


class GSAdapter(RemoteAdapter):
    """Store adapter for Google Cloud Storage URLs using fsspec."""

    adapter_name = "gs"

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        return ["gs"]


class LoggingAdapter(StoreAdapter):
    """Store adapter that wraps any other store with logging capabilities.

    This adapter enables logging of all store operations by wrapping the result
    of any preceding URL chain with a LoggingStore. It can be used to debug
    and monitor store operations.

    Examples
    --------
    >>> import zarr
    >>> # Log all operations on a memory store
    >>> store = zarr.open("memory:|log:", mode="w")
    >>>
    >>> # Log operations on a remote S3 store
    >>> store = zarr.open("s3://bucket/data.zarr|log:", mode="r")
    >>>
    >>> # Log operations with custom log level
    >>> store = zarr.open("file:/tmp/data.zarr|log:?log_level=INFO", mode="r")
    """

    adapter_name = "log"

    @classmethod
    async def from_url_segment(
        cls,
        segment: URLSegment,
        preceding_url: str,
        **kwargs: Any,
    ) -> Store:
        """Create a LoggingStore that wraps a store created from the preceding URL.

        Parameters
        ----------
        segment : URLSegment
            The URL segment with adapter='log'. The segment path can contain
            query parameters for configuring the logging behavior:
            - log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        preceding_url : str
            The URL for the store to wrap with logging
        **kwargs : Any
            Additional arguments passed through to the wrapped store

        Returns
        -------
        Store
            A LoggingStore wrapping the store created from preceding_url
        """

        # Parse query parameters from the segment path for logging configuration
        log_level = "DEBUG"  # default
        log_handler = None

        if segment.path and "?" in segment.path:
            # Parse the segment path as a URL to extract query parameters
            parsed = urlparse(f"log:{segment.path}")
            query_params = parse_qs(parsed.query)

            if "log_level" in query_params:
                log_level = query_params["log_level"][0].upper()

        # Create the underlying store from the preceding URL
        resolver = URLStoreResolver()
        underlying_store = await resolver.resolve_url(preceding_url, **kwargs)

        # Wrap it with logging
        return LoggingStore(store=underlying_store, log_level=log_level, log_handler=log_handler)

    @classmethod
    def can_handle_scheme(cls, scheme: str) -> bool:
        # LoggingAdapter doesn't handle schemes directly, it wraps other stores
        return False

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        return []


class ZipAdapter(StoreAdapter):
    """Store adapter for ZIP file access supporting both local and remote storage."""

    adapter_name = "zip"

    @classmethod
    async def from_url_segment(
        cls,
        segment: URLSegment,
        preceding_url: str,
        **kwargs: Any,
    ) -> Store:
        """Create a ZipStore from a ZIP URL segment.

        Supports:
        - Local paths: /path/to/file.zip
        - File URLs: file:/path/to/file.zip
        - Remote URLs: s3://bucket/file.zip, https://example.com/file.zip, gs://bucket/file.zip
        """
        # Determine read-only mode
        read_only = kwargs.get("storage_options", {}).get("read_only", True)
        if "mode" in kwargs:
            mode = kwargs["mode"]
            read_only = mode == "r"

        # Handle different URL types
        if cls._is_remote_url(preceding_url):
            # For remote URLs, we need to create a custom ZipStore that can handle remote files
            return await cls._create_remote_zip_store(preceding_url, segment, read_only, kwargs)
        elif preceding_url.startswith("file:"):
            # Convert file: URL to local path
            zip_file_path = preceding_url[5:]
        else:
            # Assume it's already a local path
            zip_file_path = preceding_url

        # Create local ZipStore instance
        # Use the same mode as requested, defaulting to "w" for write operations
        zip_mode_raw = "r" if read_only else kwargs.get("mode", "w")
        # For ZIP files, we need to map zarr modes to appropriate ZIP modes
        if zip_mode_raw == "w-":
            zip_mode = "w"  # ZIP doesn't support "w-", use "w" instead
        elif zip_mode_raw in ("r", "w", "a"):
            zip_mode = zip_mode_raw
        else:
            zip_mode = "w"  # Default fallback

        zip_store = ZipStore(
            path=zip_file_path, mode=cast("Literal['r', 'w', 'a']", zip_mode), read_only=read_only
        )

        # Open the store
        await zip_store._open()

        # If there's a path specified in the segment, we need to handle it
        # For now, ZipStore doesn't support sub-paths within ZIP files in the constructor
        # This would need to be handled by the zarr path resolution

        return zip_store

    @classmethod
    def _is_remote_url(cls, url: str) -> bool:
        """Check if the URL is a remote URL that needs special handling."""
        return "://" in url and not url.startswith("file:")

    @classmethod
    async def _create_remote_zip_store(
        cls,
        remote_url: str,
        segment: URLSegment,
        read_only: bool,
        kwargs: dict[str, Any],
    ) -> Store:
        """Create a ZipStore for remote URLs using fsspec.

        Uses fsspec's remote file object directly with ZipFile for efficient access
        without downloading the entire file.
        """
        # Import fsspec for remote file access
        try:
            import fsspec
        except ImportError as e:
            raise ValueError(
                f"fsspec is required for remote ZIP access but is not installed. "
                f"Install with: pip install fsspec[{cls._get_fsspec_protocol(remote_url)}]"
            ) from e

        # Extract storage options for fsspec
        storage_options = kwargs.get("storage_options", {})

        # Open the remote file using fsspec
        # The file object will be seekable and can be used directly with ZipFile
        remote_file_opener = fsspec.open(remote_url, "rb", **storage_options)

        # Create a ZipStore that uses the remote file object directly
        zip_store = ZipStore(
            path=remote_file_opener,  # Pass file opener instead of path
            mode="r",
            read_only=True,
        )

        # Open the store
        await zip_store._open()

        return zip_store

    @classmethod
    def _get_fsspec_protocol(cls, url: str) -> str:
        """Get the fsspec protocol name for installation hints."""
        if url.startswith("s3://"):
            return "s3"
        elif url.startswith("gs://"):
            return "gs"
        elif url.startswith(("http://", "https://")):
            return "http"
        else:
            return "full"

    @classmethod
    def can_handle_scheme(cls, scheme: str) -> bool:
        return False  # ZIP is an adapter, not a scheme

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        return []
