"""
Built-in store adapters for ZEP 8 URL syntax.

This module provides store adapters for common store types that are
built into zarr-python.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from zarr.abc.store_adapter import StoreAdapter
from zarr.storage._local import LocalStore
from zarr.storage._memory import MemoryStore

if TYPE_CHECKING:
    from typing import Any

    from zarr.abc.store import Store
    from zarr.abc.store_adapter import URLSegment

__all__ = ["FileSystemAdapter", "GCSAdapter", "HttpsAdapter", "MemoryAdapter", "S3Adapter"]


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
        # For memory scheme, the preceding_url should be 'memory:'
        if preceding_url != "memory:":
            raise ValueError(f"Expected memory: URL, got: {preceding_url}")

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


class HttpsAdapter(StoreAdapter):
    """Store adapter for HTTPS URLs using fsspec."""

    adapter_name = "https"

    @classmethod
    async def from_url_segment(
        cls,
        segment: URLSegment,
        preceding_url: str,
        **kwargs: Any,
    ) -> Store:
        """Create an FsspecStore for HTTPS URLs."""
        from zarr.storage._fsspec import FsspecStore

        # For https scheme, use the full preceding URL
        if not preceding_url.startswith(("http://", "https://")):
            raise ValueError(f"Expected HTTP/HTTPS URL, got: {preceding_url}")

        # Extract storage options
        storage_options = kwargs.get("storage_options", {})
        read_only = storage_options.get("read_only", True)  # HTTPS is typically read-only

        # Create fsspec store
        return FsspecStore.from_url(
            preceding_url, storage_options=storage_options, read_only=read_only
        )

    @classmethod
    def can_handle_scheme(cls, scheme: str) -> bool:
        return scheme in ("http", "https")

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        return ["http", "https"]


class S3Adapter(StoreAdapter):
    """Store adapter for S3 URLs using fsspec."""

    adapter_name = "s3"

    @classmethod
    async def from_url_segment(
        cls,
        segment: URLSegment,
        preceding_url: str,
        **kwargs: Any,
    ) -> Store:
        """Create an FsspecStore for S3 URLs."""
        from zarr.storage._fsspec import FsspecStore

        # For s3 scheme, use the full preceding URL
        if not preceding_url.startswith("s3://"):
            raise ValueError(f"Expected s3:// URL, got: {preceding_url}")

        # Extract storage options
        storage_options = kwargs.get("storage_options", {})
        read_only = storage_options.get("read_only", False)
        if "mode" in kwargs:
            mode = kwargs["mode"]
            read_only = mode == "r"

        # Create fsspec store
        return FsspecStore.from_url(
            preceding_url, storage_options=storage_options, read_only=read_only
        )

    @classmethod
    def can_handle_scheme(cls, scheme: str) -> bool:
        return scheme == "s3"

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        return ["s3"]


class GCSAdapter(StoreAdapter):
    """Store adapter for Google Cloud Storage URLs using fsspec."""

    adapter_name = "gcs"

    @classmethod
    async def from_url_segment(
        cls,
        segment: URLSegment,
        preceding_url: str,
        **kwargs: Any,
    ) -> Store:
        """Create an FsspecStore for GCS URLs."""
        from zarr.storage._fsspec import FsspecStore

        # For gcs scheme, use the full preceding URL
        if not preceding_url.startswith(("gcs://", "gs://")):
            raise ValueError(f"Expected gcs:// or gs:// URL, got: {preceding_url}")

        # Extract storage options
        storage_options = kwargs.get("storage_options", {})
        read_only = storage_options.get("read_only", False)
        if "mode" in kwargs:
            mode = kwargs["mode"]
            read_only = mode == "r"

        # Normalize URL to gs:// (fsspec standard)
        url = preceding_url
        if url.startswith("gcs://"):
            url = "gs://" + url[6:]

        return FsspecStore.from_url(url, storage_options=storage_options, read_only=read_only)

    @classmethod
    def can_handle_scheme(cls, scheme: str) -> bool:
        return scheme in ("gcs", "gs")

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        return ["gcs", "gs"]


# Additional adapter for gs scheme (alias for gcs)
class GSAdapter(GCSAdapter):
    """Alias adapter for gs:// URLs (same as gcs)."""

    adapter_name = "gs"
