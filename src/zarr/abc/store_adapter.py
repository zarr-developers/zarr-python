"""
Store adapter interface for ZEP 8 URL syntax support.

This module defines the protocol that store implementations must follow
to be usable in ZEP 8 URL chains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from zarr.abc.store import Store

__all__ = ["StoreAdapter", "URLSegment"]


@dataclass(frozen=True)
class URLSegment:
    """
    Represents a segment in a ZEP 8 URL chain.

    Examples:
    - "zip:" -> URLSegment(scheme=None, adapter="zip", path="")
    - "s3://bucket/data" -> URLSegment(scheme="s3", adapter=None, path="bucket/data")
    - "zip:inner/path" -> URLSegment(scheme=None, adapter="zip", path="inner/path")
    """

    scheme: str | None = None
    """The URL scheme (e.g., 's3', 'file', 'https') for the first segment."""

    adapter: str | None = None
    """The store adapter name (e.g., 'zip', 'icechunk', 'zarr3')."""

    path: str = ""
    """Path component for the segment."""

    def __post_init__(self) -> None:
        """Validate the URL segment."""
        import re

        from zarr.storage._zep8 import ZEP8URLError

        if not self.scheme and not self.adapter:
            raise ZEP8URLError("URL segment must have either scheme or adapter")
        if self.adapter and not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", self.adapter):
            raise ZEP8URLError(f"Invalid adapter name: {self.adapter}")


class StoreAdapter(ABC):
    """
    Abstract base class for store adapters that can be resolved from ZEP 8 URLs.

    Store adapters enable stores to participate in ZEP 8 URL chains by implementing
    the from_url_segment class method. This allows stores to be created from URL
    components and optionally wrap or chain with other stores.

    Examples
    --------
    A memory adapter that creates in-memory storage:

    >>> class MemoryAdapter(StoreAdapter):
    ...     adapter_name = "memory"
    ...
    ...     @classmethod
    ...     async def from_url_segment(cls, segment, preceding_url, **kwargs):
    ...         from zarr.storage import MemoryStore
    ...         return await MemoryStore.open()

    An icechunk adapter that uses native icechunk storage:

    >>> class IcechunkAdapter(StoreAdapter):
    ...     adapter_name = "icechunk"
    ...
    ...     @classmethod
    ...     async def from_url_segment(cls, segment, preceding_url, **kwargs):
    ...         import icechunk
    ...         if preceding_url.startswith('s3://'):
    ...             storage = icechunk.s3_storage(bucket='...', prefix='...')
    ...         elif preceding_url.startswith('file:'):
    ...             storage = icechunk.local_filesystem_storage(path='...')
    ...         repo = icechunk.Repository.open_existing(storage)
    ...         return repo.readonly_session('main').store
    """

    # Class-level registration info
    adapter_name: str
    """The name used to identify this adapter in URLs (e.g., 'zip', 'icechunk')."""

    @classmethod
    @abstractmethod
    async def from_url_segment(
        cls,
        segment: URLSegment,
        preceding_url: str,
        **kwargs: Any,
    ) -> Store:
        """
        Create a store from a URL segment and preceding URL.

        This method is the core of the store adapter interface. It receives
        a URL segment and the full preceding URL, allowing each adapter to
        use its native storage implementations.

        Parameters
        ----------
        segment : URLSegment
            The URL segment containing adapter name and optional path.
        preceding_url : str
            The full URL before this adapter segment (e.g., 'file:/path', 's3://bucket/key').
            This allows the adapter to use its native storage implementations.
        **kwargs : Any
            Additional keyword arguments from the URL resolution context,
            such as storage_options, mode, etc.

        Returns
        -------
        Store
            A configured store instance ready for use.

        Raises
        ------
        ValueError
            If required parameters are missing or invalid.
        NotImplementedError
            If the adapter cannot handle the given configuration.

        Notes
        -----
        This design allows each adapter to interpret the preceding URL using its own
        native storage backends. For example:
        - Icechunk adapter can use icechunk.s3_storage() for s3:// URLs
        - ZIP adapter can use fsspec for remote file access
        - Each adapter maintains full control over its storage layer

        Examples
        --------
        For URL "file:/tmp/repo|icechunk:branch:main":
        - segment.adapter = "icechunk"
        - segment.path = "branch:main"
        - preceding_url = "file:/tmp/repo"
        """
        ...

    @classmethod
    def can_handle_scheme(cls, scheme: str) -> bool:
        """
        Check if this adapter can handle a given URL scheme.

        This method allows adapters to indicate they can handle
        specific URL schemes directly, even when not in a ZEP 8 chain.

        Parameters
        ----------
        scheme : str
            The URL scheme to check (e.g., 's3', 'https', 'file').

        Returns
        -------
        bool
            True if this adapter can handle the scheme.
        """
        return False

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        """
        Get list of URL schemes this adapter supports.

        Returns
        -------
        list[str]
            List of supported URL schemes.
        """
        return []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate adapter implementation on subclass creation."""
        super().__init_subclass__(**kwargs)

        # Ensure adapter_name is defined
        if not hasattr(cls, "adapter_name") or not cls.adapter_name:
            raise TypeError(f"StoreAdapter subclass {cls.__name__} must define 'adapter_name'")

        # Validate adapter_name format
        if not isinstance(cls.adapter_name, str):
            raise TypeError(f"adapter_name must be a string, got {type(cls.adapter_name)}")

        import re

        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_+-]*$", cls.adapter_name):
            raise ValueError(f"Invalid adapter_name format: {cls.adapter_name}")
