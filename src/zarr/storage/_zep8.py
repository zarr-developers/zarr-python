"""
ZEP 8 URL syntax parsing and store resolution.

This module implements the ZEP 8 URL syntax specification for zarr-python,
enabling pipe-separated store chaining and third-party store integration.
It provides both URL parsing capabilities and store resolution.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from zarr.abc.store_adapter import URLSegment
from zarr.registry import get_store_adapter

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.common import ZarrFormat

__all__ = [
    "URLParser",
    "URLStoreResolver",
    "ZEP8URLError",
    "is_zep8_url",
    "resolve_url",
]


class ZEP8URLError(ValueError):
    """Exception raised for invalid ZEP 8 URL syntax."""


class URLParser:
    """Parse ZEP 8 URL syntax into components."""

    def parse(self, url: str) -> list[URLSegment]:
        """
        Parse a ZEP 8 URL into ordered list of segments.

        Parameters
        ----------
        url : str
            ZEP 8 URL to parse (e.g., "s3://bucket/data.zip|zip:|zarr3:")

        Returns
        -------
        List[URLSegment]
            Ordered list of URL segments representing the adapter chain.

        Examples
        --------
        >>> parser = URLParser()
        >>> segments = parser.parse("file:///data.zip|zip:inner|zarr3:")
        >>> segments[0].scheme
        'file'
        >>> segments[1].adapter
        'zip'
        >>> segments[1].path
        'inner'
        >>> segments[2].adapter
        'zarr3'
        """
        if not url:
            raise ZEP8URLError("URL cannot be empty")

        if url.startswith("|"):
            raise ZEP8URLError("URL cannot start with pipe")

        # Split on pipe characters
        parts = url.split("|")
        segments = []

        for i, part in enumerate(parts):
            if not part.strip():
                raise ZEP8URLError("Empty URL segment found")

            if i == 0:
                # First part is the base URL/path
                segments.append(self._parse_base_url(part))
            else:
                # Subsequent parts are adapter specifications
                segments.append(self._parse_adapter_spec(part))

        return segments

    @staticmethod
    def _is_windows_path(url: str) -> bool:
        r"""Check if URL is a Windows absolute path like C:\... or C:/..."""
        return re.match(r"^[A-Za-z]:[/\\]", url) is not None

    @staticmethod
    def _parse_base_url(url: str) -> URLSegment:
        """Parse the base URL component."""
        # Fast path: Standard URLs with :// (most common: s3://, https://, file://)
        if "://" in url:
            parsed = urlparse(url)
            path = f"{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
            return URLSegment(scheme=parsed.scheme, path=path)

        # Check for Windows paths before other colon processing (C:\, D:\, etc.)
        if URLParser._is_windows_path(url):
            return URLSegment(scheme="file", path=url)

        # Adapter syntax with colon but no :// (memory:, zip:path, file:path)
        if ":" in url:
            scheme_or_adapter, path = url.split(":", 1)
            if scheme_or_adapter == "file":
                return URLSegment(scheme="file", path=path)
            else:
                return URLSegment(adapter=scheme_or_adapter, path=path)

        # Plain filesystem path (no colon at all)
        return URLSegment(scheme="file", path=url)

    @staticmethod
    def _parse_adapter_spec(spec: str) -> URLSegment:
        """Parse an adapter specification like 'zip:path' or 'zarr3:'."""
        if not spec:
            raise ZEP8URLError("Empty adapter specification")

        if ":" in spec:
            adapter, path_part = spec.split(":", 1)
            path = path_part if path_part else ""
        else:
            # No colon - treat entire spec as adapter name
            adapter = spec
            path = ""

        return URLSegment(adapter=adapter, path=path)

    def resolve_relative(self, base: URLSegment, relative_path: str) -> URLSegment:
        """
        Resolve a relative path against a base URLSegment.

        Parameters
        ----------
        base : URLSegment
            Base URL segment to resolve against.
        relative_path : str
            Relative path to resolve.

        Returns
        -------
        URLSegment
            New URLSegment with resolved path.
        """
        if not relative_path:
            return base

        if relative_path.startswith("/"):
            # Absolute path - replace base path
            return URLSegment(scheme=base.scheme, adapter=base.adapter, path=relative_path)

        # Relative path - combine with base path
        base_path = base.path
        if base_path and not base_path.endswith("/"):
            base_path += "/"

        new_path = base_path + relative_path
        return URLSegment(scheme=base.scheme, adapter=base.adapter, path=new_path)

    @staticmethod
    def resolve_relative_url(base_url: str, relative_url: str) -> str:
        """
        Resolve relative URLs using Unix-style .. syntax.

        Currently only supports Unix-style relative paths (e.g., "../path/file.zarr").
        Pipe-prefixed relative URLs (e.g., "|..|file.zarr") are not implemented.

        Parameters
        ----------
        base_url : str
            The base ZEP 8 URL to resolve against.
        relative_url : str
            Unix-style relative URL with .. components.

        Returns
        -------
        str
            The resolved URL (currently just returns relative_url unchanged).

        Examples
        --------
        >>> URLParser.resolve_relative_url(
        ...     "s3://bucket/data/exp1.zip|zip:|zarr3:",
        ...     "../control.zip|zip:|zarr3:"
        ... )
        '../control.zip|zip:|zarr3:'
        """
        # Currently only supports Unix-style relative paths by returning them unchanged
        # TODO: Implement proper relative URL resolution if needed
        return relative_url


def is_zep8_url(url: Any) -> bool:
    """
    Check if a string is a ZEP 8 URL.

    According to ZEP 8, all URLs with schemes (like s3://, file://, https://)
    are valid ZEP 8 URLs, either as single segments or chained with adapters.

    Returns True for:
    - Chained URLs with pipe separators: "s3://bucket/data.zip|zip:"
    - Simple scheme URLs: "s3://bucket/path", "file:///data", "https://example.com"
    - Adapter-only URLs: "memory:", "zip:path"

    Returns False for:
    - Plain file paths: "/absolute/path", "relative/path"
    - Empty or non-string inputs
    - Windows drive letters: "C:/path"

    Parameters
    ----------
    url : str
        String to check.

    Returns
    -------
    bool
        True if the string appears to be a ZEP 8 URL.

    Examples
    --------
    >>> is_zep8_url("s3://bucket/data.zip|zip:|zarr3:")
    True
    >>> is_zep8_url("memory:")
    True
    >>> is_zep8_url("s3://bucket/data.zarr")
    True
    >>> is_zep8_url("file:///data.zarr")
    True
    >>> is_zep8_url("/absolute/path")
    False
    >>> is_zep8_url("relative/path")
    False
    """
    if not url or not isinstance(url, str):
        return False

    # Check for pipe character (chained URLs) - definitely ZEP 8
    if "|" in url:
        return True

    # Check for scheme URLs (scheme://) - these are ZEP 8 URLs
    # TODO: Consider optimizing file:// and local:// to use LocalStore directly
    # instead of routing through FsspecStore for better performance.
    if "://" in url:
        # Exclude Windows UNC paths (//server/share)
        return not url.startswith("//")

    # Check for simple adapter syntax (single colon without ://)
    # Examples: "memory:", "zip:path", or custom scheme adapters
    # TODO: Consider checking against registered adapters via get_store_adapter(scheme)
    # to return True only for known adapters. This would make is_zep8_url() more accurate
    # but adds a dependency on the registry. Current approach is permissive - any
    # valid-looking scheme returns True, runtime will fail if adapter not registered.
    # Trade-off: Syntax check (current) vs semantic check (proposed).
    if ":" in url:
        parts = url.split(":", 1)
        if len(parts) == 2:
            scheme = parts[0]
            path_part = parts[1]

            # Exclude Windows drive letters (C:\path or D:/path)
            # But allow single letters as adapter names (z:, a:) if not followed by path separator
            if (
                len(scheme) == 1
                and scheme.isalpha()
                and path_part
                and path_part.startswith(("/", "\\"))
            ):
                return False  # Windows path like C:/path or D:\path

            # If scheme looks like a valid adapter name, it's ZEP 8
            # Valid adapter names are alphanumeric with optional underscores/hyphens
            if scheme and (scheme.isalnum() or scheme.replace("_", "").replace("-", "").isalnum()):
                return True

    return False


class URLStoreResolver:
    """
    Resolve ZEP 8 URLs to stores.

    This class handles the conversion of ZEP 8 URL syntax into store chains,
    processing each segment in order and chaining stores together.

    Examples
    --------
    >>> resolver = URLStoreResolver()
    >>> store = await resolver.resolve_url("file:///data.zip|zip:|zarr3:")
    >>> isinstance(store, ZipStore)
    True

    >>> zarr_format = resolver.extract_zarr_format("file:///data|zarr3:")
    >>> zarr_format
    3
    """

    def __init__(self) -> None:
        self.parser = URLParser()

    async def resolve_url_with_path(
        self, url: str, storage_options: dict[str, Any] | None = None, **kwargs: Any
    ) -> tuple[Store, str, ZarrFormat | None]:
        """
        Resolve a ZEP 8 URL to a store and extract the zarr path and format in one pass.

        This is more efficient than calling resolve_url(), extract_path(), and
        extract_zarr_format() separately since it only parses the URL once.

        Parameters
        ----------
        url : str
            ZEP 8 URL (with pipes) or simple scheme URL to resolve.
        storage_options : dict, optional
            Storage options to pass to store adapters.
        **kwargs : Any
            Additional keyword arguments to pass to store adapters.

        Returns
        -------
        tuple[Store, str, ZarrFormat | None]
            The resolved store, the extracted zarr path, and the zarr format (2, 3, or None).

        Raises
        ------
        ValueError
            If the URL is malformed or contains unsupported segments.
        KeyError
            If a required store adapter is not registered.
        """
        # Validate that this is a ZEP 8 URL
        if not is_zep8_url(url):
            raise ValueError(f"Not a valid URL: {url}")

        # Parse the URL into segments (only once!)
        segments = self.parser.parse(url)

        if not segments:
            raise ValueError(f"Empty URL segments in: {url}")

        # Extract path and format from segments
        zarr_path = self._extract_path_from_segments(segments)
        zarr_format = self._extract_format_from_segments(segments)

        # Resolve store from segments
        store = await self._resolve_store_from_segments(
            url, segments, storage_options=storage_options, **kwargs
        )

        return store, zarr_path, zarr_format

    async def resolve_url(
        self, url: str, storage_options: dict[str, Any] | None = None, **kwargs: Any
    ) -> Store:
        """
        Resolve a ZEP 8 URL or simple scheme URL to a store.

        Parameters
        ----------
        url : str
            ZEP 8 URL (with pipes) or simple scheme URL to resolve.
        storage_options : dict, optional
            Storage options to pass to store adapters.
        **kwargs : Any
            Additional keyword arguments to pass to store adapters.

        Returns
        -------
        Store
            The resolved store at the end of the chain.

        Raises
        ------
        ValueError
            If the URL is malformed or contains unsupported segments.
        KeyError
            If a required store adapter is not registered.
        """
        # Validate that this is a ZEP 8 URL
        # Note: make_store_path() already checks this before calling us, but we validate
        # again here since resolve_url() is a public API that can be called directly
        if not is_zep8_url(url):
            raise ValueError(f"Not a valid URL: {url}")

        # Parse the URL into segments
        segments = self.parser.parse(url)

        if not segments:
            raise ValueError(f"Empty URL segments in: {url}")

        # Delegate to helper method
        return await self._resolve_store_from_segments(
            url, segments, storage_options=storage_options, **kwargs
        )

    async def _resolve_store_from_segments(
        self,
        url: str,
        segments: list[URLSegment],
        storage_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Store:
        """
        Internal helper to resolve a store from parsed URL segments.

        Parameters
        ----------
        url : str
            Original URL (for error messages).
        segments : list[URLSegment]
            Parsed URL segments.
        storage_options : dict, optional
            Storage options to pass to store adapters.
        **kwargs : Any
            Additional keyword arguments to pass to store adapters.

        Returns
        -------
        Store
            The resolved store at the end of the chain.
        """
        # Validate all adapters are registered BEFORE creating any stores
        # This prevents side effects (disk writes, network calls) before discovering missing adapters

        for segment in segments:
            adapter_name = segment.adapter or segment.scheme
            if adapter_name and adapter_name not in ("zarr2", "zarr3"):
                try:
                    get_store_adapter(adapter_name)
                except KeyError:
                    raise ValueError(
                        f"Unknown adapter '{adapter_name}' in URL: {url}. "
                        f"Ensure the required package is installed and provides "
                        f'an entry point under [project.entry-points."zarr.stores"].'
                    ) from None

        # Process segments in order, building preceding URL for each adapter
        current_store: Store | None = None

        # Build list of segments that create stores (excluding zarr format segments and path-only segments)
        store_segments = []
        for i, segment in enumerate(segments):
            if segment.adapter in ("zarr2", "zarr3"):
                # Skip zarr format segments - they don't create stores
                # TODO: these should propagate to the open call somehow
                continue

            # Check if this is a path segment that should be consumed by the next adapter
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                # Format segments (zarr2, zarr3) don't consume paths, so skip them when checking
                if (
                    segment.scheme
                    and not segment.adapter
                    and next_segment.adapter
                    and next_segment.adapter not in ("zarr2", "zarr3")
                ):
                    # This segment provides a path for the next adapter, skip it as a store creator
                    # Example: in "file:data.zip|zip", "file:data.zip" is just a path for the zip adapter
                    continue

            store_segments.append(segment)

        # Process each store-creating segment
        for segment in store_segments:
            # Determine the adapter name to use
            adapter_name = segment.adapter or segment.scheme
            if not adapter_name:
                raise ValueError(f"Segment has neither adapter nor scheme: {segment}")

            # Get the store adapter class
            try:
                adapter_cls = get_store_adapter(adapter_name)
            except KeyError:
                raise ValueError(
                    f"Unknown store adapter '{adapter_name}' in URL: {url}. "
                    f"Ensure the required package is installed and provides "
                    f'an entry point under [project.entry-points."zarr.stores"].'
                ) from None

            # Build preceding URL - need to find this segment in the original segments list
            # and include all segments before it (including skipped path-only segments)
            segment_index_in_original = -1
            for orig_i, orig_segment in enumerate(segments):
                if orig_segment is segment:
                    segment_index_in_original = orig_i
                    break

            if segment_index_in_original <= 0:
                # This is the first segment or we couldn't find it, build from current segment
                if segment.scheme:
                    # Handle schemes that need :// vs :
                    if segment.scheme in (
                        "s3",
                        "s3+http",
                        "s3+https",
                        "gcs",
                        "gs",
                        "http",
                        "https",
                        "ftp",
                        "ftps",
                    ):  # pragma: no cover
                        preceding_url = f"{segment.scheme}://{segment.path}"  # pragma: no cover
                    else:
                        preceding_url = f"{segment.scheme}:{segment.path}"
                elif segment.adapter:
                    # First segment is an adapter (e.g., "memory:")
                    preceding_url = f"{segment.adapter}:{segment.path}"
                else:  # pragma: no cover
                    # This shouldn't happen for first segment but handle gracefully  # pragma: no cover
                    preceding_url = segment.path  # pragma: no cover
            else:
                # Build preceding URL from all original segments before this one
                preceding_segments = segments[:segment_index_in_original]
                preceding_parts = []

                for prev_segment in preceding_segments:
                    if prev_segment.scheme:
                        # Handle schemes that need :// vs :
                        if prev_segment.scheme in (
                            "s3",
                            "s3+http",
                            "s3+https",
                            "gcs",
                            "gs",
                            "http",
                            "https",
                            "ftp",
                            "ftps",
                        ):  # pragma: no cover
                            preceding_parts.append(
                                f"{prev_segment.scheme}://{prev_segment.path}"
                            )  # pragma: no cover
                        else:
                            preceding_parts.append(f"{prev_segment.scheme}:{prev_segment.path}")
                    elif prev_segment.adapter:
                        # Adapter segment - reconstruct format
                        preceding_parts.append(f"{prev_segment.adapter}:{prev_segment.path}")

                preceding_url = "|".join(preceding_parts)

            # Create the store using the adapter with preceding URL
            store_kwargs = kwargs.copy()
            if storage_options:
                store_kwargs["storage_options"] = storage_options

            current_store = await adapter_cls.from_url_segment(
                segment, preceding_url=preceding_url, **store_kwargs
            )

        if current_store is None:
            raise ValueError(f"URL resolved to no store: {url}")

        return current_store

    def extract_zarr_format(self, url: str) -> int | None:
        """
        Extract zarr format from URL (zarr2: or zarr3:).

        Parameters
        ----------
        url : str
            ZEP 8 URL to analyze.

        Returns
        -------
        int or None
            The zarr format version (2 or 3), or None if not specified.

        Examples
        --------
        >>> resolver = URLStoreResolver()
        >>> resolver.extract_zarr_format("file:///data|zarr3:")
        3
        >>> resolver.extract_zarr_format("s3://bucket/data.zip|zip:|zarr2:")
        2
        >>> resolver.extract_zarr_format("file:///data|zip:")
        """
        if not is_zep8_url(url):
            return None

        segments = self.parser.parse(url)

        # Look for zarr format segments (scan from right to left for latest)
        for segment in reversed(segments):
            if segment.adapter == "zarr2":
                return 2
            elif segment.adapter == "zarr3":
                return 3

        return None

    def extract_path(self, url: str) -> str:
        """
        Extract path component from final URL segment.

        Parameters
        ----------
        url : str
            ZEP 8 URL to analyze.

        Returns
        -------
        str
            The path component from the final segment, or empty string.

        Examples
        --------
        >>> resolver = URLStoreResolver()
        >>> resolver.extract_path("file:///data|zip:inner/path|zarr3:")
        'inner/path'
        >>> resolver.extract_path("s3://bucket/data.zip|zip:|zarr3:group")
        'group'
        """
        if not is_zep8_url(url):
            return ""

        segments = self.parser.parse(url)

        if not segments:
            return ""

        return self._extract_path_from_segments(segments)

    def _extract_path_from_segments(self, segments: list[URLSegment]) -> str:
        """
        Internal helper to extract path from parsed URL segments.

        Combines all format segment paths, then falls back to adapter paths.

        Parameters
        ----------
        segments : list[URLSegment]
            Parsed URL segments.

        Returns
        -------
        str
            The combined path from format segments, or adapter path, or empty string.
        """
        # Collect all format segment paths and combine them
        format_paths = [s.path for s in segments if s.adapter in ("zarr2", "zarr3") and s.path]

        if format_paths:
            # Combine all format segment paths
            return "/".join(format_paths)

        # Fallback: look for adapter path (for non-format segments)
        for segment in reversed(segments):
            if segment.adapter and segment.path and not segment.scheme:
                # Delegate to the adapter's extract_zarr_path() method if registered
                from zarr.registry import get_store_adapter

                try:
                    adapter_cls = get_store_adapter(segment.adapter)
                    extracted_path = adapter_cls.extract_zarr_path(segment)
                    if extracted_path:
                        return extracted_path
                except KeyError:
                    # Adapter not registered - use default behavior
                    return segment.path

        return ""

    def _extract_format_from_segments(self, segments: list[URLSegment]) -> ZarrFormat | None:
        """
        Internal helper to extract zarr format from parsed URL segments.

        Uses the rightmost (last) format segment to determine format.

        Parameters
        ----------
        segments : list[URLSegment]
            Parsed URL segments.

        Returns
        -------
        ZarrFormat or None
            The zarr format version (2 or 3), or None if not specified.
        """
        # Find all format segments
        format_segments = [s for s in segments if s.adapter in ("zarr2", "zarr3")]

        if format_segments:
            # Use the last (rightmost) format segment
            last_segment = format_segments[-1]
            return 2 if last_segment.adapter == "zarr2" else 3

        return None

    def resolve_relative_url(self, base_url: str, relative_url: str) -> str:
        """
        Resolve relative URLs using .. syntax.

        Parameters
        ----------
        base_url : str
            The base ZEP 8 URL to resolve against.
        relative_url : str
            Relative URL with .. components.

        Returns
        -------
        str
            The resolved absolute URL.
        """
        return self.parser.resolve_relative_url(base_url, relative_url)


async def resolve_url(
    url: str, storage_options: dict[str, Any] | None = None, **kwargs: Any
) -> Store:
    """
    Resolve a ZEP 8 URL to a store.

    This is a convenience function that creates a URLStoreResolver
    and resolves the URL.

    Parameters
    ----------
    url : str
        ZEP 8 URL to resolve.
    storage_options : dict, optional
        Storage options to pass to store adapters.
    **kwargs : Any
        Additional keyword arguments to pass to store adapters.

    Returns
    -------
    Store
        The resolved store.

    Examples
    --------
    >>> store = await resolve_url("file:///data.zip|zip:|zarr3:")
    >>> isinstance(store, ZipStore)
    True
    """
    resolver = URLStoreResolver()
    return await resolver.resolve_url(url, storage_options=storage_options, **kwargs)
