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
        parsed = urlparse(url)

        if parsed.scheme and ("://" in url or parsed.scheme == "file"):
            # Handle schemes like s3://, file://, https://, and also file: (without //)
            if parsed.scheme == "file":
                return URLSegment(scheme="file", path=parsed.path)
            else:
                return URLSegment(scheme=parsed.scheme, path=f"{parsed.netloc}{parsed.path}")
        elif URLParser._is_windows_path(url):
            # Windows absolute path like C:\... or C:/... - treat as filesystem path
            return URLSegment(scheme="file", path=url)
        elif ":" in url:
            # Adapter syntax like "memory:", "zip:path", etc.
            adapter, path = url.split(":", 1)
            return URLSegment(adapter=adapter, path=path)
        else:
            # Local filesystem path
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

        Examples
        --------
        >>> URLParser.resolve_relative(
        ...     "s3://bucket/data/exp1.zip|zip:|zarr3:",
        ...     "|..|control.zip|zip:|zarr3:"
        ... )
        's3://bucket/control.zip|zip:|zarr3:'
        """
        if not relative_url.startswith("|"):
            return relative_url

        parser = URLParser()
        base_segments = parser.parse(base_url)
        rel_segments = parser.parse(relative_url)

        # Find the base path to navigate from
        base_path = None
        if base_segments:
            base_segment = base_segments[0]
            if base_segment.path:
                if "/" in base_segment.path:
                    base_path = "/".join(base_segment.path.split("/")[:-1])
                else:
                    base_path = ""

        # Process .. navigation
        current_path = base_path or ""
        resolved_segments = []

        for segment in rel_segments:
            if segment.adapter == "..":
                # Navigate up one level
                if current_path and "/" in current_path:
                    current_path = "/".join(current_path.split("/")[:-1])
                elif current_path:
                    current_path = ""
            else:
                # First non-.. segment - update path and continue
                if segment.adapter == "file" and current_path:
                    new_path = f"{current_path}/{segment.path}" if segment.path else current_path
                    resolved_segments.append(URLSegment(segment.adapter, new_path))
                else:
                    resolved_segments.append(segment)
                break

        # Add remaining segments
        if len(rel_segments) > len(resolved_segments):
            resolved_segments.extend(rel_segments[len(resolved_segments) :])

        # Reconstruct URL
        if not resolved_segments:
            return base_url

        result_parts = []
        for i, segment in enumerate(resolved_segments):
            if i == 0:
                result_parts.append(segment.path or segment.adapter or "")
            else:
                if segment.path:
                    result_parts.append(f"{segment.adapter}:{segment.path}")
                else:
                    result_parts.append(f"{segment.adapter}:")

        return "|".join(result_parts)


def is_zep8_url(url: Any) -> bool:
    """
    Check if a string is a ZEP 8 URL.

    ZEP 8 URLs are identified by:
    1. Presence of pipe (|) characters (for chained URLs)
    2. Simple adapter syntax like "memory:", "zip:", etc. (single segment)

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
    False
    >>> is_zep8_url("file:///data.zarr")
    False
    """
    if not url or not isinstance(url, str):
        return False

    # Check for pipe character (chained URLs)
    if "|" in url:
        # Exclude FSSpec URIs that might contain pipes in query parameters
        # This is a simple heuristic - FSSpec URIs with pipes are rare
        if "://" in url:
            # If there's a pipe after the first ://, it's likely ZEP 8
            scheme_pos = url.find("://")
            pipe_pos = url.find("|")
            if (pipe_pos != -1 and pipe_pos > scheme_pos) or (
                pipe_pos != -1 and pipe_pos < scheme_pos
            ):
                return True
        else:
            # No scheme, so any pipe indicates ZEP 8
            return True

    # Check for simple adapter syntax (single colon at end or with simple path)
    if ":" in url and "://" not in url:
        # Could be adapter syntax like "memory:", "zip:path", etc.
        parts = url.split(":")
        if len(parts) == 2:
            adapter_name = parts[0]

            # Exclude standard URI schemes that should NOT be treated as ZEP 8 URLs
            standard_schemes = {
                "file",
                "http",
                "https",
                "ftp",
                "ftps",
                "s3",
                "gcs",
                "gs",
                "azure",
                "abfs",
                "hdfs",
                "ssh",
                "sftp",
                "webhdfs",
                "github",
                "gitlab",
            }

            # Check if adapter name looks like a ZEP 8 adapter and is not a standard scheme
            # Exclude Windows drive letters (single letter followed by backslash or forward slash)
            if (
                adapter_name
                and adapter_name.lower() not in standard_schemes
                and "/" not in adapter_name
                and "\\" not in adapter_name
                and not (len(adapter_name) == 1 and adapter_name.isalpha() and len(parts) == 2 and (parts[1].startswith("/") or parts[1].startswith("\\")))
                and (
                    adapter_name.isalnum()
                    or adapter_name.replace("_", "").replace("-", "").isalnum()
                )
            ):
                # Looks like a ZEP 8 adapter name
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
        # Handle simple scheme URLs (like file:/path, s3://bucket/path) by treating them as single-segment URLs
        if not is_zep8_url(url):
            # Check if it's a simple scheme URL that we can handle
            if "://" in url or ((":" in url) and not url.startswith("/")):
                # Parse as a single segment URL - the parser should handle this
                try:
                    segments = self.parser.parse(url)
                except Exception:
                    raise ValueError(f"Not a valid URL: {url}") from None
            else:
                raise ValueError(f"Not a valid URL: {url}")
        else:
            # Parse ZEP 8 URL normally
            segments = self.parser.parse(url)

        if not segments:
            raise ValueError(f"Empty URL segments in: {url}")

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
                if segment.scheme and not segment.adapter and next_segment.adapter:
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
                    if segment.scheme in ("s3", "gcs", "gs", "http", "https", "ftp", "ftps"):
                        preceding_url = f"{segment.scheme}://{segment.path}"
                    else:
                        preceding_url = f"{segment.scheme}:{segment.path}"
                elif segment.adapter:
                    # First segment is an adapter (e.g., "memory:")
                    preceding_url = f"{segment.adapter}:{segment.path}"
                else:
                    # This shouldn't happen for first segment but handle gracefully
                    preceding_url = segment.path
            else:
                # Build preceding URL from all original segments before this one
                preceding_segments = segments[:segment_index_in_original]
                preceding_parts = []

                for prev_segment in preceding_segments:
                    if prev_segment.scheme:
                        # Handle schemes that need :// vs :
                        if prev_segment.scheme in (
                            "s3",
                            "gcs",
                            "gs",
                            "http",
                            "https",
                            "ftp",
                            "ftps",
                        ):
                            preceding_parts.append(f"{prev_segment.scheme}://{prev_segment.path}")
                        else:
                            preceding_parts.append(f"{prev_segment.scheme}:{prev_segment.path}")
                    elif prev_segment.adapter:
                        # Adapter segment - reconstruct format
                        preceding_parts.append(f"{prev_segment.adapter}:{prev_segment.path}")

                preceding_url = "|".join(preceding_parts)

            # Create the store using the adapter with preceding URL
            store_kwargs = kwargs.copy()
            if storage_options:
                store_kwargs.update(storage_options)

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

        try:
            segments = self.parser.parse(url)
        except Exception:
            return None

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

        try:
            segments = self.parser.parse(url)
        except Exception:
            return ""

        if not segments:
            return ""

        # Look for path in segments, prioritizing zarr format segments for zarr paths
        zarr_path = ""
        adapter_path = ""

        for segment in reversed(segments):
            # Check for zarr format segments first (these contain the zarr path)
            if segment.adapter in ("zarr2", "zarr3") and segment.path and not zarr_path:
                zarr_path = segment.path
            elif (
                segment.adapter
                and segment.adapter not in ("zarr2", "zarr3")
                and segment.path
                and not adapter_path
                and not segment.scheme
            ):
                # Only extract paths from adapter segments, not scheme segments
                # Scheme segments (like file:, s3:, https:) contain paths to the resource, not zarr paths within it
                # Special handling for icechunk: paths with metadata references
                # Both old format "branch:main", "tag:v1.0", "snapshot:abc123"
                # and new format "@branch.main", "@tag.v1.0", "@abc123def456"
                if segment.adapter in ("icechunk", "ic"):
                    # Check old format: branch:main, tag:v1.0, snapshot:abc123
                    if ":" in segment.path and segment.path.split(":")[0] in (
                        "branch",
                        "tag",
                        "snapshot",
                    ):
                        continue  # Skip icechunk metadata paths

                    # Check new format: @branch.main, @tag.v1.0, @abc123def456
                    # Parse the path to extract the zarr path component
                    if segment.path.startswith("@"):
                        try:
                            # Use icechunk's parser to extract the zarr path
                            from zarr.registry import get_store_adapter

                            # Try both possible registry names for icechunk
                            adapter_cls = None
                            for name in ("icechunk", "icechunk.zarr_adapter.IcechunkStoreAdapter"):
                                try:
                                    adapter_cls = get_store_adapter(name)
                                    break
                                except KeyError:
                                    continue

                            if adapter_cls and hasattr(
                                adapter_cls, "_extract_zarr_path_from_segment"
                            ):
                                zarr_path_component = adapter_cls._extract_zarr_path_from_segment(
                                    segment.path
                                )
                                if zarr_path_component:
                                    adapter_path = zarr_path_component
                                continue
                            # Fallback: if starts with @ and has /, extract part after first /
                            if "/" in segment.path:
                                _, path_part = segment.path.split("/", 1)
                                adapter_path = path_part
                            continue
                        except Exception:
                            # If parsing fails, treat as regular path
                            pass
                adapter_path = segment.path

        # Prefer zarr format path over adapter path
        return zarr_path or adapter_path

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
