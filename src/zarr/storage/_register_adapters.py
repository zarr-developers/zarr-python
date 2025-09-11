"""
Auto-registration of built-in store adapters.

This module ensures that built-in store adapters are registered
when zarr-python is imported, providing ZEP 8 URL syntax support
out of the box.
"""

from zarr.registry import register_store_adapter


def register_builtin_adapters() -> None:
    """Register all built-in store adapters."""
    # Import all the adapter classes
    # Register all adapters
    from typing import TYPE_CHECKING

    from zarr.storage._builtin_adapters import (
        FileSystemAdapter,
        GSAdapter,
        HttpsAdapter,
        LoggingAdapter,
        MemoryAdapter,
        S3Adapter,
        S3HttpAdapter,
        S3HttpsAdapter,
        ZipAdapter,
    )

    if TYPE_CHECKING:
        from zarr.abc.store_adapter import StoreAdapter

    adapters: list[type[StoreAdapter]] = [
        FileSystemAdapter,
        MemoryAdapter,
        HttpsAdapter,
        S3Adapter,
        S3HttpAdapter,
        S3HttpsAdapter,
        GSAdapter,
        LoggingAdapter,
        ZipAdapter,
    ]

    for adapter in adapters:
        register_store_adapter(adapter)


# Auto-register when this module is imported
register_builtin_adapters()
