from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import zarr.registry

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def clean_url_adapter_registry() -> Generator[None, None, None]:
    """Snapshot and restore the URL adapter registry around a test."""
    registry = zarr.registry._url_adapter_registry
    saved = dict(registry)
    saved_lazy = list(registry.lazy_load_list)
    yield
    registry.clear()
    registry.update(saved)
    registry.lazy_load_list[:] = saved_lazy
