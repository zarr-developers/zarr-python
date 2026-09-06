"""Keep benchmark control flags compatible with strict configuration warnings."""

import pytest

from zarr.core.config import ZarrConfigManager
from zarr.errors import ZarrUserWarning


def test_cache_control_flag_allows_config_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ZarrConfigManager()
    before = cfg.to_dict()
    monkeypatch.setenv("ZARR_BENCHMARK_CLEAR_CACHE", "1")
    cfg.reset()
    assert cfg.to_dict() == before


def test_unknown_benchmark_config_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZARR_BENCHMARK_CLEAR_CACH", "1")
    with pytest.raises(ZarrUserWarning, match="'benchmark_clear_cach'"):
        ZarrConfigManager()
