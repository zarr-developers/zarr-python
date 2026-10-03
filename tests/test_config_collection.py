"""Environment ownership and the boundary between collection and construction."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

from zarr.core.config import collect_config, collect_environment, create_config
from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("environment", "config", "controls"),
    [
        ({}, {}, {}),
        ({"PATH": "/bin"}, {}, {}),
        ({"ZARR_ARRAY__ORDER": "F"}, {"ZARR_ARRAY__ORDER": "F"}, {}),
        ({"ZARR_CODEC-PIPELINE__MAX-WORKERS": "4"}, {"ZARR_CODEC-PIPELINE__MAX-WORKERS": "4"}, {}),
        (
            {"ZARR_CODECS__CUSTOM_CODEC": "pkg.Custom"},
            {"ZARR_CODECS__CUSTOM_CODEC": "pkg.Custom"},
            {},
        ),
        (
            {
                "ZARR_CONFIG": "/tmp/zarr.yaml",
                "ZARR_ROOT_CONFIG": "/tmp/config",
                "ZARR_BENCHMARK_CLEAR_CACHE": "1",
                "ZARR_ASYNC__CONCURRENCY": "8",
            },
            {"ZARR_ASYNC__CONCURRENCY": "8"},
            {
                "ZARR_CONFIG": "/tmp/zarr.yaml",
                "ZARR_ROOT_CONFIG": "/tmp/config",
                "ZARR_BENCHMARK_CLEAR_CACHE": "1",
            },
        ),
    ],
)
def test_collect_environment(
    environment: dict[str, str], config: dict[str, str], controls: dict[str, str]
) -> None:
    before = environment.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        assert collect_environment(environment) == {"config": config, "controls": controls}
    assert environment == before


@pytest.mark.parametrize(
    "name",
    [
        "ZARR_BENCHMARK_CLEAR_CACH",
        "ZARR_ARRAY__ORDRE",
        "ZARR_ARRAY__ORDER__UPPER",
        "ZARR_FUTURE__KEY",
    ],
)
def test_collect_environment_unknown_name_warns(name: str) -> None:
    with pytest.warns(ZarrUserWarning, match=name) as caught:
        result = collect_environment({name: "1", "ZARR_ARRAY__ORDER": "F"})
    assert len(caught) == 1
    assert result == {"config": {"ZARR_ARRAY__ORDER": "F"}, "controls": {}}


def test_collect_config_separates_controls_and_defers_value_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "zarr.yaml"
    config_file.write_text("array:\n  order: Q\n")
    monkeypatch.setenv("ZARR_CONFIG", str(config_file))
    monkeypatch.setenv("ZARR_BENCHMARK_CLEAR_CACHE", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        collected = collect_config()
    assert collected["array.order"] == "Q"
    assert "config" not in collected
    assert "benchmark_clear_cache" not in collected
    with pytest.raises(ValueError, match="array.order"):
        create_config(collected)


def test_collect_config_preserves_invalid_leaf_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "zarr.yaml"
    config_file.write_text("async:\n  concurrency:\n    invalid: 1\n")
    monkeypatch.setenv("ZARR_CONFIG", str(config_file))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        collected = collect_config()
    assert collected["async.concurrency"] == {"invalid": 1}
    with pytest.raises(ValueError, match="async.concurrency"):
        create_config(collected)


def test_collect_config_unknown_empty_mapping_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "zarr.yaml"
    config_file.write_text("future: {}\n")
    monkeypatch.setenv("ZARR_CONFIG", str(config_file))
    with pytest.warns(ZarrUserWarning, match="future"):
        collected = collect_config()
    assert "future" not in collected


@pytest.mark.parametrize("key", ["config", "root_config", "benchmark_clear_cache", "future"])
def test_collect_config_warns_for_control_names_in_yaml(
    key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "zarr.yaml"
    config_file.write_text(f"{key}: 1\narray:\n  order: F\n")
    monkeypatch.setenv("ZARR_CONFIG", str(config_file))
    with pytest.warns(ZarrUserWarning, match=key):
        collected = collect_config()
    assert key not in collected
    assert create_config(collected).array.order == "F"


def test_create_config_is_independent_of_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZARR_ARRAY__ORDER", "F")
    assert create_config({}).array.order == "C"
    cfg = create_config({"array.order": "F", "codecs.custom": "pkg.Custom"})
    assert cfg.array.order == "F"
    assert cfg.codecs["custom"] == "pkg.Custom"


@pytest.mark.parametrize("key", ["benchmark_clear_cache", "array.ordre", "future.key"])
def test_create_config_unknown_key_raises(key: str) -> None:
    with pytest.raises(KeyError):
        create_config({key: 1})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("array.order", "Q"),
        ("async.concurrency", "many"),
        ("array.write_empty_chunks", "false"),
        ("codecs.custom", 1),
    ],
)
def test_create_config_invalid_value_raises(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key):
        create_config({key: value})
