"""Compatibility at the public config API and donfig reader boundaries."""

from __future__ import annotations

import copy
import dataclasses
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints

import donfig
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from zarr.core.config import (
    _SERIALIZED_NAMES,
    ZarrConfigManager,
    collect_environment,
    make_default_config,
)


def _config_values(annotation: Any) -> st.SearchStrategy[object]:
    """Use field types, with bounded numbers and strings safe for env/YAML transport."""
    if annotation is int:
        return st.integers(min_value=1, max_value=256)
    if annotation is float:
        return st.floats(min_value=0, max_value=256, allow_nan=False, allow_infinity=False)
    if annotation is str:
        return st.from_regex(r"custom\.[A-Za-z][A-Za-z0-9]{0,10}", fullmatch=True)
    if get_origin(annotation) is UnionType:
        return st.one_of(*(_config_values(member) for member in get_args(annotation)))
    # Includes bool, NoneType, and Literal fields such as order and Zarr format.
    return st.from_type(annotation)


def _config_leaves(
    node: object, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], st.SearchStrategy[object]]]:
    """Discover schema leaves and registered default codec names, retaining literal dots."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield (*prefix, key), _config_values(type(value))
    else:
        assert dataclasses.is_dataclass(node)
        assert not isinstance(node, type)
        hints = get_type_hints(type(node))
        for field in dataclasses.fields(node):
            path = (*prefix, _SERIALIZED_NAMES.get(field.name, field.name))
            value = getattr(node, field.name)
            if dataclasses.is_dataclass(value) or isinstance(value, Mapping):
                yield from _config_leaves(value, path)
            else:
                yield path, _config_values(hints[field.name])


def _config_cases(*, dotted_api: bool) -> st.SearchStrategy[tuple[tuple[str, ...], object, object]]:
    # Donfig's dotted API cannot address a literal dot inside a codec name.
    # YAML tests retain these leaves because YAML preserves mapping boundaries.
    return st.one_of(
        *(
            st.tuples(st.just(path), values, values)
            for path, values in _config_leaves(make_default_config())
            if not dotted_api or all("." not in part for part in path)
        )
    )


def _alternate_key(key: str) -> str:
    return ".".join(
        part.replace("_", "-") if "_" in part else part.replace("-", "_") for part in key.split(".")
    )


@given(case=_config_cases(dotted_api=True))
def test_collect_environment_recognizes_schema_keys(
    case: tuple[tuple[str, ...], object, object],
) -> None:
    path, value, _ = case
    name = "ZARR_" + "__".join(path).replace("-", "_").upper()
    config_environment = {name: repr(value)}
    controls = {"ZARR_BENCHMARK_CLEAR_CACHE": "1"}
    assert collect_environment(config_environment | controls) == {
        "config": config_environment,
        "controls": controls,
    }


@given(case=_config_cases(dotted_api=True), alternate=st.booleans())
def test_config_key_spellings_match_donfig(
    case: tuple[tuple[str, ...], object, object], alternate: bool
) -> None:
    """Aliases select the same leaf and scoped updates restore the entire config."""
    path, selected, _ = case
    key = ".".join(path)
    cfg = ZarrConfigManager()
    reference = donfig.Config("zarr", defaults=[cfg.to_dict()], paths=[], env={})
    alias = _alternate_key(key)
    written_key = alias if alternate else key
    before = cfg.to_dict()
    with reference.set({written_key: selected}), cfg.set({written_key: selected}):
        assert cfg.get(key) == reference.get(key)
        assert cfg.get(alias) == reference.get(alias)
        assert alias in cfg
        assert cfg.to_dict() == reference.config
    assert cfg.to_dict() == reference.config == before


@given(case=_config_cases(dotted_api=True))
def test_config_nested_kwargs_match_donfig(case: tuple[tuple[str, ...], object, object]) -> None:
    """Nested kwargs override mapping entries, including when their spellings differ."""
    path, first, second = case
    key = ".".join(path)
    cfg = ZarrConfigManager()
    reference = donfig.Config("zarr", defaults=[cfg.to_dict()], paths=[], env={})
    keyword = key.replace("-", "_").replace(".", "__")
    before = cfg.to_dict()
    with (
        reference.set({key: first}, **{keyword: second}),
        cfg.set({key: first}, **{keyword: second}),
    ):
        assert cfg.get(key) == second
        assert cfg.to_dict() == reference.config
    assert cfg.to_dict() == reference.config == before


@given(case=_config_cases(dotted_api=False), env_override=st.booleans())
def test_config_yaml_aliases_match_donfig(
    case: tuple[tuple[str, ...], object, object], env_override: bool
) -> None:
    """Hyphen/underscore YAML spellings retain defaults and environment precedence."""
    path, selected, override = case
    key = ".".join(path)
    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        for name in list(os.environ):
            if name.startswith("ZARR_"):
                patch.delenv(name)
        defaults = ZarrConfigManager().defaults
        data: object = selected
        for part in reversed(path):
            data = {_alternate_key(part): data}
        config_file = Path(directory) / "zarr.yaml"
        config_file.write_text(yaml.safe_dump(data))
        patch.setenv("ZARR_CONFIG", str(config_file))
        # Donfig's environment reader also interprets literal dots as nesting.
        if env_override and all("." not in part for part in path):
            env_key = "ZARR_" + "__".join(path).replace("-", "_").upper()
            selected = override
            patch.setenv(env_key, repr(selected))
        reference = donfig.Config("zarr", defaults=[defaults])
        reference.config.pop("config", None)
        cfg = ZarrConfigManager()
        assert cfg.get(key) == selected
        assert cfg.to_dict() == reference.config


@given(case=_config_cases(dotted_api=True))
def test_config_manager_deepcopy_is_independent(
    case: tuple[tuple[str, ...], object, object],
) -> None:
    """Copy the public manager, including its lock, rather than just its snapshot."""
    path, value, replacement = case
    key = ".".join(path)
    cfg = ZarrConfigManager()
    cfg.set({key: value, "codecs.custom": "custom.Original"})
    restored = copy.deepcopy(cfg)
    assert restored.to_dict() == cfg.to_dict()
    with restored.set({key: replacement}):
        assert restored.get(key) == replacement
        assert cfg.get(key) == value
    assert restored.get(key) == value
    # The currently mutable subtree must also be deeply copied.
    restored.get("codecs")["custom"] = "custom.Copy"
    assert cfg.get("codecs.custom") == "custom.Original"
