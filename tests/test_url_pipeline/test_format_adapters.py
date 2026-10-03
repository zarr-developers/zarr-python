"""Tests for the builtin `zarr:`, `zarr2:` and `zarr3:` URL pipeline adapters."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import uuid
from importlib.metadata import EntryPoint
from typing import TYPE_CHECKING

import numpy as np
import pytest

import zarr
import zarr.registry
from zarr.abc.url_pipeline import (
    AdapterResolution,
    PipelineContext,
    PipelineSegment,
    URLPipelineAdapter,
)
from zarr.errors import URLPipelineError, ZarrUserWarning
from zarr.registry import get_url_adapter, list_url_adapter_schemes, register_url_adapter
from zarr.storage import LocalStore, ManagedMemoryStore, WrapperStore
from zarr.storage._common import make_store_path
from zarr.storage._url_adapters._format import Zarr2Adapter, Zarr3Adapter, ZarrAdapter
from zarr.storage._url_pipeline import is_url_pipeline, resolve_pipeline
from zarr.storage._utils import _join_paths

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from zarr.core.common import ZarrFormat

pytestmark = pytest.mark.usefixtures("clean_url_adapter_registry")

FORMAT_SEGMENTS: list[tuple[str, ZarrFormat | None]] = [
    ("zarr2:", 2),
    ("zarr3:", 3),
    ("zarr:", None),
]


@pytest.fixture(params=["local", "memory"])
def root(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[str, None, None]:
    """A pipeline root: a local directory, or a fresh named in-memory store."""
    if request.param == "local":
        yield str(tmp_path / "root")
        return
    # a named managed memory store lives only while something references it,
    # so hold one for the duration of the test
    name = uuid.uuid4().hex
    keepalive = ManagedMemoryStore(name=name)
    yield f"memory:{name}"
    del keepalive


class _WrapAdapter(URLPipelineAdapter):
    """A wrapper adapter that follows the documented wrapper contract."""

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> AdapterResolution:
        preceding = await context.resolve_preceding()
        return dataclasses.replace(
            preceding,
            store=WrapperStore(preceding.store),
            path=_join_paths([preceding.path, segment.body]),
        )


def _expected(fmt: ZarrFormat | None) -> ZarrFormat:
    """The format a node created through a segment ends up with."""
    return 3 if fmt is None else fmt


class TestRegistration:
    def test_builtin_schemes_are_listed(self) -> None:
        assert {"zarr", "zarr2", "zarr3"} <= list_url_adapter_schemes()

    @pytest.mark.parametrize(
        ("scheme", "cls"), [("zarr", ZarrAdapter), ("zarr2", Zarr2Adapter), ("zarr3", Zarr3Adapter)]
    )
    def test_builtin_lookup(self, scheme: str, cls: type[URLPipelineAdapter]) -> None:
        assert get_url_adapter(scheme) is cls
        assert get_url_adapter(scheme.upper()) is cls

    def test_import_zarr_does_not_import_adapters(self) -> None:
        # builtins are lazily loaded entry points, like third-party adapters
        code = (
            "import sys, zarr, zarr.storage\n"
            "assert 'zarr.storage._url_adapters._format' not in sys.modules\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_third_party_entry_point_is_shadowed_with_warning(self) -> None:
        registry = zarr.registry._url_adapter_registry
        registry.pop("zarr3", None)
        registry.lazy_load_list[:] = [
            zarr.registry._builtin_url_adapter_entry_point("zarr3"),
            *(e for e in registry.lazy_load_list if e.name != "zarr3"),
            EntryPoint(name="zarr3", value="other_pkg:Adapter", group="zarr.url_adapters"),
        ]
        with pytest.warns(ZarrUserWarning, match="builtin URL pipeline adapter.*takes precedence"):
            assert get_url_adapter("zarr3") is Zarr3Adapter
        assert not any(e.name == "zarr3" for e in registry.lazy_load_list)

    def test_explicit_registration_overrides_builtin_silently(self) -> None:
        # an explicit register_url_adapter call is deliberate: the pending
        # builtin entry point is dropped without a warning
        registry = zarr.registry._url_adapter_registry
        registry.pop("zarr3", None)
        registry.lazy_load_list[:] = [
            zarr.registry._builtin_url_adapter_entry_point("zarr3"),
            *(e for e in registry.lazy_load_list if e.name != "zarr3"),
        ]
        register_url_adapter("zarr3", ZarrAdapter)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert get_url_adapter("zarr3") is ZarrAdapter

    def test_collect_entrypoints_does_not_duplicate_builtins(self) -> None:
        zarr.registry._collect_entrypoints()
        zarr.registry._collect_entrypoints()
        pending = [
            e
            for e in zarr.registry._url_adapter_registry.lazy_load_list
            if zarr.registry._is_builtin_url_adapter_entry_point(e)
        ]
        assert len(pending) == len({e.name for e in pending})

    @pytest.mark.parametrize("url", ["zarr:foo", "zarr2:foo", "zarr3://foo", "ZARR3:foo"])
    def test_builtin_schemes_are_not_root_schemes(self, url: str) -> None:
        # a string without '|' keeps its pre-pipeline meaning
        assert not is_url_pipeline(url)


class TestResolution:
    @pytest.mark.parametrize(("segment", "fmt"), FORMAT_SEGMENTS)
    async def test_sets_format(self, tmp_path: Path, segment: str, fmt: ZarrFormat | None) -> None:
        result = await resolve_pipeline(f"{tmp_path}|{segment}")
        assert isinstance(result.store, LocalStore)
        assert result.path == ""
        assert result.zarr_format == fmt

    @pytest.mark.parametrize("scheme", ["zarr", "zarr2", "zarr3"])
    async def test_colon_is_optional(self, tmp_path: Path, scheme: str) -> None:
        with_colon = await resolve_pipeline(f"{tmp_path}|{scheme}:")
        without = await resolve_pipeline(f"{tmp_path}|{scheme}")
        assert with_colon.zarr_format == without.zarr_format

    async def test_body_is_a_path_within_the_preceding_resource(self, tmp_path: Path) -> None:
        result = await resolve_pipeline(f"{tmp_path}|zarr3:a/b")
        assert result.path == "a/b"
        assert result.zarr_format == 3

    @pytest.mark.parametrize("body", ["/a/b", "a/b/", "/a//b/"])
    async def test_body_is_normalized(self, tmp_path: Path, body: str) -> None:
        result = await resolve_pipeline(f"{tmp_path}|zarr3:{body}")
        assert result.path == "a/b"

    async def test_body_joins_onto_preceding_residual_path(self) -> None:
        # memory:name/sub roots carry their path inside the store; a wrapper
        # carries a residual path; both are kept
        register_url_adapter("wrap", _WrapAdapter)
        result = await resolve_pipeline("memory:fmt-join/sub|wrap:w|zarr2:a/b")
        assert result.path == "w/a/b"
        assert result.zarr_format == 2

    async def test_intermediate_segment(self) -> None:
        # a format segment may precede another adapter, which carries the
        # format and the residual path forward
        register_url_adapter("wrap", _WrapAdapter)
        result = await resolve_pipeline("memory:fmt-mid|zarr2:a|wrap:b")
        assert isinstance(result.store, WrapperStore)
        assert result.path == "a/b"
        assert result.zarr_format == 2

    async def test_keeps_other_fields(self) -> None:
        # the adapter uses dataclasses.replace, so the preceding store is kept
        result = await resolve_pipeline("memory:fmt-keep|zarr3:")
        assert isinstance(result.store, ManagedMemoryStore)

    async def test_auto_keeps_an_earlier_format(self, tmp_path: Path) -> None:
        result = await resolve_pipeline(f"{tmp_path}|zarr2:a|zarr:b")
        assert result.zarr_format == 2
        assert result.path == "a/b"

    async def test_conflicting_format_segments_raise(self, tmp_path: Path) -> None:
        with pytest.raises(URLPipelineError, match="earlier segment selected Zarr format 2"):
            await resolve_pipeline(f"{tmp_path}|zarr2:|zarr3:")

    async def test_query_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(URLPipelineError, match="do not accept a query"):
            await resolve_pipeline(f"{tmp_path}|zarr3:a?x=1")

    @pytest.mark.parametrize("body", ["..", "a/../b", "./a"])
    async def test_invalid_path_is_rejected(self, tmp_path: Path, body: str) -> None:
        with pytest.raises(URLPipelineError, match="invalid path"):
            await resolve_pipeline(f"{tmp_path}|zarr3:{body}")

    async def test_mode_is_forwarded(self, tmp_path: Path) -> None:
        zarr.create_group(tmp_path)
        store_path = await make_store_path(f"{tmp_path}|zarr3:", mode="r")
        assert store_path.read_only
        assert store_path.zarr_format == 3


class TestEndToEnd:
    @pytest.mark.parametrize(("segment", "fmt"), FORMAT_SEGMENTS)
    def test_open_group(self, root: str, segment: str, fmt: ZarrFormat | None) -> None:
        group = zarr.open_group(f"{root}|{segment}", mode="w")
        assert group.metadata.zarr_format == _expected(fmt)
        group.create_array("x", shape=(3,), dtype="i4")[:] = [1, 2, 3]
        reopened = zarr.open_group(f"{root}|{segment}", mode="r")
        assert reopened.metadata.zarr_format == _expected(fmt)
        arr = reopened["x"]
        assert isinstance(arr, zarr.Array)
        np.testing.assert_array_equal(arr[:], [1, 2, 3])

    @pytest.mark.parametrize(("segment", "fmt"), FORMAT_SEGMENTS)
    def test_open(self, root: str, segment: str, fmt: ZarrFormat | None) -> None:
        group = zarr.open(f"{root}|{segment}", mode="w")
        assert isinstance(group, zarr.Group)
        assert group.metadata.zarr_format == _expected(fmt)
        arr = zarr.open(f"{root}|{segment}x", mode="w", shape=(2,), dtype="i4")
        assert isinstance(arr, zarr.Array)
        assert arr.metadata.zarr_format == _expected(fmt)
        reopened = zarr.open(f"{root}|{segment}x", mode="r")
        assert isinstance(reopened, zarr.Array)
        assert reopened.metadata.zarr_format == _expected(fmt)

    @pytest.mark.parametrize(("segment", "fmt"), FORMAT_SEGMENTS)
    def test_create_and_open_array(self, root: str, segment: str, fmt: ZarrFormat | None) -> None:
        arr = zarr.create_array(f"{root}|{segment}", name="a/b", shape=(4,), dtype="i4")
        assert arr.metadata.zarr_format == _expected(fmt)
        arr[:] = np.arange(4)
        opened = zarr.open_array(f"{root}|{segment}a/b", mode="r")
        assert opened.metadata.zarr_format == _expected(fmt)
        np.testing.assert_array_equal(opened[:], np.arange(4))
        # the user-supplied path joins onto the segment path
        via_path = zarr.open_array(f"{root}|{segment}a", path="b", mode="r")
        np.testing.assert_array_equal(via_path[:], np.arange(4))

    @pytest.mark.parametrize(("segment", "fmt"), FORMAT_SEGMENTS)
    def test_group_and_array_open_classmethods(
        self, root: str, segment: str, fmt: ZarrFormat | None
    ) -> None:
        zarr.create_array(f"{root}|{segment}", name="x", shape=(2,), dtype="i4")
        group = zarr.Group.open(f"{root}|{segment}")
        assert group.metadata.zarr_format == _expected(fmt)
        arr = zarr.Array.open(f"{root}|{segment}x")
        assert arr.metadata.zarr_format == _expected(fmt)

    @pytest.mark.parametrize("stored", [2, 3])
    def test_auto_detects_existing_format(self, root: str, stored: ZarrFormat) -> None:
        zarr.create_array(f"{root}|zarr{stored}:", name="x", shape=(2,), dtype="i4")
        assert zarr.open_group(f"{root}|zarr:", mode="r").metadata.zarr_format == stored
        assert zarr.open_array(f"{root}|zarr:x", mode="r").metadata.zarr_format == stored
        assert zarr.Group.open(f"{root}|zarr:").metadata.zarr_format == stored
        assert zarr.Array.open(f"{root}|zarr:x").metadata.zarr_format == stored

    def test_stacked_segment_paths(self, root: str) -> None:
        zarr.create_array(f"{root}|zarr3:a/b", name="c", shape=(2,), dtype="i4")
        assert zarr.open_array(f"{root}|zarr3:a/b/c", mode="r").shape == (2,)
        group = zarr.open_group(f"{root}|zarr3:a", mode="r")
        assert "b" in group

    def test_wrong_format_is_not_found(self, root: str) -> None:
        zarr.create_group(f"{root}|zarr2:")
        with pytest.raises(FileNotFoundError):
            zarr.open_group(f"{root}|zarr3:", mode="r")

    @pytest.mark.parametrize(
        ("segment", "explicit"), [("zarr2:", 3), ("zarr3:", 2)], ids=["zarr2-vs-3", "zarr3-vs-2"]
    )
    def test_explicit_format_conflict(self, root: str, segment: str, explicit: ZarrFormat) -> None:
        url = f"{root}|{segment}"
        with pytest.raises(ValueError, match="conflicts with"):
            zarr.open_group(url, mode="w", zarr_format=explicit)
        with pytest.raises(ValueError, match="conflicts with"):
            zarr.create_array(url, name="x", shape=(1,), dtype="i4", zarr_format=explicit)
        with pytest.raises(ValueError, match="conflicts with"):
            zarr.Group.open(url, zarr_format=explicit)

    def test_explicit_format_with_auto_segment(self, root: str) -> None:
        # zarr: selects no format, so an explicit format is not a conflict
        group = zarr.open_group(f"{root}|zarr:", mode="w", zarr_format=2)
        assert group.metadata.zarr_format == 2


ZARR_SPEC_EXAMPLES: list[tuple[str, ZarrFormat | None]] = [
    ("file:///path/to/node.zarr/|zarr3:", 3),
    ("file:///path/to/node.zarr/|zarr3", 3),
    ("file:///path/to/node.zarr/|zarr2:", 2),
    ("file:///path/to/node.zarr/|zarr2", 2),
    ("file:///path/to/node.zarr/|zarr:", None),
    ("file:///path/to/node.zarr/|zarr", None),
]


@pytest.mark.parametrize(("example", "fmt"), ZARR_SPEC_EXAMPLES)
def test_zarr_scheme_spec_examples(tmp_path: Path, example: str, fmt: ZarrFormat | None) -> None:
    """Every example in the spec's `schemes/zarr.md` resolves and opens."""
    posix = tmp_path.as_posix()
    url_path = posix if posix.startswith("/") else f"/{posix}"
    url = example.replace("/path/to", url_path, 1)
    stored: ZarrFormat = 3 if fmt is None else fmt
    zarr.create_group(tmp_path / "node.zarr", zarr_format=stored)
    group = zarr.open_group(url, mode="r")
    assert group.metadata.zarr_format == stored
    assert zarr.Group.open(url).metadata.zarr_format == stored
