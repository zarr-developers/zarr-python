from __future__ import annotations

import contextlib
import dataclasses
import sys
import threading
import warnings
from importlib.metadata import EntryPoint, version
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
from packaging.version import parse as parse_version

import zarr
import zarr.registry
from zarr.abc.store import Store
from zarr.abc.url_pipeline import URLPipelineAdapter
from zarr.errors import URLPipelineError, ZarrUserWarning
from zarr.registry import (
    _get_url_adapter,
    _list_url_adapter_schemes,
    register_url_adapter,
)
from zarr.storage import (
    LocalStore,
    ManagedMemoryStore,
    MemoryStore,
    PipelineContext,
    PipelineSegment,
    StorePath,
    URLPipeline,
    WrapperStore,
)
from zarr.storage._common import _has_fsspec, make_store, make_store_path
from zarr.storage._utils import _join_paths

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from zarr.core.common import AccessModeLiteral

# fsspec < 2024.12.0 has no AsyncFileSystemWrapper, so a plain memory:// URL
# (a sync filesystem) cannot be opened at all there
_has_async_fsspec_wrapper = _has_fsspec and parse_version(version("fsspec")) >= parse_version(
    "2024.12.0"
)

pytestmark = pytest.mark.usefixtures("clean_url_adapter_registry")


class TracingStore(WrapperStore[Store]):
    """Wrapper that records the context and segment it was created from."""

    context: PipelineContext
    segment: PipelineSegment


class WrapperAdapter(URLPipelineAdapter):
    """
    A wrapper-style adapter: resolves the preceding pipeline into a store.

    Follows the wrapper contract: the preceding residual path is carried
    forward, joined with this segment's own path.
    """

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> StorePath:
        preceding = await context.resolve_preceding()
        store = TracingStore(preceding.store)
        store.context = context
        store.segment = segment
        return StorePath(store, _join_paths([preceding.path, segment.body]))


class NativeAdapter(URLPipelineAdapter):
    """A native-style adapter: consumes the preceding URL as a string."""

    seen_urls: ClassVar[list[str]] = []

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> StorePath:
        cls.seen_urls.append(context.preceding_url)
        store = await MemoryStore.open(read_only=context.read_only)
        return StorePath(store, path=segment.body)


class RootAdapter(URLPipelineAdapter):
    """A read-only root-scheme adapter (no preceding segments)."""

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> StorePath:
        assert context.preceding == ()
        if context.mode in ("w", "w-", "r+"):
            raise ValueError("read-only scheme")
        store = await MemoryStore.open(read_only=True)
        return StorePath(store, path=segment.body.lstrip("/"))


class DisobedientAdapter(URLPipelineAdapter):
    """An adapter that ignores `context.read_only` (a contract violation)."""

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> StorePath:
        return StorePath(await MemoryStore.open(read_only=False))


class TestRegistry:
    def test_register_and_get(self) -> None:
        register_url_adapter("demo", WrapperAdapter)
        assert _get_url_adapter("demo") is WrapperAdapter
        assert _get_url_adapter("DEMO") is WrapperAdapter
        assert "demo" in _list_url_adapter_schemes()

    def test_unknown_scheme(self) -> None:
        with pytest.raises(URLPipelineError, match="no URL pipeline adapter is registered"):
            _get_url_adapter("nonexistent-scheme")

    def test_reregistering_scheme_warns(self) -> None:
        register_url_adapter("demo", WrapperAdapter)
        with pytest.warns(ZarrUserWarning, match="is being replaced"):
            register_url_adapter("demo", NativeAdapter)
        assert _get_url_adapter("demo") is NativeAdapter
        # re-registering the same class is not a collision
        register_url_adapter("demo", NativeAdapter)

    @pytest.mark.usefixtures("set_path")
    def test_entrypoint_discovery(self) -> None:
        assert "example-pkg.entrypoint-scheme" in _list_url_adapter_schemes()
        cls = _get_url_adapter("example-pkg.entrypoint-scheme")
        assert cls.__name__ == "TestEntrypointURLAdapter"

    @pytest.mark.usefixtures("set_path")
    async def test_entrypoint_end_to_end(self) -> None:
        result = await URLPipeline.from_url(
            "memory://src|example-pkg.entrypoint-scheme:sub/path"
        )._resolve_store_path()
        assert result.path == "sub/path"

    @pytest.mark.usefixtures("set_path")
    def test_entrypoint_scheme_lookup_is_case_insensitive(self) -> None:
        # entry-point names are matched case-insensitively, like schemes
        cls = _get_url_adapter("Example-PKG.Entrypoint-Scheme")
        assert cls.__name__ == "TestEntrypointURLAdapter"

    @pytest.mark.usefixtures("set_path")
    def test_loading_one_scheme_leaves_others_pending(self) -> None:
        # resolving one scheme must not import other providers' entry points
        registry = zarr.registry._url_adapter_registry
        assert any(e.name == "example-pkg.entrypoint-scheme" for e in registry.lazy_load_list)
        with pytest.raises(URLPipelineError, match="no URL pipeline adapter"):
            _get_url_adapter("some-other-scheme")
        assert any(e.name == "example-pkg.entrypoint-scheme" for e in registry.lazy_load_list)
        assert _get_url_adapter("example-pkg.entrypoint-scheme").__name__ == (
            "TestEntrypointURLAdapter"
        )


def _fake_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, scheme: str, module: str, source: str
) -> EntryPoint:
    """
    Write `source` to an importable module and return a `zarr.url_adapters`
    entry point for `scheme` pointing at its `Adapter` attribute.
    """
    (tmp_path / f"{module}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    return EntryPoint(name=scheme, value=f"{module}:Adapter", group="zarr.url_adapters")


_ADAPTER_SOURCE = """
from zarr.abc.url_pipeline import URLPipelineAdapter
from zarr.storage import MemoryStore, StorePath
{prelude}

class Adapter(URLPipelineAdapter):
    @classmethod
    async def open_pipeline_segment(cls, segment, context):
        return StorePath(await MemoryStore.open(read_only=context.read_only))
"""


class TestEntryPointLoading:
    """Lazy loading of `zarr.url_adapters` entry points."""

    def test_adapter_import_may_resolve_other_adapters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # an adapter module that itself resolves a URL adapter at import
        # time must not deadlock on the registry lock
        pending = zarr.registry._url_adapter_registry.lazy_load_list
        pending.append(
            _fake_entry_point(
                tmp_path,
                monkeypatch,
                scheme="reentrant",
                module="reentrant_adapter",
                source=_ADAPTER_SOURCE.format(
                    prelude=(
                        "import zarr.registry\n"
                        "try:\n"
                        "    zarr.registry._get_url_adapter('no-such-scheme-at-import')\n"
                        "except Exception:\n"
                        "    pass\n"
                    )
                ),
            )
        )
        result: list[object] = []
        thread = threading.Thread(target=lambda: result.append(_get_url_adapter("reentrant")))
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive(), "_get_url_adapter deadlocked on a re-entrant lookup"
        assert getattr(result[0], "__name__", None) == "Adapter"

    def test_concurrent_loads_keep_other_schemes_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # two threads resolving different schemes at once, one with a slow
        # import, must not drop a third, unrelated pending entry point
        pending = zarr.registry._url_adapter_registry.lazy_load_list
        pending.clear()  # drop installed entry points, e.g. the builtin zip:
        for scheme, prelude in [
            ("slow", "import time\ntime.sleep(0.3)\n"),
            ("quick", ""),
            ("untouched", ""),
        ]:
            pending.append(
                _fake_entry_point(
                    tmp_path,
                    monkeypatch,
                    scheme=scheme,
                    module=f"{scheme}_adapter",
                    source=_ADAPTER_SOURCE.format(prelude=prelude),
                )
            )
        threads = [
            threading.Thread(target=_get_url_adapter, args=(scheme,))
            for scheme in ("slow", "quick")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert not any(thread.is_alive() for thread in threads)
        assert "untouched" in _list_url_adapter_schemes()
        assert [e.name for e in pending] == ["untouched"]
        assert _get_url_adapter("untouched").__name__ == "Adapter"

    def test_import_failure_is_wrapped_and_not_permanent(self) -> None:
        pending = zarr.registry._url_adapter_registry.lazy_load_list
        pending.append(
            EntryPoint(name="broken", value="no_such_module_xyz:Adapter", group="zarr.url_adapters")
        )
        with pytest.raises(URLPipelineError, match="could not be loaded"):
            _get_url_adapter("broken")
        # still advertised, so a transient failure can be retried
        assert "broken" in _list_url_adapter_schemes()
        with pytest.raises(URLPipelineError, match="could not be loaded"):
            _get_url_adapter("broken")

    def test_registered_adapter_shadows_entry_point_with_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # a builtin registered under a scheme wins over a same-named entry
        # point, which is discarded loudly rather than left pending forever
        pending = zarr.registry._url_adapter_registry.lazy_load_list
        pending.append(
            _fake_entry_point(
                tmp_path,
                monkeypatch,
                scheme="shadowed",
                module="shadowed_adapter",
                source=_ADAPTER_SOURCE.format(prelude="raise RuntimeError('must not import')\n"),
            )
        )
        register_url_adapter("shadowed", NativeAdapter)
        with pytest.warns(ZarrUserWarning, match="already registered"):
            assert _get_url_adapter("shadowed") is NativeAdapter
        assert not any(e.name == "shadowed" for e in pending)
        # a second lookup is silent
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _get_url_adapter("shadowed") is NativeAdapter

    def test_duplicate_entry_points_warn_and_first_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pending = zarr.registry._url_adapter_registry.lazy_load_list
        for module in ("dup_first", "dup_second"):
            pending.append(
                _fake_entry_point(
                    tmp_path,
                    monkeypatch,
                    scheme="dup",
                    module=module,
                    source=_ADAPTER_SOURCE.format(prelude=f"ORIGIN = {module!r}\n"),
                )
            )
        with pytest.warns(ZarrUserWarning, match="multiple 'zarr.url_adapters' entry points"):
            cls = _get_url_adapter("dup")
        assert getattr(sys.modules[cls.__module__], "ORIGIN", None) == "dup_first"
        assert not any(e.name == "dup" for e in pending)


class TestStringsAreNeverPipelines:
    """
    Only a `URLPipeline` object is resolved as a pipeline. A plain string
    keeps the meaning it had before URL pipelines existed, whatever adapters
    are registered.
    """

    @pytest.mark.skipif(sys.platform == "win32", reason="'|' is not allowed in Windows file names")
    async def test_local_path_containing_pipe_is_a_local_store(self, tmp_path: Path) -> None:
        """A string path with `|` and a registered adapter's name is still a local directory."""
        register_url_adapter("wrap", WrapperAdapter)
        store_path = await make_store_path(f"{tmp_path}/a|wrap:x", mode="w")
        assert isinstance(store_path.store, LocalStore)
        assert store_path.store.root == tmp_path / "a|wrap:x"

    async def test_registered_root_scheme_string_is_not_routed(self) -> None:
        """A URL whose scheme names a registered root adapter is not handed to that adapter."""
        calls: list[PipelineSegment] = []

        class CountingRootAdapter(RootAdapter):
            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> StorePath:
                calls.append(segment)
                return await super().open_pipeline_segment(segment, context)

        register_url_adapter("rooty", CountingRootAdapter)
        # the string falls through to the fsspec route, which may raise for
        # an unknown protocol; whatever happens, the adapter is not consulted
        with contextlib.suppress(Exception):
            await make_store_path("rooty://org/repo")
        assert calls == []

    async def test_native_schemes_do_not_route_to_adapters(self) -> None:
        """Even an adapter forced into the registry never handles `file:`/`memory:`."""
        zarr.registry._url_adapter_registry.register(RootAdapter, "file")
        zarr.registry._url_adapter_registry.register(RootAdapter, "memory")
        store = await make_store(URLPipeline.from_url("memory://x"))
        assert isinstance(store, ManagedMemoryStore)
        for url in ("memory:|file:/etc", "memory:|memory:x"):
            with pytest.raises(URLPipelineError, match="cannot be an adapter segment"):
                await make_store_path(URLPipeline.from_url(url))


class TestResolve:
    async def test_wrapper_adapter_chain(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        result = await URLPipeline.from_url(
            f"file:{tmp_path.as_posix()}|wrap:inner/path"
        )._resolve_store_path()
        assert isinstance(result.store, TracingStore)
        assert result.path == "inner/path"
        assert result.store.context.preceding_url == f"file:{tmp_path.as_posix()}"
        assert result.store.segment.scheme == "wrap"
        assert result.store.segment.body == "inner/path"

    async def test_nested_wrappers_preserve_residual_paths(self, tmp_path: Path) -> None:
        # each wrapper joins the preceding residual path with its own, so
        # no segment's path is lost in root|wrap:a|wrap:b
        register_url_adapter("wrap", WrapperAdapter)
        result = await URLPipeline.from_url(
            f"file:{tmp_path.as_posix()}|wrap:a|wrap:b"
        )._resolve_store_path()
        assert result.path == "a/b"

    async def test_native_adapter_gets_preceding_url(self) -> None:
        register_url_adapter("native", NativeAdapter)
        NativeAdapter.seen_urls.clear()
        await URLPipeline.from_url("s3://bucket/repo|native:")._resolve_store_path()
        assert NativeAdapter.seen_urls == ["s3://bucket/repo"]

    async def test_multi_segment_preceding_url(self) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        register_url_adapter("native", NativeAdapter)
        NativeAdapter.seen_urls.clear()
        await URLPipeline.from_url("memory://base|wrap:a|native:x")._resolve_store_path()
        assert NativeAdapter.seen_urls == ["memory://base|wrap:a"]

    async def test_root_adapter(self) -> None:
        register_url_adapter("rooty", RootAdapter)
        result = await URLPipeline.from_url("rooty://org/repo")._resolve_store_path()
        assert result.path == "org/repo"
        assert result.store.read_only

    async def test_root_adapter_composes_with_chain(self) -> None:
        register_url_adapter("rooty", RootAdapter)
        register_url_adapter("native", NativeAdapter)
        NativeAdapter.seen_urls.clear()
        await URLPipeline.from_url("rooty://org/repo|native:x")._resolve_store_path()
        assert NativeAdapter.seen_urls == ["rooty://org/repo"]

    async def test_read_only_flag(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        result = await URLPipeline.from_url(
            f"file:{tmp_path.as_posix()}|wrap:"
        )._resolve_store_path(mode="r")
        assert isinstance(result.store, TracingStore)
        assert result.store.context.read_only
        assert result.store.context.mode == "r"
        result = await URLPipeline.from_url(
            f"file:{tmp_path.as_posix()}|wrap:"
        )._resolve_store_path()
        assert isinstance(result.store, TracingStore)
        assert not result.store.context.read_only
        assert result.store.context.mode is None

    async def test_read_only_is_enforced_on_disobedient_adapters(self) -> None:
        # the resolver downgrades a writable store returned under mode "r"
        # instead of trusting the adapter to have honored context.read_only
        register_url_adapter("bad", DisobedientAdapter)
        result = await URLPipeline.from_url("memory://base|bad:")._resolve_store_path(mode="r")
        assert result.store.read_only
        store = await make_store(URLPipeline.from_url("memory://base|bad:"), mode="r")
        assert store.read_only

    async def test_read_only_enforcement_without_conversion_raises(self) -> None:
        # a disobedient adapter whose store cannot be converted read-only
        class StubbornStore(MemoryStore):
            def with_read_only(self, read_only: bool = False) -> MemoryStore:
                raise NotImplementedError

        class StubbornAdapter(URLPipelineAdapter):
            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> StorePath:
                return StorePath(await StubbornStore.open(read_only=False))

        register_url_adapter("stubborn", StubbornAdapter)
        with pytest.raises(URLPipelineError, match="does not support read-only conversion"):
            await URLPipeline.from_url("memory://base|stubborn:")._resolve_store_path(mode="r")

    async def test_read_only_enforcement_failure_closes_store(self) -> None:
        # the adapter's store must not leak when it cannot be made read-only
        closed: list[bool] = []

        class StubbornStore(MemoryStore):
            def with_read_only(self, read_only: bool = False) -> MemoryStore:
                raise NotImplementedError

            def close(self) -> None:
                closed.append(True)
                super().close()

        class StubbornAdapter(URLPipelineAdapter):
            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> StorePath:
                return StorePath(await StubbornStore.open(read_only=False))

        register_url_adapter("stubborn", StubbornAdapter)
        with pytest.raises(URLPipelineError):
            await URLPipeline.from_url("memory://base|stubborn:")._resolve_store_path(mode="r")
        assert closed == [True]

    async def test_resolve_preceding_mode_override(self, tmp_path: Path) -> None:
        # a wrapper that only reads the preceding resource can open it
        # read-only regardless of the caller's mode
        class ReadOnlyRootWrapper(WrapperAdapter):
            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> StorePath:
                preceding = await dataclasses.replace(context, mode="r").resolve_preceding()
                assert preceding.store.read_only
                return StorePath(preceding.store, segment.body)

        register_url_adapter("rowrap", ReadOnlyRootWrapper)
        result = await URLPipeline.from_url(
            f"file:{tmp_path.as_posix()}|rowrap:x"
        )._resolve_store_path(mode="w")
        assert result.path == "x"

    async def test_wrapper_on_local_file_root_read_only(self, tmp_path: Path) -> None:
        # a wrapper that opens the preceding local *file* read-only gets a
        # store rooted at that file (the mechanism a read-only zip: adapter
        # can build on)
        target = tmp_path / "data.bin"
        target.write_bytes(b"payload")
        register_url_adapter("wrap", WrapperAdapter)
        result = await URLPipeline.from_url(f"file:{target.as_posix()}|wrap:")._resolve_store_path(
            mode="r"
        )
        assert isinstance(result.store, TracingStore)
        assert result.store.read_only

    @pytest.mark.parametrize("mode", [None, "a", "w", "r+", "w-"])
    @pytest.mark.xfail(
        strict=True,
        raises=FileExistsError,
        reason=(
            "known limitation: there is no file-resource primitive yet, so resolving a "
            "local file root in a writable mode hits LocalStore's create-on-open mkdir"
        ),
    )
    async def test_wrapper_on_local_file_root_writable_modes(
        self, tmp_path: Path, mode: AccessModeLiteral | None
    ) -> None:
        target = tmp_path / "data.bin"
        target.write_bytes(b"payload")
        register_url_adapter("wrap", WrapperAdapter)
        await URLPipeline.from_url(f"file:{target.as_posix()}|wrap:")._resolve_store_path(mode=mode)

    async def test_resolve_preceding_storage_options_override(self, tmp_path: Path) -> None:
        # an adapter that consumed its namespaced keys strips them before
        # the root store ever sees them
        class ConsumingWrapper(WrapperAdapter):
            seen: ClassVar[object | None] = None

            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> StorePath:
                options = dict(context.storage_options or {})
                cls.seen = options.pop("consuming_secret", None)
                preceding = await dataclasses.replace(
                    context, storage_options=options or None
                ).resolve_preceding()
                return StorePath(preceding.store, segment.body)

        register_url_adapter("consuming", ConsumingWrapper)
        # the local-path root rejects any surviving storage_options, so this
        # passing proves the adapter's keys were stripped before resolution
        result = await URLPipeline.from_url(
            f"file:{tmp_path.as_posix()}|consuming:x"
        )._resolve_store_path(
            storage_options={"consuming_secret": "s3cr3t"},
        )
        assert result.path == "x"
        assert ConsumingWrapper.seen == "s3cr3t"

    async def test_storage_options_visible_to_adapter(self) -> None:
        class OptionsProbe(NativeAdapter):
            seen_options: dict[str, object] | None = None

            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> StorePath:
                cls.seen_options = context.storage_options
                return await super().open_pipeline_segment(segment, context)

        register_url_adapter("probe", OptionsProbe)
        opts = {"anon": True}
        await URLPipeline.from_url("s3://bucket/x|probe:")._resolve_store_path(storage_options=opts)
        assert OptionsProbe.seen_options == opts

    async def test_wrapper_at_root_position_raises(self) -> None:
        # a wrapper adapter used as the pipeline root has nothing to wrap
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="no preceding sub-URL"):
            await URLPipeline.from_url("wrap:whatever")._resolve_store_path()

    async def test_single_url_is_a_trivial_pipeline(self) -> None:
        """A URL without `|` resolves as a pipeline consisting of its root alone."""
        result = await URLPipeline.from_url("memory://plain")._resolve_store_path()
        assert isinstance(result.store, ManagedMemoryStore)
        assert result.path == ""

    async def test_format_segment_path_joins_residual_path(self, tmp_path: Path) -> None:
        """The node path of a trailing format segment extends the adapters' residual path."""
        register_url_adapter("wrap", WrapperAdapter)
        result = await URLPipeline.from_url(
            f"file:{tmp_path.as_posix()}|wrap:a|zarr3:b"
        )._resolve_store_path()
        assert result.path == "a/b"

    async def test_unknown_adapter_scheme_raises(self) -> None:
        with pytest.raises(URLPipelineError, match="no URL pipeline adapter is registered"):
            await URLPipeline.from_url("memory://base|no-such-adapter:")._resolve_store_path()

    async def test_root_value_error_is_wrapped(self) -> None:
        # a ValueError from the root store is re-raised as URLPipelineError
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="could not resolve the pipeline root"):
            await URLPipeline.from_url("vendor.x://[authority]/path|wrap:")._resolve_store_path()


class TestSpecRoots:
    """The spec's `memory:` / `file:` root semantics inside pipelines."""

    @pytest.mark.parametrize("root", ["memory:", "memory:/", "memory://"])
    async def test_memory_root_spellings_equivalent(self, root: str) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        result = await URLPipeline.from_url(f"{root}|wrap:")._resolve_store_path()
        assert isinstance(result.store, TracingStore)
        inner = result.store._store
        assert isinstance(inner, ManagedMemoryStore)
        assert inner._name == ""
        assert inner.path == ""

    @pytest.mark.parametrize("root", ["memory:a/b", "memory:/a/b", "memory://a/b"])
    async def test_memory_root_path_spellings_equivalent(self, root: str) -> None:
        """There is one in-memory store; the whole body is a path within it."""
        register_url_adapter("wrap", WrapperAdapter)
        result = await URLPipeline.from_url(f"{root}|wrap:")._resolve_store_path()
        assert isinstance(result.store, TracingStore)
        inner = result.store._store
        assert isinstance(inner, ManagedMemoryStore)
        assert inner._name == ""
        assert inner.path == "a/b"

    async def test_memory_root_rejects_query(self) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="do not accept a query"):
            await URLPipeline.from_url("memory:a?opt=1|wrap:")._resolve_store_path()

    async def test_file_root_localhost_authority(self, tmp_path: Path) -> None:
        # file://localhost/p and file:///p are equivalent to file:/p
        register_url_adapter("wrap", WrapperAdapter)
        # spell the path URL-style: on Windows C:/... gains a leading slash
        posix = tmp_path.as_posix()
        url_path = posix if posix.startswith("/") else f"/{posix}"
        zarr.open_group(URLPipeline.from_url(f"file://localhost{url_path}|wrap:"), mode="w")
        opened = zarr.open_group(URLPipeline.from_url(f"file:{tmp_path}|wrap:"), mode="r")
        assert isinstance(opened, zarr.Group)
        zarr.open_group(URLPipeline.from_url(f"file://{url_path}|wrap:"), mode="r")

    @pytest.mark.skipif(sys.platform != "win32", reason="drive paths are Windows-only")
    async def test_file_root_windows_drive_spellings(self, tmp_path: Path) -> None:
        # on Windows, drive paths count as absolute, and the URL spelling
        # file:///C:/... has its leading slash stripped before the drive
        register_url_adapter("wrap", WrapperAdapter)
        drive_path = tmp_path.as_posix()  # C:/Users/...
        r1 = await URLPipeline.from_url(f"file:{drive_path}|wrap:")._resolve_store_path()
        r2 = await URLPipeline.from_url(f"file:///{drive_path}|wrap:")._resolve_store_path()
        assert isinstance(r1.store, TracingStore)
        assert isinstance(r2.store, TracingStore)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX path semantics")
    async def test_file_root_drive_letter_is_not_special_on_posix(self) -> None:
        """Off Windows, `/C:/x` is an ordinary absolute path and `C:/x` is relative."""
        with pytest.raises(URLPipelineError, match="absolute local path"):
            await make_store_path(URLPipeline.from_url("file:C:/drive/data"), mode="r")
        with pytest.raises(FileNotFoundError, match="/C:/drive/data"):
            await make_store_path(URLPipeline.from_url("file:///C:/drive/data"), mode="r")

    async def test_file_root_rejects_other_authority(self) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="unsupported authority"):
            await URLPipeline.from_url("file://example.com/tmp/x|wrap:")._resolve_store_path()

    @pytest.mark.parametrize("root", ["file:relative/path", "file://localhost"])
    async def test_file_root_rejects_relative_paths(self, root: str) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="absolute local path"):
            await URLPipeline.from_url(f"{root}|wrap:")._resolve_store_path()

    async def test_file_root_rejects_query(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="do not accept a query"):
            await URLPipeline.from_url(f"file:{tmp_path}?v=1|wrap:")._resolve_store_path()

    @pytest.mark.skipif(
        not _has_async_fsspec_wrapper,
        reason="plain memory:// routes to fsspec only with fsspec>=2024.12.0",
    )
    async def test_memory_root_is_distinct_from_fsspec_memory_url(self) -> None:
        # known divergence: a plain memory:// URL is fsspec's in-memory
        # filesystem when fsspec is installed, while a pipeline's memory: root
        # is zarr's ManagedMemoryStore; the two do not share data
        register_url_adapter("wrap", WrapperAdapter)
        plain = await make_store("memory://distinct-store")
        piped = await make_store(URLPipeline.from_url("memory://distinct-store|wrap:"))
        assert not isinstance(plain, ManagedMemoryStore)
        assert isinstance(piped, TracingStore)
        assert isinstance(piped._store, ManagedMemoryStore)

    async def test_file_root_does_not_percent_decode(self, tmp_path: Path) -> None:
        # documented divergence from RFC 8089: consistent with LocalStore, no
        # percent-escape is decoded, so %20 names a literal directory
        register_url_adapter("wrap", WrapperAdapter)
        await URLPipeline.from_url(f"file:{tmp_path.as_posix()}/a%20b|wrap:")._resolve_store_path()
        assert (tmp_path / "a%20b").is_dir()
        assert not (tmp_path / "a b").exists()


class TestMakeStoreIntegration:
    """`make_store` / `make_store_path` accept a `URLPipeline` like any other StoreLike."""

    async def test_make_store_path_combines_paths(self, tmp_path: Path) -> None:
        """Adapter residual path, format-segment node path and the caller's path join in order."""
        register_url_adapter("wrap", WrapperAdapter)
        pipeline = URLPipeline.from_url(f"file:{tmp_path.as_posix()}|wrap:residual|zarr3:node")
        store_path = await make_store_path(pipeline, path="user/sub")
        assert store_path.path == "residual/node/user/sub"
        assert isinstance(store_path.store, TracingStore)

    async def test_make_store_rejects_residual_path(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="resolves to a path inside a store"):
            await make_store(URLPipeline.from_url(f"file:{tmp_path.as_posix()}|wrap:residual"))

    async def test_make_store_rejects_format_segment_node_path(self, tmp_path: Path) -> None:
        """A node path given on the format segment also addresses a path inside the store."""
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="resolves to a path inside a store"):
            await make_store(URLPipeline.from_url(f"file:{tmp_path.as_posix()}|wrap:|zarr3:node"))

    async def test_make_store_accepts_slash_root_path(self, tmp_path: Path) -> None:
        # an adapter returning path="/" addresses the store root; the raw
        # value is normalized before the residual-path rejection
        class SlashRootAdapter(WrapperAdapter):
            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> StorePath:
                preceding = await context.resolve_preceding()
                return StorePath(preceding.store, "/")

        register_url_adapter("slashy", SlashRootAdapter)
        store = await make_store(URLPipeline.from_url(f"file:{tmp_path.as_posix()}|slashy:"))
        assert store is not None

    async def test_make_store_no_residual_path(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        store = await make_store(URLPipeline.from_url(f"file:{tmp_path.as_posix()}|wrap:"))
        assert isinstance(store, TracingStore)

    async def test_make_store_rejects_invalid_mode_before_resolving(self, tmp_path: Path) -> None:
        # an invalid mode is rejected upfront, like every other StoreLike,
        # rather than being handed to adapters
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(ValueError, match="Invalid mode"):
            await make_store(
                URLPipeline.from_url(f"file:{tmp_path.as_posix()}|wrap:"),
                mode="invalid",  # type: ignore[arg-type]
            )

    async def test_zarr_open_end_to_end(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        group = zarr.open_group(URLPipeline.from_url(f"file:{tmp_path.as_posix()}|wrap:"), mode="w")
        array = group.create_array("x", shape=(4,), dtype="i4")
        array[:] = [1, 2, 3, 4]
        assert zarr.open_array(URLPipeline.from_url(f"file:{tmp_path.as_posix()}|wrap:x"))[2] == 3

    async def test_pipeline_store_read_only_mode(self) -> None:
        register_url_adapter("native", NativeAdapter)
        store_path = await make_store_path(URLPipeline.from_url("memory://base|native:"), mode="r")
        assert store_path.read_only

    async def test_mode_a_on_read_only_store_raises(self) -> None:
        """Mode "a" on a read-only adapter store raises, as for any read-only store."""
        register_url_adapter("rooty", RootAdapter)
        with pytest.raises(ValueError, match="Store is read-only but mode is 'a'"):
            await make_store_path(URLPipeline.from_url("rooty://org/repo"), mode="a")

    async def test_explicit_write_mode_reaches_adapter(self) -> None:
        register_url_adapter("rooty", RootAdapter)
        with pytest.raises(ValueError, match="read-only scheme"):
            await make_store_path(URLPipeline.from_url("rooty://org/repo"), mode="w")

    async def test_storage_options_forwarded_to_root(self) -> None:
        # storage_options reach the root sub-URL via resolve_preceding ->
        # make_store. A local-path root does not accept storage_options, so
        # forwarding them must raise the same TypeError as a non-pipeline open.
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(TypeError, match="'storage_options' was provided but unused"):
            await make_store(
                URLPipeline.from_url("file:/tmp/some/path|wrap:"), storage_options={"anon": True}
            )


class TestOpenFunctionsAcceptPipelines:
    """
    `zarr.open`, `open_group`, `open_array` and `open_consolidated` accept a
    `URLPipeline` and open the node it addresses; the body of a trailing format
    segment is the node path.
    """

    @pytest.fixture
    def root(self, tmp_path: Path) -> str:
        register_url_adapter("wrap", WrapperAdapter)
        url = f"file:{tmp_path.as_posix()}|wrap:"
        group = zarr.create_group(URLPipeline.from_url(url + "|zarr2:"), zarr_format=2)
        group.create_array("sub/x", shape=(3,), dtype="i4")[:] = [1, 2, 3]
        zarr.consolidate_metadata(URLPipeline.from_url(url + "|zarr2:"))
        return url

    async def test_open(self, root: str) -> None:
        group = zarr.open(URLPipeline.from_url(root + "|zarr2:"), mode="r")
        assert isinstance(group, zarr.Group)
        assert group.metadata.zarr_format == 2
        assert isinstance(group.store, TracingStore)
        array = zarr.open(URLPipeline.from_url(root + "|zarr2:sub/x"), mode="r")
        assert isinstance(array, zarr.Array)
        assert array[1] == 2

    async def test_open_group(self, root: str) -> None:
        group = zarr.open_group(URLPipeline.from_url(root + "|zarr2:sub"), mode="r")
        assert group.path == "sub"
        assert group.read_only

    async def test_open_array(self, root: str) -> None:
        array = zarr.open_array(URLPipeline.from_url(root + "sub|zarr2:x"), mode="r")
        assert array.store_path.path == "sub/x"
        assert array[2] == 3

    async def test_open_consolidated(self, root: str) -> None:
        group = zarr.open_consolidated(URLPipeline.from_url(root + "|zarr2:"), mode="r")
        assert group.metadata.consolidated_metadata is not None
        assert group.metadata.zarr_format == 2

    async def test_async_open(self, root: str) -> None:
        group = await zarr.api.asynchronous.open(
            store=URLPipeline.from_url(root + "|zarr2:"), mode="r"
        )
        assert isinstance(group, zarr.AsyncGroup)

    @pytest.mark.parametrize(
        "open_func",
        [zarr.open, zarr.open_group, zarr.open_array, zarr.open_consolidated],
    )
    @pytest.mark.skipif(sys.platform == "win32", reason="'|' is not allowed in Windows file names")
    async def test_strings_are_not_pipelines(
        self, open_func: Callable[..., object], root: str, tmp_path: Path
    ) -> None:
        """The same text passed as a string addresses a local path, not the pipeline."""
        with pytest.raises(FileNotFoundError):
            open_func(f"{tmp_path}|wrap:|zarr2:", mode="r")


class TestZarrFormatFromPipeline:
    """
    Every function taking a StoreLike honors the format selected by a
    `URLPipeline`, not only the top-level zarr.api functions. Functions whose
    `zarr_format` defaults to 3 raise when the default conflicts with the
    pipeline; `create_array` and `Group.open` accept `zarr_format=None` to defer
    to it.
    """

    @pytest.fixture
    def v2_pipeline(self, tmp_path: Path) -> URLPipeline:
        return URLPipeline.from_url(f"file:{tmp_path.as_posix()}|zarr2:")

    async def test_create_array(self, v2_pipeline: URLPipeline) -> None:
        arr = zarr.create_array(v2_pipeline, name="x", shape=(2,), dtype="i4", zarr_format=None)
        assert arr.metadata.zarr_format == 2
        with pytest.raises(ValueError, match="conflicts with"):
            zarr.create_array(v2_pipeline, name="y", shape=(2,), dtype="i4")

    async def test_create_array_from_data(self, v2_pipeline: URLPipeline) -> None:
        # the data path of create_array goes through from_array
        arr = zarr.create_array(v2_pipeline, name="x", data=np.arange(3), zarr_format=None)
        assert arr.metadata.zarr_format == 2
        with pytest.raises(ValueError, match="conflicts with"):
            zarr.create_array(v2_pipeline, name="y", data=np.arange(3))

    async def test_create_array_default_without_pipeline(self, tmp_path: Path) -> None:
        """The default format for plain stores is still 3, independent of the config default."""
        with zarr.config.set({"default_zarr_format": 2}):
            arr = zarr.create_array(tmp_path, shape=(2,), dtype="i4")
        assert arr.metadata.zarr_format == 3

    async def test_from_array(self, v2_pipeline: URLPipeline) -> None:
        arr = zarr.from_array(v2_pipeline, name="x", data=np.arange(3))
        assert arr.metadata.zarr_format == 2

    async def test_group_from_store(self, v2_pipeline: URLPipeline, tmp_path: Path) -> None:
        group = zarr.Group.from_store(v2_pipeline, zarr_format=2)
        assert group.metadata.zarr_format == 2
        with pytest.raises(ValueError, match="conflicts with"):
            zarr.Group.from_store(URLPipeline.from_url(f"file:{tmp_path.as_posix()}/other|zarr2:"))

    async def test_group_from_store_default_without_pipeline(self, tmp_path: Path) -> None:
        """The default format for plain stores is still 3, independent of the config default."""
        with zarr.config.set({"default_zarr_format": 2}):
            group = zarr.Group.from_store(tmp_path)
        assert group.metadata.zarr_format == 3

    async def test_group_open(self, v2_pipeline: URLPipeline) -> None:
        zarr.Group.from_store(v2_pipeline, zarr_format=2)
        group = zarr.Group.open(v2_pipeline, zarr_format=None)
        assert group.metadata.zarr_format == 2
        with pytest.raises(ValueError, match="conflicts with"):
            zarr.Group.open(v2_pipeline)

    async def test_array_open(self, v2_pipeline: URLPipeline, tmp_path: Path) -> None:
        zarr.create_array(v2_pipeline, name="x", shape=(2,), dtype="i4", zarr_format=None)
        # Array.open opens zarr v3 only, so a v2 pipeline conflicts with it
        with pytest.raises(ValueError, match="conflicts with"):
            zarr.Array.open(URLPipeline.from_url(f"file:{tmp_path.as_posix()}|zarr2:x"))
        zarr.create_array(
            URLPipeline.from_url(f"file:{tmp_path.as_posix()}/v3|zarr3:"),
            name="y",
            shape=(2,),
            dtype="i4",
        )
        arr = zarr.Array.open(URLPipeline.from_url(f"file:{tmp_path.as_posix()}/v3|zarr3:y"))
        assert arr.metadata.zarr_format == 3

    async def test_open_like(self, v2_pipeline: URLPipeline) -> None:
        # open_like inherits v2 filters/compressor from the reference; the
        # pipeline supplies the format, which open_array then creates with
        ref = zarr.create_array({}, shape=(2,), dtype="i4", zarr_format=2)
        arr = zarr.open_like(ref, store=v2_pipeline, path="x")
        assert arr.metadata.zarr_format == 2

    async def test_deprecated_async_array_create(
        self, v2_pipeline: URLPipeline, tmp_path: Path
    ) -> None:
        from zarr.core.array import AsyncArray

        arr = await AsyncArray._create(v2_pipeline, shape=(2,), dtype="i4", zarr_format=2)
        assert arr.metadata.zarr_format == 2
        with pytest.raises(ValueError, match="conflicts with"):
            await AsyncArray._create(
                URLPipeline.from_url(f"file:{tmp_path.as_posix()}/o|zarr2:"),
                shape=(2,),
                dtype="i4",
                zarr_format=3,
            )
