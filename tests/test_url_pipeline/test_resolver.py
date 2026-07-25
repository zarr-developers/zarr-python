from __future__ import annotations

import dataclasses
import sys
from typing import TYPE_CHECKING, ClassVar

import pytest

import zarr
import zarr.registry
from zarr.abc.store import Store
from zarr.abc.url_pipeline import (
    AdapterResolution,
    PipelineContext,
    PipelineSegment,
    URLPipelineAdapter,
)
from zarr.errors import URLPipelineError, ZarrUserWarning
from zarr.registry import (
    get_url_adapter,
    list_url_adapter_schemes,
    register_url_adapter,
)
from zarr.storage import ManagedMemoryStore, MemoryStore, WrapperStore
from zarr.storage._common import make_store, make_store_path
from zarr.storage._url_pipeline import is_url_pipeline, resolve_pipeline
from zarr.storage._utils import _join_paths

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures("clean_url_adapter_registry")


class TracingStore(WrapperStore[Store]):
    """Wrapper that records the context it was created from."""

    context: PipelineContext
    segment: PipelineSegment


class WrapperAdapter(URLPipelineAdapter):
    """
    A wrapper-style adapter: resolves the preceding pipeline into a store.

    Follows the wrapper contract: the preceding resolution's residual path
    is carried forward (joined with this segment's own path), and unchanged
    fields survive via `dataclasses.replace`.
    """

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> AdapterResolution:
        preceding = await context.resolve_preceding()
        store = TracingStore(preceding.store)
        store.context = context
        store.segment = segment
        return dataclasses.replace(
            preceding, store=store, path=_join_paths([preceding.path, segment.body])
        )


class NativeAdapter(URLPipelineAdapter):
    """A native-style adapter: consumes the preceding URL as a string."""

    seen_urls: ClassVar[list[str]] = []

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> AdapterResolution:
        cls.seen_urls.append(context.preceding_url)
        store = await MemoryStore.open(read_only=context.read_only)
        return AdapterResolution(store=store, path=segment.body)


class RootAdapter(URLPipelineAdapter):
    """A root-scheme adapter (no preceding segments), like al://."""

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> AdapterResolution:
        assert context.preceding == ()
        if context.mode in ("w", "w-", "r+"):
            raise ValueError("read-only scheme")
        store = await MemoryStore.open(read_only=True)
        return AdapterResolution(store=store, path=segment.body.lstrip("/"))


class DisobedientAdapter(URLPipelineAdapter):
    """An adapter that ignores `context.read_only` (a contract violation)."""

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> AdapterResolution:
        return AdapterResolution(store=await MemoryStore.open(read_only=False))


class FormatAdapter(URLPipelineAdapter):
    """A dummy format-selecting adapter, like the builtin `zarr2:`."""

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> AdapterResolution:
        preceding = await context.resolve_preceding()
        return dataclasses.replace(preceding, zarr_format=2)


class TestRegistry:
    def test_register_and_get(self) -> None:
        register_url_adapter("demo", WrapperAdapter)
        assert get_url_adapter("demo") is WrapperAdapter
        assert get_url_adapter("DEMO") is WrapperAdapter
        assert "demo" in list_url_adapter_schemes()

    def test_unknown_scheme(self) -> None:
        with pytest.raises(URLPipelineError, match="no URL pipeline adapter is registered"):
            get_url_adapter("nonexistent-scheme")

    def test_reregistering_scheme_warns(self) -> None:
        register_url_adapter("demo", WrapperAdapter)
        with pytest.warns(ZarrUserWarning, match="is being replaced"):
            register_url_adapter("demo", NativeAdapter)
        assert get_url_adapter("demo") is NativeAdapter
        # re-registering the same class is not a collision
        register_url_adapter("demo", NativeAdapter)

    @pytest.mark.usefixtures("set_path")
    def test_entrypoint_discovery(self) -> None:
        assert "example-pkg.entrypoint-scheme" in list_url_adapter_schemes()
        cls = get_url_adapter("example-pkg.entrypoint-scheme")
        assert cls.__name__ == "TestEntrypointURLAdapter"

    @pytest.mark.usefixtures("set_path")
    async def test_entrypoint_end_to_end(self) -> None:
        result = await resolve_pipeline("memory://src|example-pkg.entrypoint-scheme:sub/path")
        assert result.path == "sub/path"

    @pytest.mark.usefixtures("set_path")
    def test_entrypoint_scheme_lookup_is_case_insensitive(self) -> None:
        # entry-point names are matched case-insensitively, like schemes
        cls = get_url_adapter("Example-PKG.Entrypoint-Scheme")
        assert cls.__name__ == "TestEntrypointURLAdapter"

    @pytest.mark.usefixtures("set_path")
    def test_loading_one_scheme_leaves_others_pending(self) -> None:
        # resolving one scheme must not import other providers' entry points
        registry = zarr.registry._url_adapter_registry
        assert any(e.name == "example-pkg.entrypoint-scheme" for e in registry.lazy_load_list)
        with pytest.raises(URLPipelineError, match="no URL pipeline adapter"):
            get_url_adapter("some-other-scheme")
        assert any(e.name == "example-pkg.entrypoint-scheme" for e in registry.lazy_load_list)
        assert get_url_adapter("example-pkg.entrypoint-scheme").__name__ == (
            "TestEntrypointURLAdapter"
        )


class TestIsURLPipeline:
    def test_pipe_routes(self) -> None:
        assert is_url_pipeline("memory://x|demo:")

    def test_registered_root_scheme_routes(self) -> None:
        register_url_adapter("rooty", RootAdapter)
        assert is_url_pipeline("rooty://org/repo")

    @pytest.mark.parametrize("url", ["s3://bucket/key", "/local/path", "memory://x", "C:.zarr"])
    def test_plain_urls_do_not_route(self, url: str) -> None:
        assert not is_url_pipeline(url)

    def test_fsspec_chained_urls_do_not_route(self) -> None:
        # fsspec's ``scheme::`` chaining is not a URL pipeline: even with a
        # same-named root adapter registered, these keep flowing to fsspec.
        register_url_adapter("zip", RootAdapter)
        assert not is_url_pipeline("zip::file:///tmp/data.zip")
        assert not is_url_pipeline("simplecache::s3://bucket/key")

    def test_native_schemes_do_not_route_to_adapters(self) -> None:
        # an installed package must not be able to intercept zarr's own
        # local-path and in-memory routing by registering file:/memory:
        register_url_adapter("file", RootAdapter)
        register_url_adapter("memory", RootAdapter)
        assert not is_url_pipeline("file:/tmp/x")
        assert not is_url_pipeline("memory://x")

    def test_exotic_authority_root_scheme(self) -> None:
        # The spec's own root-URL example: urlparse rejects the bracketed
        # non-IP authority, so scheme detection must work on the raw string
        # rather than raising before routing.
        url = "vendor1-2.custom-1+proto.ext://[authority]/path?query/part?x"
        assert not is_url_pipeline(url)
        register_url_adapter("vendor1-2.custom-1+proto.ext", RootAdapter)
        assert is_url_pipeline(url)


class TestResolve:
    async def test_wrapper_adapter_chain(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        result = await resolve_pipeline(f"{tmp_path}|wrap:inner/path")
        assert isinstance(result.store, TracingStore)
        assert result.path == "inner/path"
        assert result.store.context.preceding_url == str(tmp_path)

    async def test_nested_wrappers_preserve_residual_paths(self, tmp_path: Path) -> None:
        # each wrapper joins the preceding residual path with its own, so
        # no segment's path is lost in root|wrap:a|wrap:b
        register_url_adapter("wrap", WrapperAdapter)
        result = await resolve_pipeline(f"{tmp_path}|wrap:a|wrap:b")
        assert result.path == "a/b"

    async def test_native_adapter_gets_preceding_url(self) -> None:
        register_url_adapter("native", NativeAdapter)
        NativeAdapter.seen_urls.clear()
        await resolve_pipeline("s3://bucket/repo|native:")
        assert NativeAdapter.seen_urls == ["s3://bucket/repo"]

    async def test_multi_segment_preceding_url(self) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        register_url_adapter("native", NativeAdapter)
        NativeAdapter.seen_urls.clear()
        await resolve_pipeline("memory://base|wrap:a|native:x")
        assert NativeAdapter.seen_urls == ["memory://base|wrap:a"]

    async def test_root_adapter(self) -> None:
        register_url_adapter("rooty", RootAdapter)
        result = await resolve_pipeline("rooty://org/repo")
        assert result.path == "org/repo"
        assert result.store.read_only

    async def test_root_adapter_composes_with_chain(self) -> None:
        register_url_adapter("rooty", RootAdapter)
        register_url_adapter("native", NativeAdapter)
        NativeAdapter.seen_urls.clear()
        await resolve_pipeline("rooty://org/repo|native:x")
        assert NativeAdapter.seen_urls == ["rooty://org/repo"]

    async def test_read_only_flag(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        result = await resolve_pipeline(f"{tmp_path}|wrap:", mode="r")
        assert isinstance(result.store, TracingStore)
        assert result.store.context.read_only
        assert result.store.context.mode == "r"
        result = await resolve_pipeline(f"{tmp_path}|wrap:")
        assert isinstance(result.store, TracingStore)
        assert not result.store.context.read_only
        assert result.store.context.mode is None

    async def test_read_only_is_enforced_on_disobedient_adapters(self) -> None:
        # the resolver downgrades a writable store returned under mode "r"
        # instead of trusting the adapter to have honored context.read_only
        register_url_adapter("bad", DisobedientAdapter)
        result = await resolve_pipeline("memory://base|bad:", mode="r")
        assert result.store.read_only
        store = await make_store("memory://base|bad:", mode="r")
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
            ) -> AdapterResolution:
                return AdapterResolution(store=await StubbornStore.open(read_only=False))

        register_url_adapter("stubborn", StubbornAdapter)
        with pytest.raises(URLPipelineError, match="does not support read-only conversion"):
            await resolve_pipeline("memory://base|stubborn:", mode="r")

    async def test_resolve_preceding_mode_override(self, tmp_path: Path) -> None:
        # a wrapper that only reads the preceding resource can open it
        # read-only regardless of the caller's mode
        class ReadOnlyRootWrapper(WrapperAdapter):
            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> AdapterResolution:
                preceding = await context.resolve_preceding(mode="r")
                assert preceding.store.read_only
                return dataclasses.replace(preceding, path=segment.body)

        register_url_adapter("rowrap", ReadOnlyRootWrapper)
        result = await resolve_pipeline(f"{tmp_path}|rowrap:x", mode="w")
        assert result.path == "x"

    async def test_resolve_preceding_storage_options_override(self, tmp_path: Path) -> None:
        # an adapter that consumed its namespaced keys strips them before
        # the root store ever sees them
        class ConsumingWrapper(WrapperAdapter):
            seen: ClassVar[object | None] = None

            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> AdapterResolution:
                options = dict(context.storage_options or {})
                cls.seen = options.pop("consuming_secret", None)
                preceding = await context.resolve_preceding(storage_options=options or None)
                return dataclasses.replace(preceding, path=segment.body)

        register_url_adapter("consuming", ConsumingWrapper)
        # the local-path root rejects any surviving storage_options, so this
        # passing proves the adapter's keys were stripped before resolution
        result = await resolve_pipeline(
            f"{tmp_path}|consuming:x", storage_options={"consuming_secret": "s3cr3t"}
        )
        assert result.path == "x"
        assert ConsumingWrapper.seen == "s3cr3t"

    async def test_storage_options_visible_to_adapter(self) -> None:
        register_url_adapter("native", NativeAdapter)

        class OptionsProbe(NativeAdapter):
            seen_options: dict[str, object] | None = None

            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> AdapterResolution:
                cls.seen_options = context.storage_options
                return await super().open_pipeline_segment(segment, context)

        register_url_adapter("probe", OptionsProbe)
        opts = {"anon": True}
        await resolve_pipeline("s3://bucket/x|probe:", storage_options=opts)
        assert OptionsProbe.seen_options == opts

    async def test_wrapper_at_root_position_raises(self) -> None:
        # a wrapper adapter used as the pipeline root has nothing to wrap
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="no preceding sub-URL"):
            await resolve_pipeline("wrap:whatever")

    async def test_not_a_pipeline_raises(self) -> None:
        with pytest.raises(URLPipelineError, match="is not a URL pipeline"):
            await resolve_pipeline("s3://bucket/plain")

    async def test_unknown_adapter_scheme_raises(self) -> None:
        with pytest.raises(URLPipelineError, match="no URL pipeline adapter is registered"):
            await resolve_pipeline("memory://base|no-such-adapter:")

    async def test_unknown_adapter_error_hints_at_reserved_pipe(self, tmp_path: Path) -> None:
        # a filename containing '|' routes here; the error points at the fix
        with pytest.raises(URLPipelineError, match="pass a\\s+pathlib.Path"):
            await resolve_pipeline(f"{tmp_path}/a|b")

    async def test_base_case_value_error_is_wrapped(self) -> None:
        # a root that only the fallback scheme detection accepts must not
        # leak urlparse's ValueError out of the base case
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="could not resolve the pipeline root"):
            await resolve_pipeline("vendor.x://[authority]/path|wrap:")

    async def test_zarr_format_flows_from_adapter(self, tmp_path: Path) -> None:
        register_url_adapter("fmt2", FormatAdapter)
        result = await resolve_pipeline(f"{tmp_path}|fmt2:")
        assert result.zarr_format == 2
        # wrapper adapters carry the format through (dataclasses.replace)
        register_url_adapter("wrap", WrapperAdapter)
        result = await resolve_pipeline(f"{tmp_path}|fmt2:|wrap:")
        assert result.zarr_format == 2


class TestSpecRoots:
    """The spec's `memory:` / `file:` root semantics inside pipelines."""

    @pytest.mark.parametrize("root", ["memory:", "memory:/", "memory://"])
    async def test_memory_root_spellings_equivalent(self, root: str) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        result = await resolve_pipeline(f"{root}|wrap:")
        assert isinstance(result.store, TracingStore)
        inner = result.store._store
        assert isinstance(inner, ManagedMemoryStore)
        assert inner._name == ""
        assert inner.path == ""

    @pytest.mark.parametrize("root", ["memory:a/b", "memory:/a/b", "memory://a/b"])
    async def test_memory_root_named_spellings_equivalent(self, root: str) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        result = await resolve_pipeline(f"{root}|wrap:")
        assert isinstance(result.store, TracingStore)
        inner = result.store._store
        assert isinstance(inner, ManagedMemoryStore)
        assert inner._name == "a"
        assert inner.path == "b"

    async def test_memory_root_shares_data_across_spellings(self) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        group = zarr.open_group("memory:pipe-share|wrap:", mode="w")
        group.create_array("x", shape=(2,), dtype="i4")
        reopened = zarr.open_group("memory://pipe-share|wrap:", mode="r")
        assert "x" in reopened

    async def test_memory_root_rejects_query(self) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="do not accept a query"):
            await resolve_pipeline("memory:a?opt=1|wrap:")

    async def test_memory_root_rejects_storage_options(self) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(TypeError, match="'storage_options' was provided but unused"):
            await resolve_pipeline("memory:a|wrap:", storage_options={"anon": True})

    async def test_file_root_localhost_authority(self, tmp_path: Path) -> None:
        # file://localhost/p and file:///p are equivalent to file:/p
        register_url_adapter("wrap", WrapperAdapter)
        # spell the path URL-style: on Windows C:/... gains a leading slash
        posix = tmp_path.as_posix()
        url_path = posix if posix.startswith("/") else f"/{posix}"
        zarr.open_group(f"file://localhost{url_path}|wrap:", mode="w")
        opened = zarr.open_group(f"file:{tmp_path}|wrap:", mode="r")
        assert isinstance(opened, zarr.Group)
        zarr.open_group(f"file://{url_path}|wrap:", mode="r")

    async def test_file_root_windows_drive_spellings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Windows drive paths count as absolute, and the URL spelling
        # file:///C:/... has its leading slash stripped before the drive
        register_url_adapter("wrap", WrapperAdapter)
        if sys.platform == "win32":
            drive_path = tmp_path.as_posix()  # C:/Users/...
        else:
            # off Windows a drive path is just an odd relative directory;
            # chdir into tmp so it is created there
            monkeypatch.chdir(tmp_path)
            drive_path = "C:/drive/data"
        r1 = await resolve_pipeline(f"file:{drive_path}|wrap:")
        r2 = await resolve_pipeline(f"file:///{drive_path}|wrap:")
        assert isinstance(r1.store, TracingStore)
        assert isinstance(r2.store, TracingStore)

    async def test_file_root_rejects_other_authority(self) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="unsupported authority"):
            await resolve_pipeline("file://example.com/tmp/x|wrap:")

    @pytest.mark.parametrize("root", ["file:relative/path", "file://localhost"])
    async def test_file_root_rejects_relative_paths(self, root: str) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="absolute path"):
            await resolve_pipeline(f"{root}|wrap:")

    async def test_file_root_rejects_query(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="do not accept a query"):
            await resolve_pipeline(f"file:{tmp_path}?v=1|wrap:")


class TestMakeStoreIntegration:
    async def test_make_store_path_combines_paths(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        store_path = await make_store_path(f"{tmp_path}|wrap:residual", path="user/sub")
        assert store_path.path == "residual/user/sub"

    async def test_make_store_rejects_residual_path(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="resolves to a path inside a store"):
            await make_store(f"{tmp_path}|wrap:residual")

    async def test_make_store_rejects_slash_root_path(self, tmp_path: Path) -> None:
        # an adapter returning path="/" addresses the store root; the raw
        # value is normalized before the residual-path rejection
        class SlashRootAdapter(WrapperAdapter):
            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> AdapterResolution:
                preceding = await context.resolve_preceding()
                return dataclasses.replace(preceding, path="/")

        register_url_adapter("slashy", SlashRootAdapter)
        store = await make_store(f"{tmp_path}|slashy:")
        assert store is not None

    async def test_make_store_closes_store_on_residual_path_error(self) -> None:
        closed: list[bool] = []

        class ClosingStore(MemoryStore):
            def close(self) -> None:
                closed.append(True)
                super().close()

        class LeakProbe(URLPipelineAdapter):
            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> AdapterResolution:
                return AdapterResolution(store=await ClosingStore.open(), path="residual")

        register_url_adapter("leaky", LeakProbe)
        with pytest.raises(URLPipelineError, match="resolves to a path inside a store"):
            await make_store("memory://base|leaky:")
        assert closed == [True]

    async def test_make_store_no_residual_path(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        store = await make_store(f"{tmp_path}|wrap:")
        assert isinstance(store, TracingStore)

    async def test_zarr_open_end_to_end(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        group = zarr.open_group(f"{tmp_path}|wrap:", mode="w")
        array = group.create_array("x", shape=(4,), dtype="i4")
        array[:] = [1, 2, 3, 4]
        assert zarr.open_array(f"{tmp_path}|wrap:x")[2] == 3

    async def test_pipeline_store_read_only_mode(self) -> None:
        register_url_adapter("native", NativeAdapter)
        store_path = await make_store_path("memory://base|native:", mode="r")
        assert store_path.read_only

    async def test_mode_a_downgrades_to_read_only_open(self) -> None:
        # mode "a" (the zarr.open default) is open-or-create: a pipeline
        # that resolves to a read-only store serves the "open" half instead
        # of failing outright.
        register_url_adapter("rooty", RootAdapter)
        store_path = await make_store_path("rooty://org/repo", mode="a")
        assert store_path.read_only

    async def test_mode_a_downgrade_applies_to_plain_stores_too(self) -> None:
        # the open-or-create downgrade lives in StorePath.open, so it is
        # uniform across pipeline and non-pipeline stores
        store = await MemoryStore.open(read_only=True)
        store_path = await make_store_path(store, mode="a")
        assert store_path.read_only

    async def test_explicit_write_mode_reaches_adapter(self) -> None:
        register_url_adapter("rooty", RootAdapter)
        with pytest.raises(ValueError, match="read-only scheme"):
            await make_store_path("rooty://org/repo", mode="w")

    async def test_storage_options_forwarded_to_root(self) -> None:
        # storage_options reach the root sub-URL via resolve_preceding ->
        # make_store. A local-path root does not accept storage_options, so
        # forwarding them must raise the same TypeError as a non-pipeline open.
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(TypeError, match="'storage_options' was provided but unused"):
            await make_store("/tmp/some/path|wrap:", storage_options={"anon": True})

    async def test_store_path_truediv_keeps_format(self, tmp_path: Path) -> None:
        # StorePath.__truediv__ constructs a new instance; the pipeline-
        # selected format must survive the division
        register_url_adapter("fmt2", FormatAdapter)
        store_path = await make_store_path(f"{tmp_path}|fmt2:")
        assert store_path.zarr_format == 2
        assert (store_path / "sub/node").zarr_format == 2

    async def test_zarr_format_merges_into_open(self, tmp_path: Path) -> None:
        register_url_adapter("fmt2", FormatAdapter)
        group = zarr.open_group(f"{tmp_path}|fmt2:", mode="w")
        assert group.metadata.zarr_format == 2

    async def test_conflicting_explicit_format_raises(self, tmp_path: Path) -> None:
        register_url_adapter("fmt2", FormatAdapter)
        with pytest.raises(ValueError, match="conflicts with"):
            zarr.open_group(f"{tmp_path}|fmt2:", mode="w", zarr_format=3)

    async def test_matching_explicit_format_ok(self, tmp_path: Path) -> None:
        register_url_adapter("fmt2", FormatAdapter)
        group = zarr.open_group(f"{tmp_path}|fmt2:", mode="w", zarr_format=2)
        assert group.metadata.zarr_format == 2
