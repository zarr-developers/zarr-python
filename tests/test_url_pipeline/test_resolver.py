from __future__ import annotations

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
from zarr.errors import URLPipelineError
from zarr.registry import (
    get_url_adapter,
    list_url_adapter_schemes,
    register_url_adapter,
)
from zarr.storage import MemoryStore, WrapperStore
from zarr.storage._common import make_store, make_store_path
from zarr.storage._url_pipeline import is_url_pipeline, resolve_pipeline

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures("clean_url_adapter_registry")


class TracingStore(WrapperStore[Store]):
    """Wrapper that records the context it was created from."""

    context: PipelineContext
    segment: PipelineSegment


class WrapperAdapter(URLPipelineAdapter):
    """A wrapper-style adapter: resolves the preceding pipeline into a store."""

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> AdapterResolution:
        preceding = await context.resolve_preceding()
        store = TracingStore(preceding.store)
        store.context = context
        store.segment = segment
        return AdapterResolution(store=store, path=segment.body)


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


async def _store_of(resolution: AdapterResolution) -> Store:
    return resolution.store


class TestRegistry:
    def test_register_and_get(self) -> None:
        register_url_adapter("demo", WrapperAdapter)
        assert get_url_adapter("demo") is WrapperAdapter
        assert get_url_adapter("DEMO") is WrapperAdapter
        assert "demo" in list_url_adapter_schemes()

    def test_unknown_scheme(self) -> None:
        with pytest.raises(URLPipelineError, match="no URL pipeline adapter is registered"):
            get_url_adapter("nonexistent-scheme")

    @pytest.mark.usefixtures("set_path")
    def test_entrypoint_discovery(self) -> None:
        assert "entrypoint-scheme" in list_url_adapter_schemes()
        cls = get_url_adapter("entrypoint-scheme")
        assert cls.__name__ == "TestEntrypointURLAdapter"

    @pytest.mark.usefixtures("set_path")
    async def test_entrypoint_end_to_end(self) -> None:
        result = await resolve_pipeline("memory://src|entrypoint-scheme:sub/path")
        assert result.path == "sub/path"

    @pytest.mark.usefixtures("set_path")
    def test_loading_one_scheme_leaves_others_pending(self) -> None:
        # resolving one scheme must not import other providers' entry points
        registry = zarr.registry._url_adapter_registry
        assert any(e.name == "entrypoint-scheme" for e in registry.lazy_load_list)
        with pytest.raises(URLPipelineError, match="no URL pipeline adapter"):
            get_url_adapter("some-other-scheme")
        assert any(e.name == "entrypoint-scheme" for e in registry.lazy_load_list)
        assert get_url_adapter("entrypoint-scheme").__name__ == "TestEntrypointURLAdapter"


class TestIsURLPipeline:
    def test_pipe_routes(self) -> None:
        assert is_url_pipeline("memory://x|demo:")

    def test_registered_root_scheme_routes(self) -> None:
        register_url_adapter("rooty", RootAdapter)
        assert is_url_pipeline("rooty://org/repo")

    @pytest.mark.parametrize("url", ["s3://bucket/key", "/local/path", "memory://x", "C:.zarr"])
    def test_plain_urls_do_not_route(self, url: str) -> None:
        assert not is_url_pipeline(url)

    def test_exotic_authority_root_scheme(self) -> None:
        # The spec's own root-URL example: urlparse rejects the bracketed
        # non-IP authority, so scheme detection must use the same fallback
        # as the parser rather than raising before routing.
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
        with pytest.raises(URLPipelineError, match="requires a preceding sub-URL"):
            await resolve_pipeline("wrap:whatever")

    async def test_not_a_pipeline_raises(self) -> None:
        with pytest.raises(URLPipelineError, match="is not a URL pipeline"):
            await resolve_pipeline("s3://bucket/plain")

    async def test_unknown_adapter_scheme_raises(self) -> None:
        with pytest.raises(URLPipelineError, match="no URL pipeline adapter is registered"):
            await resolve_pipeline("memory://base|no-such-adapter:")


class TestMakeStoreIntegration:
    async def test_make_store_path_combines_paths(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        store_path = await make_store_path(f"{tmp_path}|wrap:residual", path="user/sub")
        assert store_path.path == "residual/user/sub"

    async def test_make_store_rejects_residual_path(self, tmp_path: Path) -> None:
        register_url_adapter("wrap", WrapperAdapter)
        with pytest.raises(URLPipelineError, match="resolves to a path inside a store"):
            await make_store(f"{tmp_path}|wrap:residual")

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
