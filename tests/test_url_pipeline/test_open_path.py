"""
URL pipelines through the public open path: `zarr.open_*`, `make_store_path`,
`make_store` and `consolidate_metadata`, rather than the private resolver.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, ClassVar

import pytest

import zarr
import zarr.registry
from zarr.abc.url_pipeline import URLPipelineAdapter
from zarr.errors import GroupNotFoundError, URLPipelineError, ZarrUserWarning
from zarr.registry import register_url_adapter
from zarr.storage import MemoryStore, PipelineSegment, StorePath, URLPipeline
from zarr.storage._common import make_store, make_store_path

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.storage import PipelineContext

pytestmark = pytest.mark.usefixtures("clean_url_adapter_registry")


class ClosingStore(MemoryStore):
    """A memory store that records how often it is closed."""

    closes: ClassVar[list[ClosingStore]] = []

    def close(self) -> None:
        ClosingStore.closes.append(self)
        super().close()


class RecordingAdapter(URLPipelineAdapter):
    """Records each call; returns a fresh `ClosingStore` honoring `read_only`."""

    calls: ClassVar[list[PipelineContext]] = []
    read_only_store: ClassVar[bool] = False

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> StorePath:
        cls.calls.append(context)
        store = await ClosingStore.open(read_only=cls.read_only_store or context.read_only)
        return StorePath(store, segment.body)


@pytest.fixture
def recording() -> type[RecordingAdapter]:
    RecordingAdapter.calls = []
    RecordingAdapter.read_only_store = False
    ClosingStore.closes = []
    register_url_adapter("rec", RecordingAdapter)
    return RecordingAdapter


class TestModeValidation:
    @pytest.mark.parametrize("mode", ["bogus", "R", "rw"])
    async def test_invalid_mode_rejected_before_adapters_run(
        self, recording: type[RecordingAdapter], mode: str
    ) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            zarr.open_group(URLPipeline.from_url("memory:|rec:"), mode=mode)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Invalid mode"):
            await make_store_path(URLPipeline.from_url("memory:|rec:"), mode=mode)  # type: ignore[arg-type]
        assert recording.calls == []

    async def test_invalid_mode_rejected_for_plain_stores(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            await make_store_path({}, mode="bogus")  # type: ignore[arg-type]


class TestStoresClosedOnError:
    def test_read_only_store_in_mode_a(self, recording: type[RecordingAdapter]) -> None:
        recording.read_only_store = True
        with pytest.raises(ValueError, match="Store is read-only but mode is 'a'"):
            zarr.open_group(URLPipeline.from_url("memory:|rec:"), mode="a")
        assert len(ClosingStore.closes) == 1

    async def test_mode_w_minus_on_existing_data(self, recording: type[RecordingAdapter]) -> None:
        class NonEmptyAdapter(RecordingAdapter):
            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> StorePath:
                store_path = await super().open_pipeline_segment(segment, context)
                zarr.create_group(store_path.store)
                return store_path

        register_url_adapter("full", NonEmptyAdapter)
        with pytest.raises(FileExistsError):
            await make_store_path(URLPipeline.from_url("memory:|full:"), mode="w-")
        assert len(ClosingStore.closes) == 1

    async def test_make_store_residual_path(self, recording: type[RecordingAdapter]) -> None:
        with pytest.raises(URLPipelineError, match="path inside a store"):
            await make_store(URLPipeline.from_url("memory:|rec:sub"))
        assert len(ClosingStore.closes) == 1

    def test_success_does_not_close(self, recording: type[RecordingAdapter]) -> None:
        zarr.open_group(URLPipeline.from_url("memory:|rec:"), mode="w")
        assert ClosingStore.closes == []


class TestConsolidateMetadata:
    def test_honors_format_segment(self, tmp_path: Path) -> None:
        root = tmp_path.as_posix()
        zarr.create_group(tmp_path, zarr_format=3).create_array("x", shape=(2,), dtype="i4")
        # a zarr2: pipeline over v3 data finds no v2 group
        with pytest.raises(FileNotFoundError):
            zarr.consolidate_metadata(URLPipeline.from_url(f"file:{root}|zarr2:"))
        with pytest.warns(ZarrUserWarning, match="Consolidated metadata"):
            group = zarr.consolidate_metadata(URLPipeline.from_url(f"file:{root}|zarr3:"))
        assert group.metadata.consolidated_metadata is not None

    def test_conflicting_format_raises(self, tmp_path: Path) -> None:
        zarr.create_group(tmp_path, zarr_format=2)
        with pytest.raises(ValueError, match="conflicts with"):
            zarr.consolidate_metadata(
                URLPipeline.from_url(f"file:{tmp_path.as_posix()}|zarr2:"), zarr_format=3
            )


class TestFormatSegments:
    @pytest.mark.parametrize("url", ["zarr3:", "zarr2:/data|zarr3:", "zarr:x|rec:", "ZARR3:"])
    def test_format_scheme_cannot_be_root(self, url: str) -> None:
        with pytest.raises(URLPipelineError, match="cannot be the root"):
            URLPipeline.from_url(url)

    @pytest.mark.parametrize("body", ["../x", "a/../../x", "./x", "a/./b"])
    def test_node_path_dot_segments_rejected_at_parse(self, body: str) -> None:
        with pytest.raises(URLPipelineError, match="node path of a format segment"):
            URLPipeline.from_url(f"memory:|zarr3:{body}")

    def test_node_path_traversal_creates_nothing(self, tmp_path: Path) -> None:
        target = tmp_path / "newdir"
        with pytest.raises(URLPipelineError):
            zarr.open_group(URLPipeline.from_url(f"file:{target.as_posix()}|zarr3:../x"), mode="w")
        assert not target.exists()

    def test_zarr_segment_detects_format(self, tmp_path: Path) -> None:
        root = tmp_path.as_posix()
        zarr.create_group(URLPipeline.from_url(f"file:{root}|zarr2:g"), zarr_format=2)
        group = zarr.open_group(URLPipeline.from_url(f"file:{root}|zarr:g"), mode="r")
        assert group.metadata.zarr_format == 2
        assert group.path == "g"
        # no format is selected, so a default zarr_format=3 does not conflict
        zarr.create_array(
            URLPipeline.from_url(f"file:{root}|zarr:"), name="a", shape=(2,), dtype="i4"
        )
        assert (
            zarr.open_array(URLPipeline.from_url(f"file:{root}|zarr:a")).metadata.zarr_format == 3
        )

    def test_zarr_segment_must_be_last(self) -> None:
        with pytest.raises(URLPipelineError, match="must be the last segment"):
            URLPipeline.from_url("memory:|zarr:|rec:")


class TestFileRoots:
    @pytest.mark.parametrize(
        "url", ["file:////data", "file:////nonexistent/data", "file://localhost//data"]
    )
    def test_double_slash_path_rejected(
        self, url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(URLPipelineError, match="must not start with '//'"):
            zarr.open_group(URLPipeline.from_url(url), mode="w")
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("suffix", ["/../x", "/./x", "/a/../../x"])
    def test_dot_segments_rejected(self, suffix: str, tmp_path: Path) -> None:
        with pytest.raises(URLPipelineError, match=r"'\.' or '\.\.'"):
            zarr.open_group(URLPipeline.from_url(f"file:{tmp_path.as_posix()}{suffix}"), mode="w")

    def test_root_opens_absolute_local_path(self, tmp_path: Path) -> None:
        group = zarr.open_group(URLPipeline.from_url(f"file:{tmp_path.as_posix()}/g"), mode="w")
        group.attrs["x"] = 1
        assert (tmp_path / "g" / "zarr.json").exists()

    def test_storage_options_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="'storage_options' was provided but unused"):
            zarr.open_group(
                URLPipeline.from_url(f"file:{tmp_path.as_posix()}"),
                mode="w",
                storage_options={"anon": True},
            )

    def test_authority_is_case_insensitive(self, tmp_path: Path) -> None:
        # spell the path URL-style: on Windows C:/... gains a leading slash
        posix = tmp_path.as_posix()
        url_path = posix if posix.startswith("/") else f"/{posix}"
        zarr.open_group(URLPipeline.from_url(f"file://LOCALHOST{url_path}"), mode="w")
        assert (tmp_path / "zarr.json").exists()


class TestMemoryRoot:
    async def test_single_store_body_is_a_path(self) -> None:
        """The body of a `memory:` root is a path, so it can be split across adapter segments."""

        class PathAdapter(URLPipelineAdapter):
            @classmethod
            async def open_pipeline_segment(
                cls, segment: PipelineSegment, context: PipelineContext
            ) -> StorePath:
                preceding = await context.resolve_preceding()
                return StorePath(preceding.store, f"{preceding.path}/{segment.body}".strip("/"))

        register_url_adapter("sub", PathAdapter)
        # keep the writer alive: managed memory stores drop their data once unreferenced
        writer = zarr.open_group(URLPipeline.from_url("memory:mem-one/a/b|sub:"), mode="w")
        writer.attrs["k"] = 1
        for url in ("memory:mem-one|sub:a/b", "memory:|sub:mem-one/a/b", "memory://mem-one/a/b"):
            assert zarr.open_group(URLPipeline.from_url(url), mode="r").attrs["k"] == 1

    @pytest.mark.parametrize("body", ["../x", "a/../b", "./a"])
    async def test_dot_segments_rejected(self, body: str) -> None:
        with pytest.raises(URLPipelineError, match=r"'\.' or '\.\.'"):
            await make_store_path(URLPipeline.from_url(f"memory:{body}"))

    @pytest.mark.parametrize("storage_options", [{}, {"anon": True}])
    async def test_storage_options_rejected(self, storage_options: dict[str, object]) -> None:
        with pytest.raises(TypeError, match="'storage_options' was provided but unused"):
            await make_store_path(URLPipeline.from_url("memory:x"), storage_options=storage_options)


class TestRootSchemeCase:
    def test_uppercase_native_root(self, tmp_path: Path) -> None:
        group = zarr.open_group(URLPipeline.from_url(f"FILE:{tmp_path.as_posix()}"), mode="w")
        assert isinstance(group, zarr.Group)

    async def test_uppercase_fsspec_root(self) -> None:
        """fsspec matches protocols case-sensitively, so the root gets the lowercased scheme."""
        pytest.importorskip("fsspec")
        pytest.importorskip("aiohttp")
        from zarr.storage import FsspecStore

        store_path = await make_store_path(URLPipeline.from_url("HTTPS://example.com/d"))
        assert isinstance(store_path.store, FsspecStore)
        assert store_path.store.path == "https://example.com/d"


class TestRegisterURLAdapter:
    @pytest.mark.parametrize("scheme", ["", "a b", "1abc", "a_b", "zip:"])
    def test_invalid_scheme_rejected(self, scheme: str) -> None:
        with pytest.raises(ValueError, match="not a valid URL scheme"):
            register_url_adapter(scheme, RecordingAdapter)

    @pytest.mark.parametrize("scheme", ["file", "memory", "zarr", "zarr2", "ZARR3"])
    def test_reserved_scheme_rejected(self, scheme: str) -> None:
        with pytest.raises(ValueError, match="handled by zarr-python itself"):
            register_url_adapter(scheme, RecordingAdapter)

    def test_vendor_scheme_accepted(self) -> None:
        register_url_adapter("Vendor.My-Scheme+x", RecordingAdapter)
        assert zarr.registry._get_url_adapter("vendor.my-scheme+x") is RecordingAdapter

    @pytest.mark.parametrize("scheme", ["foo\n", "\u212aelvin"])
    def test_register_rejects_scheme_tricks(self, scheme: str) -> None:
        with pytest.raises(ValueError, match="not a valid URL scheme"):
            register_url_adapter(scheme, RecordingAdapter)


class TestSegmentValidation:
    @pytest.mark.parametrize(
        ("fields", "error"),
        [
            ((1, "b", None), TypeError),
            (("s", 2, None), TypeError),
            (("s", "b", 3), TypeError),
            (("S3", "b", None), URLPipelineError),
            (("", "b", None), URLPipelineError),
            (("a b", "b", None), URLPipelineError),
            # a hand-built segment obeys the parser's rules, so its url parses back to it
            (("file", "/tmp/a|evil:", None), URLPipelineError),
            (("s3", "//b|zarr3:", None), URLPipelineError),
            (("s3", "//b", "x|evil:"), URLPipelineError),
            (("s3", "//b?q", None), URLPipelineError),
            (("s3", "//b#f", None), URLPipelineError),
            (("s3", "//b", "q#f"), URLPipelineError),
            (("s3", "//b\n", None), URLPipelineError),
            (("s3", "//b ", None), URLPipelineError),
            (("s3", "//b", " "), URLPipelineError),
        ],
    )
    def test_invalid_fields(self, fields: tuple[object, ...], error: type[Exception]) -> None:
        scheme, body, query = fields
        with pytest.raises(error):
            PipelineSegment(scheme=scheme, body=body, query=query)  # type: ignore[arg-type]

    def test_fields_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            PipelineSegment("s3", "//b", None)  # type: ignore[call-arg]

    def test_hand_built_pipe_in_body_cannot_reach_an_adapter(self) -> None:
        with pytest.raises(URLPipelineError, match="cannot contain '|'"):
            URLPipeline((PipelineSegment(scheme="file", body="/tmp/a|evil:", query=None),))


class TestMessages:
    def test_repr_is_the_url(self) -> None:
        assert repr(URLPipeline.from_url("memory:|zarr3:a")) == "URLPipeline('memory:|zarr3:a')"

    def test_invalid_root_scheme_message(self) -> None:
        with pytest.raises(URLPipelineError, match="invalid URL scheme 'a_b'"):
            URLPipeline.from_url("a_b://x")

    @pytest.mark.parametrize("url", ["/a|b", "file:/a#b", " s3://b"])
    def test_no_percent_encoding_advice(self, url: str) -> None:
        with pytest.raises(URLPipelineError) as info:
            URLPipeline.from_url(url)
        assert "%7C" not in str(info.value)
        assert "%23" not in str(info.value)
        assert "%20" not in str(info.value)

    @pytest.mark.skipif(sys.platform == "win32", reason="'|' is not allowed in Windows file names")
    def test_group_not_found_message_is_readable(self, tmp_path: Path) -> None:
        with pytest.raises(GroupNotFoundError) as info:
            zarr.open_group(URLPipeline.from_url(f"file:{tmp_path.as_posix()}|zarr3:"), mode="r")
        assert "PipelineSegment(" not in str(info.value)

    def test_no_adapter_error_does_not_list_installed_schemes(
        self, recording: type[RecordingAdapter]
    ) -> None:
        with pytest.raises(URLPipelineError) as info:
            zarr.core.sync.sync(make_store_path(URLPipeline.from_url("memory:|nope:")))
        assert "rec" not in str(info.value)


class TestSecretsAreRedacted:
    URL = "https://user:pw@h/x?token=SECRET|rec:|zarr2:"

    def test_reprs(self) -> None:
        pipeline = URLPipeline.from_url(self.URL)
        context = zarr.storage.PipelineContext(
            preceding=pipeline.segments[:1], mode="r", storage_options={"secret": "SECRET"}
        )
        for text in (repr(pipeline), repr(pipeline.segments[0]), repr(context)):
            assert "SECRET" not in text
            assert "pw" not in text
        assert str(pipeline) == self.URL  # str() stays lossless

    @pytest.mark.parametrize(
        "url",
        [
            "https://h/x?token=SECRET| zip:",
            "https://h/x?token=SECRET||zip:",
            "https://h/x?token=SECRET#frag",
            "https://h/x?token=SECRET|zarr3:|zip:",
            "/x?token=SECRET",
            "a_b://x?token=SECRET",
        ],
    )
    def test_parse_errors(self, url: str) -> None:
        with pytest.raises(URLPipelineError) as info:
            URLPipeline.from_url(url)
        assert "SECRET" not in str(info.value)

    async def test_resolution_errors(self, recording: type[RecordingAdapter]) -> None:
        with pytest.raises(ValueError, match="conflicts with") as conflict:
            zarr.open_group(URLPipeline.from_url(self.URL), mode="w", zarr_format=3)
        with pytest.raises(URLPipelineError) as residual:
            await make_store(URLPipeline.from_url("memory:?token=SECRET|rec:sub"))
        with pytest.raises(URLPipelineError) as root:
            await make_store_path(URLPipeline.from_url("vendor.x://[a]/p?token=SECRET"))
        for info in (conflict, residual, root):
            assert "SECRET" not in str(info.value)


class TestEntryPointNames:
    @pytest.mark.parametrize("name", ["file", "Memory", "zarr3", "a b", "x\n"])
    def test_unusable_entry_point_names_ignored(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from importlib.metadata import EntryPoint, EntryPoints

        fake = EntryPoints(
            [EntryPoint(name=name, value="nowhere:Adapter", group="zarr.url_adapters")]
        )
        monkeypatch.setattr(zarr.registry, "get_entry_points", lambda: fake)
        zarr.registry._url_adapter_registry.lazy_load_list.clear()
        with pytest.warns(ZarrUserWarning, match="ignoring the 'zarr.url_adapters' entry point"):
            zarr.registry._collect_entrypoints()
        assert zarr.registry._url_adapter_registry.lazy_load_list == []


class TestLimits:
    def test_hand_built_pipeline_segment_limit(self) -> None:
        segment = PipelineSegment(scheme="w", body="", query=None)
        root = PipelineSegment(scheme="memory", body="", query=None)
        URLPipeline((root, *[segment] * 31))
        with pytest.raises(URLPipelineError, match="1 to 32 segments"):
            URLPipeline((root, *[segment] * 32))


class TestWindowsFileRoots:
    """Windows path rules, checked on any platform by patching `sys.platform`."""

    @pytest.mark.parametrize(
        "url",
        [
            r"file:/\attacker\share\data",
            r"file:///\attacker\share",
            r"file://localhost/\attacker\share",
            r"file:\\attacker\share",
        ],
    )
    async def test_unc_paths_rejected(self, url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        with pytest.raises(URLPipelineError, match="authority|'//'|absolute local path"):
            await make_store_path(URLPipeline.from_url(url), mode="r")

    @pytest.mark.parametrize(
        "url", [r"file:C:x", r"file:/C:..\x", "file:C:/a/../b", "file:/C:/a:stream"]
    )
    async def test_drive_relative_and_dot_paths_rejected(
        self, url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        with pytest.raises(URLPipelineError, match=r"absolute local path|'\.' or '\.\.'|':'"):
            await make_store_path(URLPipeline.from_url(url), mode="r")
