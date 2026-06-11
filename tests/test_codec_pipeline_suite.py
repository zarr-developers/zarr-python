"""Shared codec-pipeline behavior suite, run against EVERY codec pipeline.

The defining property of a codec pipeline is that the array semantics it
produces must be identical no matter which pipeline is configured. To make
"one pipeline diverges from the others" structurally hard to ship, every
pipeline-agnostic behavior test lives as a method on ``CodecPipelineTests`` and
is instantiated once per pipeline (``TestBatchedPipeline`` / ``TestFusedPipeline``).

Each test also runs over a *store axis* that exercises both code paths the
synchronous pipelines branch on:

* ``sync``  -> ``MemoryStore`` (supports ``get_sync``/``set_sync``: fast path)
* ``async`` -> ``LatencyStore(MemoryStore())`` (NOT sync-capable: async fallback)

The async axis is deliberate: a regression that only affects the async fallback
of the default pipeline (e.g. a codec-spec-evolution bug that surfaces only on
remote stores) is invisible if every test runs on MemoryStore. Running the same
battery over a non-sync store closes that gap.

Pipeline-specific tests (construction, ``from_codecs``, the byte-range write
fast path, etc.) stay in their own modules; only behavior that ALL pipelines
must share belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numcodecs
import numpy as np
import pytest

import zarr
from zarr.codecs import BytesCodec, GzipCodec, ShardingCodec, TransposeCodec
from zarr.core.config import config as zarr_config
from zarr.errors import ChunkNotFoundError
from zarr.storage import MemoryStore
from zarr.testing.store import LatencyStore

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr.abc.store import Store
    from zarr.codecs.sharding import SubchunkWriteOrder


# --- store axis: a sync store and a non-sync (async-fallback) store ----------

STORE_KINDS = ["sync", "async"]


def _make_store(kind: str) -> Store:
    if kind == "sync":
        # MemoryStore supports get_sync/set_sync -> synchronous fast path.
        return MemoryStore()
    if kind == "async":
        # LatencyStore is NOT SupportsGetSync/SupportsSetSync, so a synchronous
        # pipeline must fall back to its async path. Zero latency keeps it fast.
        return LatencyStore(MemoryStore(), get_latency=0.0, set_latency=0.0)
    raise AssertionError(kind)


# --- scenario model ----------------------------------------------------------
#
# Most pipeline behavior tests have one shape:
#   create an array, apply some writes, (optionally) assert which chunk keys
#   exist, then assert reads come back correct. A Scenario captures exactly those
#   variables so one parametrized test covers them all. Correctness is checked
#   against a numpy reference array that the scenario mutates in lock-step with
#   the zarr array, so cases don't hand-maintain expected values.


@dataclass(frozen=True)
class Scenario:
    id: str
    array_kwargs: dict[str, Any]
    # (selection, value) writes applied in order. value may be a scalar or array.
    writes: tuple[tuple[Any, Any], ...] = ()
    # selections to read back and check against the reference. () means "read all".
    reads: tuple[Any, ...] = (slice(None),)
    # substrings of chunk keys that must be present / absent after the writes.
    # Only checked on the sync store (key layout is identical across stores, but
    # we keep it to one axis to avoid asserting store internals twice).
    keys_present: tuple[str, ...] = ()
    keys_absent: tuple[str, ...] = ()

    def reference(self) -> np.ndarray:
        """The numpy array the scenario's writes should produce, starting from
        the array's fill value."""
        kw = self.array_kwargs
        shape = kw["shape"]
        dtype = np.dtype(kw["dtype"])
        fill = kw.get("fill_value", 0)
        ref = np.full(shape, fill, dtype=dtype)
        for sel, value in self.writes:
            ref[sel] = value
        return ref


def _val(n: int, dtype: str, offset: int = 1) -> np.ndarray:
    return np.arange(offset, offset + n, dtype=dtype)


# Common dtype/chunk presets reused below.
_F64 = {"dtype": "float64", "fill_value": 0.0}
_I32 = {"dtype": "int32", "fill_value": -1}

SCENARIOS: tuple[Scenario, ...] = (
    # --- full-array roundtrips across layouts/codecs ------------------------
    Scenario(
        "1d-unsharded-roundtrip",
        {"shape": (100,), "chunks": (10,), "shards": None, "compressors": None, **_F64},
        writes=((slice(None), _val(100, "float64")),),
    ),
    Scenario(
        "1d-sharded-roundtrip",
        {"shape": (100,), "chunks": (10,), "shards": (100,), "compressors": None, **_F64},
        writes=((slice(None), _val(100, "float64")),),
    ),
    Scenario(
        "1d-multi-chunk-shard-roundtrip",
        {"shape": (100,), "chunks": (10,), "shards": (50,), "compressors": None, **_F64},
        writes=((slice(None), _val(100, "float64")),),
    ),
    Scenario(
        "2d-unsharded-roundtrip",
        {"shape": (10, 20), "chunks": (5, 10), "shards": None, "compressors": None, **_I32},
        writes=((slice(None), np.arange(200, dtype="int32").reshape(10, 20)),),
    ),
    Scenario(
        "2d-sharded-roundtrip",
        {"shape": (20, 20), "chunks": (5, 5), "shards": (10, 10), "compressors": None, **_I32},
        writes=((slice(None), np.arange(400, dtype="int32").reshape(20, 20)),),
    ),
    Scenario(
        "1d-gzip-roundtrip",
        {
            "shape": (100,),
            "chunks": (10,),
            "shards": None,
            "compressors": {"name": "gzip", "configuration": {"level": 1}},
            **_F64,
        },
        writes=((slice(None), _val(100, "float64")),),
    ),
    Scenario(
        "1d-zstd-roundtrip",
        {
            "shape": (100,),
            "chunks": (10,),
            "shards": None,
            "compressors": {"name": "zstd", "configuration": {"level": 1}},
            **_F64,
        },
        writes=((slice(None), _val(100, "float64")),),
    ),
    Scenario(
        "1d-float32-roundtrip",
        {
            "shape": (50,),
            "chunks": (10,),
            "shards": None,
            "compressors": None,
            "dtype": "float32",
            "fill_value": 0.0,
        },
        writes=((slice(None), _val(50, "float32")),),
    ),
    # zarr v2 goes through the V2Codec wrapper (filters + compressor), a
    # different codec path than the v3 AA/AB/BB chain — and a different sync
    # implementation under FusedCodecPipeline. Without these scenarios, v2 was
    # only exercised implicitly via whichever pipeline is the global default.
    Scenario(
        "v2-roundtrip",
        {
            "shape": (100,),
            "chunks": (10,),
            "shards": None,
            "compressors": None,
            "zarr_format": 2,
            **_F64,
        },
        writes=((slice(None), _val(100, "float64")),),
    ),
    Scenario(
        "v2-gzip-roundtrip",
        {
            "shape": (100,),
            "chunks": (10,),
            "shards": None,
            "compressors": numcodecs.GZip(level=1),
            "zarr_format": 2,
            **_F64,
        },
        writes=((slice(None), _val(100, "float64")),),
    ),
    # v2 filters are the other half of the V2Codec wrapper (numcodecs
    # array->array filters, a distinct branch from the compressor in
    # _encode_sync/_decode_sync).
    Scenario(
        "v2-filter-gzip-roundtrip",
        {
            "shape": (100,),
            "chunks": (10,),
            "shards": None,
            "filters": numcodecs.Delta(dtype="float64"),
            "compressors": numcodecs.GZip(level=1),
            "zarr_format": 2,
            **_F64,
        },
        writes=((slice(None), _val(100, "float64")),),
    ),
    # --- read unwritten chunks -> fill value --------------------------------
    Scenario(
        "missing-chunks-fill",
        {
            "shape": (100,),
            "chunks": (10,),
            "shards": None,
            "compressors": None,
            "dtype": "float64",
            "fill_value": -7.0,
        },
        writes=(),
    ),
    Scenario(
        "missing-chunks-fill-sharded",
        {
            "shape": (100,),
            "chunks": (10,),
            "shards": (100,),
            "compressors": None,
            "dtype": "float64",
            "fill_value": -7.0,
        },
        writes=(),
    ),
    # --- partial write, varied read selections ------------------------------
    Scenario(
        "partial-write-full-read",
        {"shape": (100,), "chunks": (10,), "shards": None, "compressors": None, **_F64},
        writes=((slice(5, 15), _val(10, "float64")),),
        reads=(slice(None),),
    ),
    Scenario(
        "full-write-strided-read",
        {"shape": (100,), "chunks": (10,), "shards": None, "compressors": None, **_F64},
        writes=((slice(None), _val(100, "float64")),),
        reads=(np.s_[::3], np.s_[10:20]),
    ),
    Scenario(
        "partial-write-partial-read-sharded",
        {"shape": (100,), "chunks": (10,), "shards": (100,), "compressors": None, **_F64},
        writes=((slice(20, 70), _val(50, "float64")),),
        reads=(np.s_[30:60], slice(None)),
    ),
    # scalar single-element reads from a sharded array hit the sharding codec's
    # partial-decode path (_decode_partial_single), distinct from slice reads.
    Scenario(
        "sharded-scalar-reads-1d",
        {"shape": (100,), "chunks": (10,), "shards": (50,), "compressors": None, **_F64},
        writes=((slice(None), _val(100, "float64")),),
        reads=(np.s_[0], np.s_[50], np.s_[99], np.s_[::3]),
    ),
    Scenario(
        "sharded-scalar-reads-2d",
        {"shape": (20, 20), "chunks": (5, 5), "shards": (10, 10), "compressors": None, **_I32},
        writes=((slice(None), np.arange(400, dtype="int32").reshape(20, 20)),),
        reads=(np.s_[0, 0], np.s_[10, 10], np.s_[19, 19]),
    ),
    # --- spec-changing codec (transpose): the async-spec-evolution guard ----
    Scenario(
        "transpose",
        {
            "shape": (8, 12),
            "chunks": (2, 4),
            "shards": None,
            "filters": [TransposeCodec(order=(1, 0))],
            "serializer": BytesCodec(),
            **_I32,
        },
        writes=((slice(None), np.arange(96, dtype="int32").reshape(8, 12)),),
        reads=(slice(None), np.s_[1:7, 2:10]),
    ),
    Scenario(
        "transpose-gzip",
        {
            "shape": (8, 12),
            "chunks": (2, 4),
            "shards": None,
            "filters": [TransposeCodec(order=(1, 0))],
            "serializer": BytesCodec(),
            "compressors": GzipCodec(level=1),
            **_I32,
        },
        writes=((slice(None), np.arange(96, dtype="int32").reshape(8, 12)),),
        reads=(slice(None), np.s_[1:7, 2:10]),
    ),
    # --- nested sharding ----------------------------------------------------
    Scenario(
        "nested-sharding",
        {
            "shape": (20, 20),
            "chunks": (10, 10),
            "shards": None,
            "compressors": None,
            **_I32,
            "fill_value": 0,
            "serializer": ShardingCodec(
                chunk_shape=(10, 10), codecs=[ShardingCodec(chunk_shape=(5, 5))]
            ),
        },
        writes=((slice(None), np.arange(400, dtype="int32").reshape(20, 20)),),
    ),
    # --- partial overwrite of an existing shard (merge) ---------------------
    Scenario(
        "partial-shard-overwrite",
        {
            "shape": (40,),
            "chunks": (4,),
            "shards": (40,),
            "compressors": None,
            **_I32,
            "config": {"write_empty_chunks": True},
        },
        writes=(
            (slice(None), np.arange(40, dtype="int32")),
            (slice(7, 18), _val(11, "int32", 700)),
        ),
    ),
    # --- write_empty_chunks: storage-key presence/absence -------------------
    Scenario(
        "write-empty-false-omits-fill-chunk",
        {
            "shape": (20,),
            "chunks": (10,),
            "shards": None,
            "compressors": None,
            **_F64,
            "config": {"write_empty_chunks": False},
        },
        writes=((slice(0, 10), _val(10, "float64")), (slice(10, 20), np.zeros(10, "float64"))),
        keys_present=("c/0",),
        keys_absent=("c/1",),
    ),
    Scenario(
        "write-empty-true-persists-fill-chunk",
        {
            "shape": (20,),
            "chunks": (10,),
            "shards": None,
            "compressors": None,
            **_F64,
            "config": {"write_empty_chunks": True},
        },
        writes=((slice(None), np.zeros(20, "float64")),),
        keys_present=("c/0", "c/1"),
    ),
    # default config (no explicit write_empty_chunks) must still skip fill chunks
    Scenario(
        "default-config-omits-fill-chunk",
        {"shape": (20,), "chunks": (10,), "shards": None, "compressors": None, **_F64},
        writes=((slice(10, 20), np.zeros(10, "float64")),),
        keys_absent=("c/1",),
    ),
)


class CodecPipelineTests:
    """Behavior every codec pipeline must satisfy, on sync and async stores.

    Subclasses set ``pipeline_path`` to the fully-qualified pipeline class.
    """

    pipeline_path: str

    @pytest.fixture(autouse=True)
    def _use_pipeline(self) -> Iterator[None]:
        with zarr_config.set({"codec_pipeline.path": self.pipeline_path}):
            yield

    @pytest.fixture(params=STORE_KINDS)
    def store(self, request: pytest.FixtureRequest) -> Store:
        return _make_store(request.param)

    @staticmethod
    def _chunk_keys(store: Store) -> set[str]:
        """All non-metadata keys currently in the store (v3 and v2 metadata)."""
        import asyncio

        def _is_metadata(key: str) -> bool:
            tail = key.rsplit("/", 1)[-1]
            return tail in ("zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata")

        async def _list() -> set[str]:
            return {k async for k in store.list() if not _is_metadata(k)}

        return asyncio.run(_list())

    # -- the common shape: create -> write -> [assert keys] -> assert reads ----

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
    def test_scenario(self, store: Store, scenario: Scenario) -> None:
        """Create an array, apply the scenario's writes, optionally assert which
        chunk keys exist, then assert each read selection matches a numpy
        reference. Run against every pipeline (subclass) and store kind (fixture).
        """
        arr = zarr.create_array(store=store, **scenario.array_kwargs)
        for sel, value in scenario.writes:
            arr[sel] = value

        ref = scenario.reference()
        for sel in scenario.reads:
            np.testing.assert_array_equal(
                arr[sel], ref[sel], err_msg=f"{scenario.id}: read {sel!r} mismatch"
            )

        if scenario.keys_present or scenario.keys_absent:
            keys = self._chunk_keys(store)
            for present in scenario.keys_present:
                assert any(present in k for k in keys), (present, keys)
            for absent in scenario.keys_absent:
                assert not any(absent in k for k in keys), (absent, keys)

    # -- outliers that don't fit the create/write/read scenario shape ----------

    def test_read_missing_chunks_false_raises(self, store: Store) -> None:
        """read_missing_chunks=False makes reading an unwritten chunk an error,
        not a fill — a different assertion (raises) than the scenario shape."""
        arr = zarr.create_array(
            store=store,
            shape=(20,),
            dtype="float64",
            chunks=(10,),
            shards=None,
            compressors=None,
            fill_value=0.0,
            config={"read_missing_chunks": False},
        )
        with pytest.raises(ChunkNotFoundError):
            arr[:]

    def test_read_missing_chunks_false_sharded_semantics(self, store: Store) -> None:
        """read_missing_chunks=False is a STORE-KEY-level promise on sharded arrays.

        The config exists to help consumers distinguish a transport error from a
        truly missing chunk. That distinction applies to store keys: a missing
        SHARD key raises ChunkNotFoundError. It does not cleanly apply to inner
        subchunks of a shard that was fetched successfully — there is no
        transport ambiguity there, the shard index simply records the subchunk
        as absent — so missing inner subchunks fill with the fill value rather
        than raising. This pins that asymmetry as intentional.
        """
        arr = zarr.create_array(
            store=store,
            shape=(100,),
            dtype="float64",
            chunks=(10,),
            shards=(50,),
            compressors=None,
            fill_value=-1.0,
            config={"read_missing_chunks": False},
        )
        # No shard key exists yet: reading is a missing-store-key error.
        with pytest.raises(ChunkNotFoundError):
            arr[:]

        # Write one inner chunk of the first shard. The shard key now exists,
        # but most inner subchunks are absent from its index.
        arr[20:30] = np.arange(10, dtype="float64")

        # Reading across written + absent inner subchunks of the EXISTING shard
        # fills rather than raises.
        out = arr[15:35]
        expected = np.full(20, -1.0)
        expected[5:15] = np.arange(10, dtype="float64")
        np.testing.assert_array_equal(out, expected)

        # Both halves of the asymmetry in ONE read against the SAME partially
        # written array: shard 0 exists (absent subchunks fill), shard 1 has no
        # store key (raises) — pins that the raise still fires once some shard
        # exists, and not only on a fully-empty array.
        with pytest.raises(ChunkNotFoundError):
            arr[:]
        with pytest.raises(ChunkNotFoundError):
            arr[45:55]  # spans the existing and the missing shard

    @pytest.mark.parametrize("subchunk_write_order", ["morton", "lexicographic", "colexicographic"])
    def test_partial_write_after_reopen_is_correct(
        self, store: Store, subchunk_write_order: SubchunkWriteOrder
    ) -> None:
        """Has an extra step the scenario shape lacks — a REOPEN between writes.

        Reopening a sharded array and partially overwriting it must read back
        correctly regardless of the original subchunk_write_order. subchunk_write_
        order is intentionally NOT recoverable on reopen, so chunk locations on a
        write to an existing shard must come from the STORED shard index, not the
        (now-default) live order. A non-square inner grid makes the orders
        physically distinct, so a wrong offset would corrupt data and fail here.
        """
        shape, shard, inner = (6, 4), (6, 4), (2, 2)
        arr = zarr.create_array(
            store=store,
            shape=shape,
            dtype="int32",
            chunks=shard,
            fill_value=-1,
            compressors=None,
            config={"write_empty_chunks": True},
            serializer=ShardingCodec(
                chunk_shape=inner, codecs=[BytesCodec()], subchunk_write_order=subchunk_write_order
            ),
        )
        ref = np.arange(24, dtype="int32").reshape(shape)
        arr[:] = ref

        reopened = zarr.open_array(store=store, mode="r+")
        reopened[1:5, 0:3] = 777  # partial overwrite into the existing shard
        ref[1:5, 0:3] = 777
        np.testing.assert_array_equal(reopened[:], ref)

    def test_empty_shard_deleted_after_overwrite_to_fill(self, store: Store) -> None:
        """A shard written with real data and then fully overwritten back to the
        fill value must have its store key deleted, not left as a stale blob.

        This has a mid-sequence key assertion (present after write 1, absent
        after write 2) that the create/write/read scenario shape can't express.
        """
        arr = zarr.create_array(
            store=store,
            shape=(16,),
            chunks=(4,),
            shards=(8,),
            dtype="float64",
            compressors=None,
            fill_value=0.0,
        )
        arr[0:8] = np.arange(8, dtype="float64") + 1
        assert any("c/0" in k for k in self._chunk_keys(store))
        arr[0:8] = 0.0
        assert not any("c/0" in k for k in self._chunk_keys(store)), (
            "shard should be deleted when fully overwritten to fill value"
        )

    def test_read_write_methods_do_not_branch_on_sharding_codec_type(self) -> None:
        """Pipeline read/write must dispatch on supports_partial_encode/decode,
        not isinstance(ShardingCodec) — a static guard against type-branching.

        Scoped to this pipeline's own read/write methods (other helpers, e.g.
        metadata validation, may legitimately isinstance-check ShardingCodec).
        """
        import inspect
        import re

        from zarr.registry import get_pipeline_class

        # The autouse _use_pipeline fixture has set codec_pipeline.path to this
        # subclass's pipeline; resolve the class it points at and guard that.
        # reload_config=False so the fixture's config override is honored
        # (reload_config=True re-reads the base config, ignoring the override).
        cls = get_pipeline_class(reload_config=False)

        pattern = re.compile(r"isinstance\s*\([^)]*ShardingCodec[^)]*\)")
        for method_name in ("read", "write", "read_sync", "write_sync"):
            method = getattr(cls, method_name, None)
            if method is None:
                continue
            matches = pattern.findall(inspect.getsource(method))
            assert not matches, (
                f"{cls.__name__}.{method_name} contains an isinstance check on "
                f"ShardingCodec; use supports_partial_encode/decode instead. "
                f"Matches: {matches}"
            )


class TestBatchedPipeline(CodecPipelineTests):
    pipeline_path = "zarr.core.codec_pipeline.BatchedCodecPipeline"


class TestFusedPipeline(CodecPipelineTests):
    pipeline_path = "zarr.core.codec_pipeline.FusedCodecPipeline"
