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

from typing import TYPE_CHECKING, Any

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


# --- shared array configs ----------------------------------------------------

ARRAY_CONFIGS = [
    pytest.param(
        {"shape": (100,), "dtype": "float64", "chunks": (10,), "shards": None, "compressors": None},
        id="1d-unsharded",
    ),
    pytest.param(
        {
            "shape": (100,),
            "dtype": "float64",
            "chunks": (10,),
            "shards": (100,),
            "compressors": None,
        },
        id="1d-sharded",
    ),
    pytest.param(
        {
            "shape": (100,),
            "dtype": "float64",
            "chunks": (10,),
            "shards": (50,),
            "compressors": None,
        },
        id="1d-multi-chunk-shard",
    ),
    pytest.param(
        {
            "shape": (10, 20),
            "dtype": "int32",
            "chunks": (5, 10),
            "shards": None,
            "compressors": None,
        },
        id="2d-unsharded",
    ),
    pytest.param(
        {
            "shape": (20, 20),
            "dtype": "int32",
            "chunks": (5, 5),
            "shards": (10, 10),
            "compressors": None,
        },
        id="2d-sharded",
    ),
    pytest.param(
        {
            "shape": (100,),
            "dtype": "float64",
            "chunks": (10,),
            "shards": None,
            "compressors": {"name": "gzip", "configuration": {"level": 1}},
        },
        id="1d-gzip",
    ),
]


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

    # -- roundtrip / fill-value ------------------------------------------------

    @pytest.mark.parametrize("arr_kwargs", ARRAY_CONFIGS)
    def test_roundtrip(self, store: Store, arr_kwargs: dict[str, Any]) -> None:
        """Data survives a full write/read roundtrip."""
        arr = zarr.create_array(store=store, fill_value=0, **arr_kwargs)
        data = np.arange(int(np.prod(arr.shape)), dtype=arr.dtype).reshape(arr.shape)
        arr[:] = data
        np.testing.assert_array_equal(arr[:], data)

    @pytest.mark.parametrize("arr_kwargs", ARRAY_CONFIGS)
    def test_missing_chunks_fill_value(self, store: Store, arr_kwargs: dict[str, Any]) -> None:
        """Reading unwritten chunks returns the fill value."""
        arr = zarr.create_array(store=store, fill_value=-1, **arr_kwargs)
        np.testing.assert_array_equal(arr[:], np.full(arr.shape, -1, dtype=arr.dtype))

    # -- write/read selection combinations ------------------------------------

    @pytest.mark.parametrize("shards", [None, (100,)], ids=["unsharded", "sharded"])
    @pytest.mark.parametrize(
        ("write_sel", "read_sel"),
        [
            pytest.param(slice(None), np.s_[:], id="full-write-full-read"),
            pytest.param(slice(5, 15), np.s_[:], id="partial-write-full-read"),
            pytest.param(slice(None), np.s_[::3], id="full-write-strided-read"),
            pytest.param(slice(None), np.s_[10:20], id="full-write-slice-read"),
            pytest.param(slice(20, 70), np.s_[30:60], id="partial-write-partial-read"),
        ],
    )
    def test_write_then_read(
        self, store: Store, shards: tuple[int, ...] | None, write_sel: slice, read_sel: Any
    ) -> None:
        arr = zarr.create_array(
            store=store,
            shape=(100,),
            dtype="float64",
            chunks=(10,),
            shards=shards,
            compressors=None,
            fill_value=0.0,
        )
        full = np.zeros(100, dtype="float64")
        write_data = np.arange(len(full[write_sel]), dtype="float64") + 1
        full[write_sel] = write_data
        arr[write_sel] = write_data
        np.testing.assert_array_equal(arr[read_sel], full[read_sel])

    # -- spec-changing codecs (regression guard for async-path spec evolution) -

    @pytest.mark.parametrize(
        "arr_kwargs",
        [
            pytest.param(
                {"filters": [TransposeCodec(order=(1, 0))], "serializer": BytesCodec()},
                id="transpose",
            ),
            pytest.param(
                {
                    "filters": [TransposeCodec(order=(1, 0))],
                    "serializer": BytesCodec(),
                    "compressors": GzipCodec(level=1),
                },
                id="transpose-gzip",
            ),
        ],
    )
    def test_spec_changing_codec_roundtrip(self, store: Store, arr_kwargs: dict[str, Any]) -> None:
        """Array->array codecs that change the chunk spec (transpose) must
        roundtrip on every pipeline AND every store path. This is the case that
        breaks if a pipeline's async path reuses one spec across the whole codec
        chain instead of evolving it per codec. Non-square chunks make a wrong
        reshape observable.
        """
        arr = zarr.create_array(
            store=store,
            shape=(8, 12),
            dtype="int32",
            chunks=(2, 4),
            shards=None,
            fill_value=0,
            **arr_kwargs,
        )
        data = np.arange(96, dtype="int32").reshape(8, 12)
        arr[:] = data
        np.testing.assert_array_equal(arr[:], data)
        # partial read too (exercises selection on the transposed chunk)
        np.testing.assert_array_equal(arr[1:7, 2:10], data[1:7, 2:10])

    # -- write_empty_chunks / read_missing_chunks -----------------------------

    @staticmethod
    def _chunk_keys(store: Store) -> set[str]:
        """All non-metadata keys currently in the store."""
        import asyncio

        async def _list() -> set[str]:
            return {k async for k in store.list() if "zarr.json" not in k}

        return asyncio.run(_list())

    @pytest.mark.parametrize("shards", [None, (20,)], ids=["unsharded", "sharded"])
    def test_write_empty_chunks_false(self, store: Store, shards: tuple[int, ...] | None) -> None:
        """write_empty_chunks=False: a fill-only chunk reads back as fill AND is
        not persisted (no store key for it)."""
        arr = zarr.create_array(
            store=store,
            shape=(20,),
            dtype="float64",
            chunks=(10,),
            shards=shards,
            compressors=None,
            fill_value=0.0,
            config={"write_empty_chunks": False},
        )
        arr[0:10] = np.arange(10, dtype="float64") + 1
        arr[10:20] = np.zeros(10, dtype="float64")  # all fill_value
        np.testing.assert_array_equal(arr[0:10], np.arange(10, dtype="float64") + 1)
        np.testing.assert_array_equal(arr[10:20], np.zeros(10, dtype="float64"))
        if shards is None:
            # The all-fill chunk must NOT be persisted; the written one must be.
            keys = self._chunk_keys(store)
            assert any("c/0" in k for k in keys), keys  # written chunk present
            assert not any("c/1" in k for k in keys), keys  # fill chunk omitted

    def test_write_empty_chunks_true_persists(self, store: Store) -> None:
        """write_empty_chunks=True: fill-only chunks are still persisted as keys."""
        arr = zarr.create_array(
            store=store,
            shape=(20,),
            dtype="float64",
            chunks=(10,),
            shards=None,
            compressors=None,
            fill_value=0.0,
            config={"write_empty_chunks": True},
        )
        arr[:] = 0.0
        np.testing.assert_array_equal(arr[:], np.zeros(20, dtype="float64"))
        keys = self._chunk_keys(store)
        assert any("c/0" in k for k in keys), keys
        assert any("c/1" in k for k in keys), keys

    def test_read_missing_chunks_false_raises(self, store: Store) -> None:
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

    def test_read_missing_chunks_true_fills(self, store: Store) -> None:
        arr = zarr.create_array(
            store=store,
            shape=(20,),
            dtype="float64",
            chunks=(10,),
            shards=None,
            compressors=None,
            fill_value=-999.0,
        )
        np.testing.assert_array_equal(arr[:], np.full(20, -999.0))

    # -- sharding specifics ----------------------------------------------------

    def test_nested_sharding_roundtrip(self, store: Store) -> None:
        arr = zarr.create_array(
            store=store,
            shape=(20, 20),
            dtype="int32",
            chunks=(10, 10),
            shards=None,
            compressors=None,
            fill_value=0,
            serializer=ShardingCodec(
                chunk_shape=(10, 10), codecs=[ShardingCodec(chunk_shape=(5, 5))]
            ),
        )
        data = np.arange(400, dtype="int32").reshape(20, 20)
        arr[:] = data
        np.testing.assert_array_equal(arr[:], data)

    @pytest.mark.parametrize("subchunk_write_order", ["morton", "lexicographic", "colexicographic"])
    def test_partial_write_after_reopen_is_correct(
        self, store: Store, subchunk_write_order: SubchunkWriteOrder
    ) -> None:
        """Reopening a sharded array and partially overwriting it must read back
        correctly, regardless of the original subchunk_write_order.

        NOTE: subchunk_write_order is intentionally NOT recoverable on reopen (it
        is not codec metadata) — so this does NOT assert the order survives. What
        it guards is the consequence that matters: chunk locations on a write to
        an existing shard must come from the STORED shard index, not from the
        (now-possibly-default) live order. A non-square inner grid makes the
        orders physically distinct, so an offset computed from the wrong order
        would corrupt data and fail this read-back.
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

    @pytest.mark.parametrize("write_empty", [True, False])
    def test_partial_shard_write_roundtrip(self, store: Store, write_empty: bool) -> None:
        """Write a full shard, then partially overwrite it; both pipelines must
        read back the merged result. Exercises the byte-range write fast path on
        the sync store and the full-rewrite path on the async store."""
        arr = zarr.create_array(
            store=store,
            shape=(40,),
            dtype="int32",
            chunks=(4,),
            shards=(40,),
            compressors=None,
            fill_value=-1,
            config={"write_empty_chunks": write_empty},
        )
        ref = np.arange(40, dtype="int32")
        arr[:] = ref
        arr[7:18] = np.arange(700, 711, dtype="int32")
        ref[7:18] = np.arange(700, 711)
        np.testing.assert_array_equal(arr[:], ref)


class TestBatchedPipeline(CodecPipelineTests):
    pipeline_path = "zarr.core.codec_pipeline.BatchedCodecPipeline"


class TestFusedPipeline(CodecPipelineTests):
    pipeline_path = "zarr.core.codec_pipeline.FusedCodecPipeline"
