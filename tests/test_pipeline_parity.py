"""Pipeline parity test — exhaustive matrix of read/write scenarios.

For every cell of the matrix (codec config x layout x operation
sequence x runtime config), assert that ``FusedCodecPipeline`` and
``BatchedCodecPipeline`` produce semantically identical results:

  * Same returned array contents on read.
  * Same set of store keys after writes (catches divergent empty-shard
    handling: one pipeline deletes, the other writes an empty blob).
  * Reading each pipeline's store contents through the *other* pipeline
    yields the same array (catches "wrote a layout that only one
    pipeline can read" bugs).

Pipeline-divergence bugs (e.g. one pipeline writes a dense shard
layout while the other writes a compact layout) fail this test
loudly with a clear diff, instead of waiting for a downstream
test to trip over the symptom.

Byte-for-byte equality of store contents is intentionally NOT
checked: codecs like gzip embed the wall-clock timestamp in their
output, so two compressions of the same data done at different
seconds produce different bytes despite being semantically
identical.

The matrix axes are:

  * codec chain — bytes-only, gzip, with/without sharding
  * layout — chunk_shape, shard_shape (None for no sharding)
  * write sequence — full overwrite, partial in middle, scalar to one
    cell, multiple overlapping writes, sequence ending in fill values
  * runtime config — write_empty_chunks True/False
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.sharding import ShardingCodec
from zarr.core.config import config as zarr_config
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def _store_snapshot(store: MemoryStore) -> dict[str, bytes]:
    """Return {key: bytes} for every entry in the store."""
    return {k: bytes(v.to_bytes()) for k, v in store._store_dict.items()}


# ---------------------------------------------------------------------------
# Matrix definitions
# ---------------------------------------------------------------------------


# Each codec config is (filters, serializer, compressors). We only vary the
# pieces that actually affect the pipeline. compressors=None means a
# fixed-size chain (the byte-range fast path is eligible when sharded).
CodecConfig = dict[str, Any]

CODEC_CONFIGS: list[tuple[str, CodecConfig]] = [
    ("bytes-only", {"compressors": None}),
    ("gzip", {"compressors": GzipCodec(level=1)}),
]


# (id, kwargs) — chunks/shards layout. kwargs are passed to create_array.
LayoutConfig = dict[str, Any]

LAYOUT_CONFIGS: list[tuple[str, LayoutConfig]] = [
    ("1d-unsharded", {"shape": (100,), "chunks": (10,), "shards": None}),
    ("1d-1chunk-per-shard", {"shape": (100,), "chunks": (10,), "shards": (10,)}),
    ("1d-multi-chunk-per-shard", {"shape": (100,), "chunks": (10,), "shards": (50,)}),
    ("2d-unsharded", {"shape": (20, 20), "chunks": (5, 5), "shards": None}),
    ("2d-sharded", {"shape": (20, 20), "chunks": (5, 5), "shards": (10, 10)}),
    # Nested sharding: outer chunk (10,10) sharded into inner chunks (5,5).
    # Restricted to bytes-only codec because combining an outer ShardingCodec
    # with a compressor (gzip) triggers a ZarrUserWarning and results in a
    # checksum mismatch inside the inner shard index — a known limitation, not
    # a pipeline-parity bug.  The bytes-only path still exercises the full
    # two-level shard encoding/decoding in both pipelines.
    (
        "2d-nested-sharded",
        {
            "shape": (20, 20),
            "chunks": (10, 10),
            "shards": None,
            "serializer": ShardingCodec(
                chunk_shape=(10, 10),
                codecs=[ShardingCodec(chunk_shape=(5, 5))],
            ),
            # Only run with the bytes-only codec config; gzip is incompatible
            # with nested sharding (see comment above).
            "_codec_ids": {"bytes-only"},
        },
    ),
]


WriteOp = tuple[Any, Any]  # (selection, value)
WriteSequence = tuple[str, list[WriteOp]]


def _full_overwrite(shape: tuple[int, ...]) -> list[WriteOp]:
    return [((slice(None),) * len(shape), np.arange(int(np.prod(shape))).reshape(shape) + 1)]


def _partial_middle(shape: tuple[int, ...]) -> list[WriteOp]:
    if len(shape) == 1:
        n = shape[0]
        return [((slice(n // 4, 3 * n // 4),), 7)]
    # 2D: write a centered block
    rs = slice(shape[0] // 4, 3 * shape[0] // 4)
    cs = slice(shape[1] // 4, 3 * shape[1] // 4)
    return [((rs, cs), 7)]


def _scalar_one_cell(shape: tuple[int, ...]) -> list[WriteOp]:
    if len(shape) == 1:
        return [((shape[0] // 2,), 99)]
    return [((shape[0] // 2, shape[1] // 2), 99)]


def _overlapping(shape: tuple[int, ...]) -> list[WriteOp]:
    if len(shape) == 1:
        n = shape[0]
        return [
            ((slice(0, n // 2),), 1),
            ((slice(n // 4, 3 * n // 4),), 2),
            ((slice(n // 2, n),), 3),
        ]
    rs1, cs1 = slice(0, shape[0] // 2), slice(0, shape[1] // 2)
    rs2, cs2 = slice(shape[0] // 4, 3 * shape[0] // 4), slice(shape[1] // 4, 3 * shape[1] // 4)
    return [((rs1, cs1), 1), ((rs2, cs2), 2)]


def _ends_in_fill(shape: tuple[int, ...]) -> list[WriteOp]:
    """Write something then overwrite it with fill — exercises empty-chunk handling."""
    full = (slice(None),) * len(shape)
    return [(full, 5), (full, 0)]


def _ends_in_partial_fill(shape: tuple[int, ...]) -> list[WriteOp]:
    """Write data, then overwrite half with fill — some chunks become empty."""
    full: tuple[slice, ...]
    half: tuple[slice, ...]
    if len(shape) == 1:
        full = (slice(None),)
        half = (slice(0, shape[0] // 2),)
    else:
        full = (slice(None), slice(None))
        half = (slice(0, shape[0] // 2), slice(None))
    return [(full, 5), (half, 0)]


SEQUENCES: list[tuple[str, Callable[[tuple[int, ...]], list[WriteOp]]]] = [
    ("full-overwrite", _full_overwrite),
    ("partial-middle", _partial_middle),
    ("scalar-one-cell", _scalar_one_cell),
    ("overlapping", _overlapping),
    ("ends-in-fill", _ends_in_fill),
    ("ends-in-partial-fill", _ends_in_partial_fill),
]


WRITE_EMPTY_CHUNKS = [False, True]


# ---------------------------------------------------------------------------
# Matrix iteration (pruned)
# ---------------------------------------------------------------------------


def _matrix() -> Iterator[Any]:
    for codec_id, codec_kwargs in CODEC_CONFIGS:
        for layout_id, layout in LAYOUT_CONFIGS:
            allowed = layout.get("_codec_ids")
            if allowed is not None and codec_id not in allowed:
                continue
            for seq_id, seq_fn in SEQUENCES:
                for wec in WRITE_EMPTY_CHUNKS:
                    yield pytest.param(
                        codec_kwargs,
                        layout,
                        seq_fn,
                        wec,
                        id=f"{layout_id}-{codec_id}-{seq_id}-wec{wec}",
                    )


# ---------------------------------------------------------------------------
# The parity test
# ---------------------------------------------------------------------------


def _write_under_pipeline(
    pipeline_path: str,
    codec_kwargs: CodecConfig,
    layout: LayoutConfig,
    sequence: list[WriteOp],
    write_empty_chunks: bool,
) -> tuple[MemoryStore, Any]:
    """Apply a sequence of writes via the chosen pipeline.

    Returns (store with the written data, final array contents read back).
    """
    # Strip private metadata keys (e.g. "_codec_ids") before passing to create_array.
    array_layout = {k: v for k, v in layout.items() if not k.startswith("_")}
    store = MemoryStore()
    with zarr_config.set({"codec_pipeline.path": pipeline_path}):
        arr = zarr.create_array(
            store=store,
            dtype="float64",
            fill_value=0.0,
            config={"write_empty_chunks": write_empty_chunks},
            **array_layout,
            **codec_kwargs,
        )
        for sel, val in sequence:
            arr[sel] = val
        contents = arr[...]
    return store, contents


def _read_under_pipeline(pipeline_path: str, store: MemoryStore) -> Any:
    """Re-open an existing store under the chosen pipeline and read it whole."""
    with zarr_config.set({"codec_pipeline.path": pipeline_path}):
        arr = zarr.open_array(store=store, mode="r")
        return arr[...]


_BATCHED = "zarr.core.codec_pipeline.BatchedCodecPipeline"
_FUSED = "zarr.core.codec_pipeline.FusedCodecPipeline"


@pytest.mark.parametrize(
    ("codec_kwargs", "layout", "sequence_fn", "write_empty_chunks"),
    list(_matrix()),
)
def test_pipeline_parity(
    codec_kwargs: CodecConfig,
    layout: LayoutConfig,
    sequence_fn: Callable[[tuple[int, ...]], list[WriteOp]],
    write_empty_chunks: bool,
) -> None:
    """FusedCodecPipeline must be semantically identical to BatchedCodecPipeline.

    Three checks, in order of decreasing diagnostic value:

      1. Both pipelines return the same array contents after the same
         write sequence (catches semantic correctness bugs).
      2. Both pipelines produce the same set of store keys (catches
         empty-shard divergence: one deletes, the other doesn't).
      3. Each pipeline can correctly read the *other* pipeline's
         output (catches layout-divergence bugs that would prevent
         interop, e.g. dense vs compact shard layouts).

    Byte-for-byte store equality is intentionally not checked: codecs
    like gzip embed wall-clock timestamps that vary between runs.
    """
    sequence = sequence_fn(layout["shape"])

    batched_store, batched_arr = _write_under_pipeline(
        _BATCHED, codec_kwargs, layout, sequence, write_empty_chunks
    )
    sync_store, sync_arr = _write_under_pipeline(
        _FUSED, codec_kwargs, layout, sequence, write_empty_chunks
    )

    # 1. Array contents must agree.
    np.testing.assert_array_equal(
        sync_arr,
        batched_arr,
        err_msg="FusedCodecPipeline returned different array contents than BatchedCodecPipeline",
    )

    # 2. Store key sets must agree.
    batched_keys = set(batched_store._store_dict) - {"zarr.json"}
    sync_keys = set(sync_store._store_dict) - {"zarr.json"}
    assert sync_keys == batched_keys, (
        f"Pipelines disagree on which store keys exist.\n"
        f"  only in batched: {sorted(batched_keys - sync_keys)}\n"
        f"  only in sync:    {sorted(sync_keys - batched_keys)}"
    )

    # 3. Cross-read: each pipeline must correctly read the other's output.
    sync_reads_batched = _read_under_pipeline(_FUSED, batched_store)
    batched_reads_sync = _read_under_pipeline(_BATCHED, sync_store)
    np.testing.assert_array_equal(
        sync_reads_batched,
        batched_arr,
        err_msg="FusedCodecPipeline could not correctly read BatchedCodecPipeline's output",
    )
    np.testing.assert_array_equal(
        batched_reads_sync,
        sync_arr,
        err_msg="BatchedCodecPipeline could not correctly read FusedCodecPipeline's output",
    )


# ---------------------------------------------------------------------------
# Read parity: cover partial reads (not just full reads as in the matrix above)
# ---------------------------------------------------------------------------


def _read_selections(shape: tuple[int, ...]) -> list[tuple[str, Any]]:
    """Selections that exercise the partial-decode path differently."""
    if len(shape) == 1:
        n = shape[0]
        return [
            ("scalar-first", (0,)),
            ("scalar-mid", (n // 2,)),
            ("partial-slice", (slice(n // 4, 3 * n // 4),)),
            ("strided", (slice(0, n, 3),)),
            ("full", (slice(None),)),
        ]
    return [
        ("scalar-first", (0,) * len(shape)),
        ("scalar-mid", tuple(s // 2 for s in shape)),
        ("partial-slice", tuple(slice(s // 4, 3 * s // 4) for s in shape)),
        ("full", (slice(None),) * len(shape)),
    ]


def _read_matrix() -> Iterator[Any]:
    for codec_id, codec_kwargs in CODEC_CONFIGS:
        for layout_id, layout in LAYOUT_CONFIGS:
            allowed = layout.get("_codec_ids")
            if allowed is not None and codec_id not in allowed:
                continue
            for sel_id, sel in _read_selections(layout["shape"]):
                yield pytest.param(
                    codec_kwargs,
                    layout,
                    sel,
                    id=f"{layout_id}-{codec_id}-{sel_id}",
                )


@pytest.mark.parametrize(
    ("codec_kwargs", "layout", "selection"),
    list(_read_matrix()),
)
def test_pipeline_read_parity(
    codec_kwargs: CodecConfig,
    layout: LayoutConfig,
    selection: Any,
) -> None:
    """Partial reads via FusedCodecPipeline must match BatchedCodecPipeline.

    The full-write/full-read parity test above doesn't exercise partial
    reads (e.g. a single element from a sharded array), which take a
    different code path (``_decode_partial_single`` on the sharding
    codec). This test fills the array under one pipeline and reads
    arbitrary selections under both, asserting equality.
    """
    # Fill under batched (the canonical pipeline) so the contents are
    # well-defined regardless of the codec under test.
    store, _full = _write_under_pipeline(
        _BATCHED, codec_kwargs, layout, _full_overwrite(layout["shape"]), True
    )

    with zarr_config.set({"codec_pipeline.path": _BATCHED}):
        batched_arr = zarr.open_array(store=store, mode="r")[selection]
    with zarr_config.set({"codec_pipeline.path": _FUSED}):
        sync_arr = zarr.open_array(store=store, mode="r")[selection]

    np.testing.assert_array_equal(
        sync_arr,
        batched_arr,
        err_msg=(
            f"FusedCodecPipeline read returned different result than BatchedCodecPipeline "
            f"for selection {selection!r}"
        ),
    )
