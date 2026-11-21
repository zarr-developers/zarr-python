"""Time how long it takes to read a metadata document written by zarr-python.

There are three documents. Each was written by `zarr==3.4.1` and
round-tripped through `json.dumps`/`json.loads`, so each holds what a
reader would receive from a store:

- a v3 array, `zarr.create_array(store, shape=(100, 100), chunks=(10, 10),
  dtype="f8", zarr_format=3)`;
- the same array as v2;
- a v3 group holding 200 `int32` arrays of shape `(10,)`, consolidated
  with `zarr.consolidate_metadata`.

Run from the package directory:

    PYTHONHASHSEED=0 uv run --group test python -m tests.benchmarks.read_documents

Each case is timed with `timeit`. The script reports the best of ten
repeats, as the mean time per read in microseconds. `--profile` instead
prints a cProfile of the v3 array read, sorted by cumulative time.
`test_read_documents.py` runs each case once, so the script keeps
working as the package changes.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import timeit
from typing import TYPE_CHECKING, Any

from zarr_metadata import ZarrV2ArrayMetadata, ZarrV3ArrayMetadata, ZarrV3GroupMetadata

if TYPE_CHECKING:
    from collections.abc import Callable

V3_ARRAY: dict[str, Any] = {
    "shape": [100, 100],
    "data_type": "float64",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10, 10]}},
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
    "fill_value": 0.0,
    "codecs": [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
    ],
    "attributes": {},
    "zarr_format": 3,
    "node_type": "array",
    "storage_transformers": [],
}

V2_ARRAY: dict[str, Any] = {
    "shape": [100, 100],
    "chunks": [10, 10],
    "dtype": "<f8",
    "fill_value": 0.0,
    "order": "C",
    "filters": None,
    "dimension_separator": ".",
    "compressor": {"id": "zstd", "level": 0},
    "zarr_format": 2,
}


def _child(index: int) -> tuple[str, dict[str, Any]]:
    return f"a{index}", {
        "shape": [10],
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
        ],
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }


V3_GROUP: dict[str, Any] = {
    "attributes": {},
    "zarr_format": 3,
    "consolidated_metadata": {
        "kind": "inline",
        "must_understand": False,
        "metadata": dict(_child(index) for index in range(200)),
    },
    "node_type": "group",
}


def _round_tripped(document: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(document))


CASES: tuple[tuple[str, Callable[[], object], int], ...] = (
    ("ZarrV3ArrayMetadata.from_json", lambda: ZarrV3ArrayMetadata.from_json(_V3_ARRAY), 2000),
    ("ZarrV2ArrayMetadata.from_json", lambda: ZarrV2ArrayMetadata.from_json(_V2_ARRAY), 2000),
    (
        "ZarrV3GroupMetadata.from_json, 200 consolidated arrays",
        lambda: ZarrV3GroupMetadata.from_json(_V3_GROUP),
        10,
    ),
)

_V3_ARRAY = _round_tripped(V3_ARRAY)
_V2_ARRAY = _round_tripped(V2_ARRAY)
_V3_GROUP = _round_tripped(V3_GROUP)


def main() -> None:
    parser = argparse.ArgumentParser(description="How long reading a metadata document takes.")
    parser.add_argument("--profile", action="store_true", help="profile the v3 array read")
    parser.add_argument("--repeat", type=int, default=10, help="timeit repeats; the best is shown")
    arguments = parser.parse_args()
    if arguments.profile:
        profiler = cProfile.Profile()
        profiler.runcall(lambda: [ZarrV3ArrayMetadata.from_json(_V3_ARRAY) for _ in range(1000)])
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(45)
        return
    for name, read, number in CASES:
        read()  # compile the checkers and warm every cache, as after reading many documents
        best = min(timeit.repeat(read, number=number, repeat=arguments.repeat))
        print(f"{name}: {best / number * 1e6:.1f} us")


if __name__ == "__main__":
    main()
