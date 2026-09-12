"""Reproducible planning measurements; no storage I/O or timing assertions.

Run with the package on PYTHONPATH in the repository's Hatch test environment.
Use the same interpreter and this script with the old/new package paths to compare.
The connected-components probe is deliberately not a production planner: it
measures independent table construction, without defining a new public row API.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_indexing import ArrayMap, IndexDomain, IndexTransform, plan_chunks
from zarr_indexing.grid import dimension_grids_from_chunks

if TYPE_CHECKING:
    from collections.abc import Callable


def measure(operation: Callable[[], Any], repeats: int) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    tracemalloc.start()
    result = operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {"median_ms": statistics.median(samples) * 1000, "peak_mib": peak / 2**20}


def consume(values: Any) -> int:
    return sum(1 for _ in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    results: dict[str, Any] = {}
    indices = np.arange(1000)
    cases = [
        ("box", IndexTransform.from_shape((1000, 1000)), (10, 10), (1000, 1000)),
        (
            "sorted",
            IndexTransform.from_shape((1_000_000,)).oindex[np.arange(1_000_000)],
            (100_000,),
            (1_000_000,),
        ),
        (
            "correlated",
            IndexTransform.from_shape((10_000, 10_000)).vindex[indices * 9, indices * 9],
            (10, 10),
            (10_000, 10_000),
        ),
        (
            "correlated_slab",
            IndexTransform.from_shape((1000, 1000, 1000)).vindex[indices, indices, :],
            (1000, 1000, 1000),
            (1000, 1000, 1000),
        ),
    ]
    for name, transform, chunks, shape in cases:
        grids = dimension_grids_from_chunks(chunks, shape=shape)
        results[name] = measure(
            lambda transform=transform, grids=grids: consume(plan_chunks(transform, grids)),
            args.repeats,
        )

    transform = IndexTransform.from_shape((10_000,)).oindex[np.arange(10_000)]
    grids = dimension_grids_from_chunks((10,), shape=(10_000,))
    (table,) = plan_chunks(transform, grids).partition().sets

    def read_local_rows() -> None:
        for row in range(len(table)):
            table.local[table.run(row)]

    results["local_rows"] = measure(read_local_rows, args.repeats)
    part = plan_chunks(
        IndexTransform.from_shape((100, 100, 100)),
        dimension_grids_from_chunks((1, 1, 1), shape=(100, 100, 100)),
    ).partition()
    results["all_coordinates"] = measure(part.chunk_coords, args.repeats)

    # TensorStore factors connected components in the input/grid dependency
    # graph. Here outputs 0 and 1 depend on u, while output 2 depends on v.
    # A single joint block unnecessarily expands these independent components.
    i = np.arange(1000)
    transform = IndexTransform(
        IndexDomain.from_shape((1000, 1000)),
        (ArrayMap(i[:, None]), ArrayMap(i[:, None]), ArrayMap(i[None, :])),
    )
    grids = dimension_grids_from_chunks((1000, 1000, 1000), shape=(1000, 1000, 1000))
    results["component_partition"] = measure(
        lambda: plan_chunks(transform, grids).partition(), args.repeats
    )
    results["component_walk"] = measure(
        lambda: consume(plan_chunks(transform, grids)), args.repeats
    )
    base = IndexTransform.from_shape((1000, 1000, 1000))
    results["component_vindex_compile"] = measure(
        lambda: base.vindex[i[:, None], i[:, None], i[None, :]], args.repeats
    )
    left = IndexTransform(IndexDomain.from_shape((1000,)), (ArrayMap(i), ArrayMap(i)))
    right = IndexTransform(IndexDomain.from_shape((1000,)), (ArrayMap(i),))
    results["independent_components_probe"] = measure(
        lambda: (
            plan_chunks(left, grids[:2]).partition(),
            plan_chunks(right, grids[2:]).partition(),
        ),
        args.repeats,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
