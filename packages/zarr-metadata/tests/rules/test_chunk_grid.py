"""What array a chunk grid governs, and what follows from knowing it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr_metadata.rules import validate_array_metadata_v3
from zarr_metadata.rules._chunk_grid import (  # pyright: ignore[reportPrivateUsage]
    governed_shape,
    shard_index_shape,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

BASE: Mapping[str, object] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (64, 64),
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_key_encoding": "default",
}
REGULAR: Mapping[str, object] = {"name": "regular", "configuration": {"chunk_shape": (32, 32)}}


def _rectilinear(chunk_shapes: object) -> Mapping[str, object]:
    return {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": chunk_shapes},
    }


# The shard index is uint64, so its bytes codec needs an endianness; these
# tests are about geometry, not about that.
_INDEX_CODECS = ({"name": "bytes", "configuration": {"endian": "little"}},)


def _shard(inner: object, index: object = _INDEX_CODECS) -> Mapping[str, object]:
    return {
        "name": "sharding_indexed",
        "configuration": {"chunk_shape": inner, "codecs": ("bytes",), "index_codecs": index},
    }


# (grid, array shape, the chunk shape it governs). A `None` entry is a
# dimension whose chunks differ or whose metadata cannot be read; a `None`
# result is a grid that does not even pin the rank.
GOVERNED: dict[str, tuple[object, object, object]] = {
    "regular": (REGULAR, (64, 64), (32, 32)),
    "regular-zero-extent-keeps-rank": (
        {"name": "regular", "configuration": {"chunk_shape": (0, 32)}},
        (64, 64),
        (None, 32),
    ),
    "rectilinear-uniform": (_rectilinear(((32, 32), (32, 32))), (64, 64), (32, 32)),
    "rectilinear-uniform-rle": (_rectilinear((((32, 2),), ((32, 2),))), (64, 64), (32, 32)),
    "rectilinear-bare-int-is-a-regular-step": (_rectilinear((32, 32)), (64, 64), (32, 32)),
    "rectilinear-mixed-resolves-per-dimension": (
        _rectilinear(((30, 34), (32, 32))),
        (64, 64),
        (None, 32),
    ),
    "unknown-grid-keeps-the-array-rank": (
        {"name": "mycorp.hilbert", "configuration": {"anything": 1}},
        (64, 64),
        (None, None),
    ),
    "no-usable-array-shape": (None, "not a shape", None),
}


@pytest.mark.parametrize(("grid", "shape", "expected"), GOVERNED.values(), ids=list(GOVERNED))
def test_governed_shape(grid: object, shape: object, expected: object) -> None:
    assert governed_shape(grid, shape) == expected


def test_shard_index_shape_is_chunks_per_shard_plus_two() -> None:
    # "a shape that matches the chunks per shard tuple with an appended
    # dimension of size 2" — 128/32 = 4 along both axes.
    assert shard_index_shape((128, 128), (32, 32)) == (4, 4, 2)
    # The rank survives even when no extent does.
    assert shard_index_shape(None, (32, 32)) == (None, None, 2)
    assert shard_index_shape((None, 128), (32, 32)) == (None, 4, 2)


@pytest.mark.parametrize(
    "chunk_shapes",
    [((32, 32), (32, 32)), (((32, 2),), ((32, 2),)), (32, 32)],
    ids=["explicit", "run-length", "bare-int"],
)
def test_error_shard_must_divide_a_uniform_rectilinear_grid(chunk_shapes: object) -> None:
    # A rectilinear grid has no single chunk shape, but a uniform one pins
    # every extent, and 7 divides neither.
    problems = validate_array_metadata_v3(
        {**BASE, "chunk_grid": _rectilinear(chunk_shapes), "codecs": (_shard((7, 7)),)}
    )
    assert len(problems) == 2
    assert all("does not evenly divide" in problem.message for problem in problems)


def test_a_varying_dimension_declines_while_a_pinned_one_is_judged() -> None:
    # chunk_shapes ((30, 34), (32, 32)): axis 0 varies, axis 1 is 32
    # everywhere. Only the axis that is knowable may be judged.
    grid = _rectilinear(((30, 34), (32, 32)))
    problems = validate_array_metadata_v3({**BASE, "chunk_grid": grid, "codecs": (_shard((7, 7)),)})
    assert [problem.loc for problem in problems] == [
        ("codecs", 0, "configuration", "chunk_shape", 1)
    ]
    # 32 divides the pinned axis, and axis 0 is unknown: nothing to report.
    assert (
        validate_array_metadata_v3({**BASE, "chunk_grid": grid, "codecs": (_shard((7, 32)),)}) == ()
    )


def test_error_index_codecs_are_judged_against_the_index_rank() -> None:
    # The index of a 2-D shard is rank 3, so a rank-1 transpose is wrong.
    index = (
        {"name": "transpose", "configuration": {"order": (0,)}},
        {"name": "bytes", "configuration": {"endian": "little"}},
    )
    problems = validate_array_metadata_v3(
        {**BASE, "chunk_grid": REGULAR, "codecs": (_shard((8, 8), index=index),)}
    )
    assert [problem.message for problem in problems] == [
        "order has 1 entries but the incoming array has 3 dimensions"
    ]


def test_error_a_bad_inner_extent_costs_that_axis_and_nothing_else() -> None:
    # The zero is reported, and the inner pipeline is still judged against
    # the rank the inner chunk shape declares.
    inner = _shard((0, 2))
    inner["configuration"] = {  # type: ignore[index]
        **inner["configuration"],  # type: ignore[dict-item]
        "codecs": ({"name": "transpose", "configuration": {"order": (0, 1, 2)}}, "bytes"),
    }
    messages = [
        problem.message
        for problem in validate_array_metadata_v3(
            {**BASE, "data_type": "uint16", "chunk_grid": REGULAR, "codecs": (inner,)}
        )
    ]
    assert any("positive chunk extent" in message for message in messages)
    assert any("order has 3 entries" in message for message in messages)
    assert any("endian is required" in message for message in messages)
