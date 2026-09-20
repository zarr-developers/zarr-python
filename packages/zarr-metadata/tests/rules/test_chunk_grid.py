"""What array a chunk grid governs, and what follows from knowing it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr_metadata.rules import validate_array_metadata_v3
from zarr_metadata.v3._parts import ChunkGrid, shard_index_grid
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS

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


def _u(*lengths: int) -> tuple[frozenset[int], ...]:
    return tuple(frozenset({length}) for length in lengths)


# (grid metadata, array shape, rank, the lengths each axis's chunks take).
# A `None` axis is one this package cannot read; a `None` extents is a grid
# that does not even pin the rank.
GRIDS: dict[str, tuple[object, object, object, object]] = {
    "regular": (REGULAR, (64, 64), 2, _u(32, 32)),
    # A zero extent is not a grid, so the entity refuses to exist and
    # only the rank the array shape pins survives -- the same answer as
    # for a grid this package cannot read at all.
    "regular-zero-length-is-unreadable": (
        {"name": "regular", "configuration": {"chunk_shape": (0, 32)}},
        (64, 64),
        2,
        (None, None),
    ),
    "rectilinear-uniform": (_rectilinear(((32, 32), (32, 32))), (64, 64), 2, _u(32, 32)),
    "rectilinear-uniform-rle": (_rectilinear((((32, 2),), ((32, 2),))), (64, 64), 2, _u(32, 32)),
    "rectilinear-bare-int-is-a-regular-step": (_rectilinear((32, 32)), (64, 64), 2, _u(32, 32)),
    "rectilinear-varying-axis-keeps-its-lengths": (
        _rectilinear(((30, 34), (32, 32))),
        (64, 64),
        2,
        (frozenset({30, 34}), frozenset({32})),
    ),
    "rectilinear-rle-varying": (
        _rectilinear(
            ((((30, 2)), ((34, 1))),),
        ),
        (64,),
        1,
        (frozenset({30, 34}),),
    ),
    "unknown-grid-keeps-the-array-rank": (
        {"name": "mycorp.hilbert", "configuration": {"anything": 1}},
        (64, 64),
        2,
        (None, None),
    ),
    "no-usable-array-shape": (None, "not a shape", None, None),
}


def _grid_of(grid: object, shape: object) -> ChunkGrid:
    """The grid `grid` describes over an array of `shape`.

    A grid entity builds its own; one out of scope pins only the rank the
    array shape gives it.
    """
    name = grid if isinstance(grid, str) else (grid or {}).get("name")  # type: ignore[union-attr]
    entity_type = CORE_AND_EXTENSIONS.resolve("chunk_grid", name) if isinstance(name, str) else None
    if entity_type is None:
        return ChunkGrid.unreadable(shape)
    entity, _ = entity_type.coerce(grid, CORE_AND_EXTENSIONS)
    if entity is None:
        return ChunkGrid.unreadable(shape)
    return entity.grid(shape)  # type: ignore[attr-defined]


@pytest.mark.parametrize(("grid", "shape", "rank", "extents"), GRIDS.values(), ids=list(GRIDS))
def test_chunk_grid_of(grid: object, shape: object, rank: object, extents: object) -> None:
    built = _grid_of(grid, shape)
    assert built.rank == rank
    assert built.extents == extents


def test_permuting_reorders_the_axes() -> None:
    grid = _grid_of(_rectilinear(((30, 34), (32, 32))), (64, 64))
    assert grid.permuted((1, 0)).extents == (frozenset({32}), frozenset({30, 34}))
    # An order that is not a permutation of the rank keeps the rank only.
    assert grid.permuted((0, 1, 2)).extents is None
    assert grid.permuted((0, 1, 2)).rank == 2


def test_shard_index_grid_is_chunks_per_shard_plus_two() -> None:
    # "a shape that matches the chunks per shard tuple with an appended
    # dimension of size 2" -- 128/32 = 4 along both axes.
    assert shard_index_grid(ChunkGrid.regular((128, 128)), (32, 32)).extents == _u(4, 4, 2)
    # A shard whose own extents vary makes the chunk count vary with it.
    assert shard_index_grid(ChunkGrid.derived((frozenset({30, 60}),)), (15,)).extents == (
        frozenset({2, 4}),
        frozenset({2}),
    )
    # The trailing 2 is fixed by the spec, so it survives knowing nothing.
    assert shard_index_grid(ChunkGrid(None, None), (32, 32)).extents == (None, None, frozenset({2}))


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


def test_error_a_shard_must_divide_every_chunk_a_varying_axis_has() -> None:
    # The case the per-grid summary could not express: dim 0's chunks are
    # 30 and 34 long, dim 1's are all 32, and a transpose swaps them before
    # the shard sees them. The inner extent must divide *every* length the
    # axis takes, so only a common divisor of 30 and 34 will do.
    grid = _rectilinear(((30, 34), (32, 32)))
    transpose = {"name": "transpose", "configuration": {"order": (1, 0)}}

    def verdict(inner: object) -> list[object]:
        return [
            problem.message
            for problem in validate_array_metadata_v3(
                {**BASE, "chunk_grid": grid, "codecs": (transpose, _shard(inner))}
            )
        ]

    assert verdict((16, 2)) == []  # 2 divides both 30 and 34
    assert verdict((16, 15)) == [  # 15 divides 30 but not 34
        "inner chunk extent 15 does not evenly divide the incoming extent 34"
    ]
    assert verdict((16, 30)) == [  # 30 divides itself but not 34
        "inner chunk extent 30 does not evenly divide the incoming extent 34"
    ]
    assert verdict((16, 34)) == [  # and the other way round
        "inner chunk extent 34 does not evenly divide the incoming extent 30"
    ]


def test_an_unreadable_axis_declines_while_its_neighbours_are_judged() -> None:
    # A grid this package cannot read pins the rank and nothing else, so
    # every axis declines; a rank mismatch is still caught.
    grid = {"name": "mycorp.hilbert", "configuration": {"order": 3}}
    assert (
        validate_array_metadata_v3({**BASE, "chunk_grid": grid, "codecs": (_shard((7, 7)),)}) == ()
    )
    problems = validate_array_metadata_v3(
        {**BASE, "chunk_grid": grid, "codecs": (_shard((7, 7, 7)),)}
    )
    assert [problem.message for problem in problems] == [
        "chunk_shape has 3 entries but the incoming array has 2 dimensions"
    ]


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


def test_error_a_bad_inner_extent_costs_the_shard() -> None:
    # A shard with a zero inner extent is not a shard, so it does not
    # exist and nothing inside it is interpreted. The JSON is still there
    # on the `Opaque` that replaces it; what is gone is the reading, and
    # the one report that matters is the one you must fix first.
    inner = _shard((0, 2))
    inner["configuration"] = {  # type: ignore[index]
        **inner["configuration"],  # type: ignore[dict-item]
        "codecs": ({"name": "transpose", "configuration": {"order": (0, 1, 2)}}, "bytes"),
    }
    problems = validate_array_metadata_v3(
        {**BASE, "data_type": "uint16", "chunk_grid": REGULAR, "codecs": (inner,)}
    )
    assert [problem.loc for problem in problems] == [
        ("codecs", 0, "configuration", "chunk_shape", 0)
    ]


# An unmodelled codec is the ordinary case, not an exotic one: every
# `numcodecs.*` filter is one. What it costs must be only what it actually
# obscures.
_UNMODELLED = {
    "name": "numcodecs.delta",
    "configuration": {"dtype": "<u2"},
    "must_understand": False,
}


def test_a_shard_behind_an_unmodelled_codec_is_still_judged() -> None:
    # The inner grid is this codec's own `chunk_shape` and the index is a
    # uint64 array, whatever reached the codec. Neither waits on upstream.
    nested = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": (16, 16),
            "codecs": ({"name": "transpose", "configuration": {"order": (0, 1, 2)}}, "bytes"),
            "index_codecs": ("bytes",),
        },
    }
    messages = [
        problem.message
        for problem in validate_array_metadata_v3(
            {**BASE, "data_type": "uint16", "chunk_grid": REGULAR, "codecs": (_UNMODELLED, nested)}
        )
    ]
    assert any("order has 3 entries" in message for message in messages)
    assert any("uint64" in message for message in messages)


def test_an_unusable_data_type_does_not_hide_the_geometry() -> None:
    # `data_type` costs itself and nothing else: the grid is still readable,
    # so a reader sees every fault at once instead of one per round trip.
    messages = [
        problem.message
        for problem in validate_array_metadata_v3(
            {**BASE, "data_type": 5, "chunk_grid": REGULAR, "codecs": (_shard((7, 7)),)}
        )
    ]
    assert any("expected a metadata field" in message for message in messages)
    assert sum("does not evenly divide" in message for message in messages) == 2


def test_permuting_by_a_non_permutation_declines_rather_than_raising() -> None:
    # `order` is shape-validated only as a tuple of integers, so this must
    # not be what decides whether a validator raises IndexError.
    grid = ChunkGrid.regular((4, 4))
    for order in ((5, 0), (0, 0), (-1, 0)):
        assert grid.permuted(order) == ChunkGrid(2, None)
