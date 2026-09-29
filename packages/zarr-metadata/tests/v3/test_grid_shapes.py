"""Chunk grids, judged against the shape of the array they chunk.

Each grid's definition says what the spec disallows in a grid of its
configuration over an array of a given shape, and the lengths its chunks
take along each axis of one it fits; `chunk_grid_lengths` reads both of
a grid field a scope read, and the v3 array validators judge a
document's `chunk_grid` against its `shape`.
"""

from __future__ import annotations

import pytest

from zarr_metadata.model import ZarrV3ArrayMetadata, validate_array_metadata_v3
from zarr_metadata.v3.codec.crc32c import Empty
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS,
    ChunkGridDefinition,
    JSONValue,
    chunk_grid_lengths,
    resolve,
)


def _regular(*lengths: int) -> JSONValue:
    return {"name": "regular", "configuration": {"chunk_shape": list(lengths)}}


def _rectilinear(*specs: JSONValue) -> JSONValue:
    return {"name": "rectilinear", "configuration": {"kind": "inline", "chunk_shapes": list(specs)}}


def _problems(grid: JSONValue, shape: tuple[int, ...]) -> list[tuple[tuple[str | int, ...], str]]:
    resolved, found = resolve(grid, ChunkGridDefinition, CORE_AND_EXTENSIONS)
    assert found == ()
    lengths, problems = chunk_grid_lengths(resolved, shape)
    # A grid that does not fit the shape says no lengths along any axis.
    assert lengths == (None,) * len(shape)
    return [(problem.loc, problem.kind) for problem in problems]


@pytest.mark.parametrize(
    ("grid", "shape", "lengths"),
    [
        (_regular(), (), ()),
        (_regular(4, 4), (10, 3), ({4}, {4})),
        # A chunk longer than its dimension, and a chunk over a dimension of
        # length 0.
        (_regular(8, 1), (3, 0), ({8}, {1})),
        # A bare integer repeats until it covers its dimension.
        (_rectilinear(4), (10,), ({4},)),
        (_rectilinear([4, 4, 2]), (10,), ({4, 2},)),
        # Overflowing the dimension is allowed, and the chunk past the edge
        # counts.
        (_rectilinear([4, 4, 4]), (10,), ({4},)),
        (_rectilinear([4, 4, 2, 7]), (10,), ({4, 2, 7},)),
        (_rectilinear([[4, 2], 2]), (10,), ({4, 2},)),
        # Nothing to cover, and no chunk.
        (_rectilinear([]), (0,), (set(),)),
        (_rectilinear(2, [3, 3]), (7, 6), ({2}, {3})),
        # A grid nothing in scope claims, or one that is not read, is left
        # unjudged, its lengths unknown along each axis of the array.
        ({"name": "acme.grid", "configuration": {"x": 1}}, (3, 2), (None, None)),
        (_regular(-1), (3,), (None,)),
    ],
)
def test_every_chunk_grid_gives_the_lengths_of_its_chunks_over_a_shape_it_fits(
    grid: JSONValue, shape: tuple[int, ...], lengths: tuple[set[int] | None, ...]
) -> None:
    resolved, _ = resolve(grid, ChunkGridDefinition, CORE_AND_EXTENSIONS)
    expected = tuple(None if axis is None else frozenset(axis) for axis in lengths)
    assert chunk_grid_lengths(resolved, shape) == (expected, ())


@pytest.mark.parametrize(
    ("grid", "shape"), [(_regular(4), (10, 3)), (_regular(4, 4), ()), (_regular(), (1,))]
)
def test_error_a_regular_grid_of_another_rank(grid: JSONValue, shape: tuple[int, ...]) -> None:
    assert _problems(grid, shape) == [(("configuration", "chunk_shape"), "invalid_value")]


@pytest.mark.parametrize(
    ("grid", "shape"), [(_rectilinear(4), (10, 3)), (_rectilinear(4, 4), (1,))]
)
def test_error_a_rectilinear_grid_of_another_rank(grid: JSONValue, shape: tuple[int, ...]) -> None:
    assert _problems(grid, shape) == [(("configuration", "chunk_shapes"), "invalid_value")]


@pytest.mark.parametrize("spec", [[4, 4], [[4, 2]], []])
def test_error_rectilinear_chunks_that_do_not_cover_their_dimension(spec: JSONValue) -> None:
    assert _problems(_rectilinear(2, spec), (4, 10)) == [
        (("configuration", "chunk_shapes", 1), "invalid_value")
    ]


def test_a_grid_that_says_nothing_of_the_shape_fits_every_one_its_lengths_unknown() -> None:
    lenient = ChunkGridDefinition(name="acme.grid", configuration=Empty)
    scope = CORE_AND_EXTENSIONS.extended_with(lenient)
    resolved, _ = resolve("acme.grid", ChunkGridDefinition, scope)
    assert chunk_grid_lengths(resolved, (3, 4)) == ((None, None), ())


def test_error_a_grid_whose_chunk_lengths_have_another_number_of_axes() -> None:
    # A fault in the definition, not the field.
    flat = ChunkGridDefinition(
        name="acme.grid", configuration=Empty, chunk_lengths=lambda c, n, s: (frozenset({1}),)
    )
    scope = CORE_AND_EXTENSIONS.extended_with(flat)
    resolved, _ = resolve("acme.grid", ChunkGridDefinition, scope)
    with pytest.raises(ValueError, match="'acme.grid': its chunk_lengths gave 1 axes"):
        chunk_grid_lengths(resolved, (3, 4))


def test_error_an_array_document_s_chunk_grid_that_does_not_fit_its_shape() -> None:
    # Located in the document's grid.
    document = dict(ZarrV3ArrayMetadata.create_default(shape=(4, 4)).to_json())
    document["chunk_grid"] = _regular(4)
    assert [(p.loc, p.kind) for p in validate_array_metadata_v3(document)] == [
        (("chunk_grid", "configuration", "chunk_shape"), "invalid_value")
    ]


def test_error_an_array_document_s_shape_that_is_not_read_leaves_its_grid_unjudged() -> None:
    document = dict(ZarrV3ArrayMetadata.create_default(shape=(4, 4)).to_json())
    document["chunk_grid"] = _regular(4)
    document["shape"] = (4, -1)
    assert [(p.loc, p.kind) for p in validate_array_metadata_v3(document)] == [
        (("shape",), "invalid_value")
    ]


@pytest.mark.parametrize(
    ("chunk_lengths", "match"),
    [
        (lambda configuration, nested, shape: (4, 4), "its chunk_lengths give a frozenset"),
        (lambda configuration, nested, shape: {}["axis"], "'axis'"),
    ],
)
def test_error_a_grid_whose_chunk_lengths_give_something_else_or_raise(
    chunk_lengths: object, match: str
) -> None:
    # A fault in the definition, and an error raised says whose.
    odd = ChunkGridDefinition(
        name="acme.grid",
        configuration=Empty,
        chunk_lengths=chunk_lengths,  # pyright: ignore[reportArgumentType]
    )
    resolved, _ = resolve("acme.grid", ChunkGridDefinition, CORE_AND_EXTENSIONS.extended_with(odd))
    with pytest.raises((TypeError, KeyError), match=match) as raised:
        chunk_grid_lengths(resolved, (3, 4), ("chunk_grid",))
    if isinstance(raised.value, KeyError):
        assert raised.value.__notes__ == [
            "raised by the chunk lengths of 'acme.grid', reading ('chunk_grid', 'configuration')"
        ]
