"""Strategies for whole Zarr v3 documents: an array's, a group's, and a group's with consolidated metadata.

An array's members depend on each other, and each is drawn for what it
depends on: the chunk grid for the shape, the fill value for the data
type, and the codecs for the chunks of that grid and that data type. A
member the spec does not define is drawn only as one a reader may
ignore: an object that says `must_understand` is `false`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

from hypothesis import reject
from hypothesis import strategies as st

from zarr_metadata.definition._definition import canonicalize
from zarr_metadata.model import read_array_metadata_v3, read_consolidated_metadata_v3
from zarr_metadata.testing._draw import (
    any_spelling,
    data_type_declarations,
    data_type_names,
    fill_values_of,
    pipelines_for,
    read,
    scope_of,
    strategy_for,
)
from zarr_metadata.testing._strategy import CHARACTERS, MAX_RANK, NOTHING_KNOWN
from zarr_metadata.v3._definition import (
    ArrayExtraFieldDefinition,
    Chunk,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    ExtraFieldDefinition,
    GroupExtraFieldDefinition,
    chunk_grid_lengths,
)
from zarr_metadata.v3._hierarchy import name_faults
from zarr_metadata.v3.consolidated import (
    ZARR_V3_CONSOLIDATED_METADATA_KEY,
    ZarrV3ConsolidatedMetadataJSON,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hypothesis.strategies import SearchStrategy

    from zarr_metadata._common import JSONValue
    from zarr_metadata.definition._definition import Definition
    from zarr_metadata.testing._strategy import (
        ChunkGridStrategy,
        ChunkKeyEncodingStrategy,
        Spellings,
        Strategies,
    )
    from zarr_metadata.v3 import ZarrV3DeclarationJSON
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
    from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON

_MAX_MEMBERS: Final = 3
"""The most members an object in attributes, or the fields a document adds, are drawn with."""

_TEXT: Final = st.text(CHARACTERS, max_size=8)

_JSON: Final = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | _TEXT,
    lambda inner: (
        st.lists(inner, max_size=_MAX_MEMBERS).map(tuple)
        | st.dictionaries(_TEXT, inner, max_size=_MAX_MEMBERS)
    ),
    max_leaves=8,
)
"""JSON values, with JSON arrays as tuples. Only finite numbers: JSON has no others."""

_ATTRIBUTES: Final = st.dictionaries(_TEXT, _JSON, max_size=_MAX_MEMBERS)

SHAPES: Final = st.lists(st.integers(0, 2**16), max_size=MAX_RANK).map(tuple)
"""The shape of an array: no axes, or some, each of length 0 or more."""


def _spelled(
    declaration: ZarrV3DeclarationJSON,
    kind: type[Definition[Any]],
    spellings: Spellings,
    strategies: Strategies,
) -> ZarrV3DeclarationJSON:
    """`declaration`, in its simplest spelling if `spellings` asks for that."""
    if spellings == "all":
        return declaration
    return cast("ZarrV3DeclarationJSON", canonicalize(declaration, kind, scope_of(strategies))[0])


def spelled_pipeline(
    pipeline: tuple[ZarrV3DeclarationJSON, ...], spellings: Spellings, strategies: Strategies
) -> tuple[ZarrV3DeclarationJSON, ...]:
    """Each codec of `pipeline`, in its simplest spelling if `spellings` asks for that."""
    return tuple(_spelled(codec, CodecDefinition, spellings, strategies) for codec in pipeline)


def _extra_fields(kind: type[ExtraFieldDefinition[Any]]) -> SearchStrategy[dict[str, JSONValue]]:
    """Members of a document of `kind` that the spec does not define, each an object saying a reader may ignore it.

    None is named `consolidated_metadata`, to which the core spec gives a
    meaning of its own.
    """
    waived = st.dictionaries(
        _TEXT.filter(lambda key: key != "must_understand"), _JSON, max_size=_MAX_MEMBERS
    ).map(lambda members: {**members, "must_understand": False})
    # The names it must not write are drawn too, so the filter is what keeps them out.
    reserved = sorted({*kind.spec_members, ZARR_V3_CONSOLIDATED_METADATA_KEY})
    names = (st.text(CHARACTERS, min_size=1, max_size=8) | st.sampled_from(reserved)).filter(
        lambda name: name != ZARR_V3_CONSOLIDATED_METADATA_KEY and name not in kind.spec_members
    )
    return st.dictionaries(names, waived, max_size=2)


@st.composite
def _chunk_grid(
    draw: st.DrawFn, shape: tuple[int, ...], spellings: Spellings, strategies: Strategies
) -> tuple[ZarrV3DeclarationJSON, Any]:
    """A chunk grid that fits `shape`, and the chunk lengths it gives along each axis; a `ValueError` for one that does not fit."""
    scope = scope_of(strategies)
    definition = scope.chunk_grids[draw(st.sampled_from(sorted(scope.chunk_grids)))]
    found = cast("ChunkGridStrategy", strategy_for(definition, strategies))
    declaration = draw(any_spelling(definition, found.configurations(shape, strategies)))
    lengths, problems = chunk_grid_lengths(
        read(declaration, ChunkGridDefinition, strategies), shape
    )
    if len(problems) != 0:
        msg = f"{declaration!r}, drawn over the shape {shape!r}, is refused: {problems!r}"
        raise ValueError(msg)
    return _spelled(declaration, ChunkGridDefinition, spellings, strategies), lengths


@st.composite
def _chunk_key_encoding(
    draw: st.DrawFn, spellings: Spellings, strategies: Strategies
) -> ZarrV3DeclarationJSON:
    scope = scope_of(strategies)
    definition = scope.chunk_key_encodings[draw(st.sampled_from(sorted(scope.chunk_key_encodings)))]
    found = cast("ChunkKeyEncodingStrategy", strategy_for(definition, strategies))
    declaration = draw(any_spelling(definition, found.configurations(strategies)))
    return _spelled(declaration, ChunkKeyEncodingDefinition, spellings, strategies)


@st.composite
def array_document(
    draw: st.DrawFn,
    shapes: SearchStrategy[tuple[int, ...]],
    spellings: Spellings,
    strategies: Strategies,
) -> ZarrV3ArrayMetadataJSON:
    """An array document, each of whose members fits the others; a `ValueError` when its definitions refuse it."""
    shape = draw(shapes)
    chunk_grid, lengths = draw(_chunk_grid(shape, spellings, strategies))
    data_type = draw(data_type_declarations(data_type_names(strategies), strategies))
    read_data_type = read(data_type, DataTypeDefinition, strategies)
    pipeline = pipelines_for(Chunk(lengths, read_data_type), strategies)
    if pipeline is None:
        reject()  # no array -> bytes codec the strategies hold takes this data type
    document: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": _spelled(data_type, DataTypeDefinition, spellings, strategies),
        "chunk_grid": chunk_grid,
        "chunk_key_encoding": draw(_chunk_key_encoding(spellings, strategies)),
        "fill_value": draw(fill_values_of(read_data_type, strategies)),
        "codecs": spelled_pipeline(draw(pipeline), spellings, strategies),
    }
    if draw(st.booleans()):
        document["attributes"] = draw(_ATTRIBUTES)
    if draw(st.booleans()):
        document["storage_transformers"] = ()
    if draw(st.booleans()):
        document["dimension_names"] = draw(st.tuples(*(st.none() | _TEXT for _ in shape)))
    document.update(draw(_extra_fields(ArrayExtraFieldDefinition)))
    problems = read_array_metadata_v3(document, context=scope_of(strategies)).problems
    if len(problems) != 0:
        msg = f"{document!r}, drawn as an array document, is refused: {problems!r}"
        raise ValueError(msg)
    return cast("ZarrV3ArrayMetadataJSON", document)


def check_arrays_can_be_drawn(strategies: Strategies) -> None:
    """A `LookupError` when the strategies hold nothing to draw some member of an array document from."""
    scope = scope_of(strategies)
    missing = [
        what
        for what, held in (
            ("data type", scope.data_types),
            ("chunk grid", scope.chunk_grids),
            ("chunk key encoding", scope.chunk_key_encodings),
        )
        if len(held) == 0
    ]
    if pipelines_for(NOTHING_KNOWN, strategies) is None:
        missing.append("array -> bytes codec")
    if len(missing) != 0:
        msg = f"no array document can be drawn: `strategies` hold no {', and no '.join(missing)}"
        raise LookupError(msg)


@st.composite
def group_document(draw: st.DrawFn) -> ZarrV3GroupMetadataJSON:
    """A group document, with no consolidated metadata."""
    document: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}
    if draw(st.booleans()):
        document["attributes"] = draw(_ATTRIBUTES)
    document.update(draw(_extra_fields(GroupExtraFieldDefinition)))
    return cast("ZarrV3GroupMetadataJSON", document)


_NODE_NAMES: Final = (
    st.text(CHARACTERS, min_size=1, max_size=8)
    | st.sampled_from(("_a", "..a", "a.json", "__a", ".", "..", "zarr.json"))
).filter(lambda name: len(name_faults(name)) == 0)
"""Node names: any text, and those beside the names the spec reserves, which the filter keeps out."""


@st.composite
def _node_paths(draw: st.DrawFn, max_nodes: int) -> list[tuple[str, bool]]:
    """Paths below a group, each with whether the node there is a group: each path's parent is listed before it, and is a group."""
    paths: dict[str, bool] = {}
    for _ in range(draw(st.integers(0, max_nodes))):
        parent = draw(
            st.sampled_from(["", *(path for path, is_group in paths.items() if is_group)])
        )
        name = draw(_NODE_NAMES)
        path = name if parent == "" else f"{parent}/{name}"
        if path not in paths:
            paths[path] = draw(st.booleans())
    return list(paths.items())


@st.composite
def consolidated_group_document(
    draw: st.DrawFn,
    shapes: SearchStrategy[tuple[int, ...]],
    max_nodes: int,
    spellings: Spellings,
    strategies: Strategies,
) -> ZarrV3GroupMetadataJSON:
    """A group document that lists the nodes below it; a `ValueError` when its definitions refuse the listing."""
    metadata: dict[str, Mapping[str, Any]] = {}
    for path, is_group in draw(_node_paths(max_nodes)):
        metadata[path] = draw(
            group_document() if is_group else array_document(shapes, spellings, strategies)
        )
    listing: ZarrV3ConsolidatedMetadataJSON = {
        "kind": "inline",
        "must_understand": False,
        "metadata": cast("Any", metadata),
    }
    problems = read_consolidated_metadata_v3(listing, context=scope_of(strategies)).problems
    if len(problems) != 0:
        msg = f"{listing!r}, drawn as consolidated metadata, is refused: {problems!r}"
        raise ValueError(msg)
    document = {**draw(group_document()), ZARR_V3_CONSOLIDATED_METADATA_KEY: listing}
    return cast("ZarrV3GroupMetadataJSON", document)


__all__ = [
    "SHAPES",
    "array_document",
    "check_arrays_can_be_drawn",
    "consolidated_group_document",
    "group_document",
    "spelled_pipeline",
]
