"""The document strategies against the readers, and against what they ought to reach.

Every drawn document is read in `ZARR_PYTHON_CONTEXT`, as written and
after a JSON round trip, and must read without a problem. Reach is checked
per definition and per member, so a document strategy that drew valid
metadata from only part of what its parts draw would fail.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest
from hypothesis import find, given, settings
from hypothesis import strategies as st

from zarr_metadata.definition import AcceptedDeclaration, Definition, resolve
from zarr_metadata.definition._definition import canonicalize
from zarr_metadata.model import (
    read_array_metadata_v3,
    read_consolidated_metadata_v3,
    read_group_metadata_v3,
)
from zarr_metadata.testing.strategies import (
    DEFAULT_STRATEGIES,
    ChunkGridStrategy,
    CodecStrategy,
    DataTypeStrategy,
    Spellings,
    Strategies,
    array_documents,
    consolidated_group_documents,
    group_documents,
    pipelines,
)
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID
from zarr_metadata.v3.codec.bytes import BYTES_CODEC
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE
from zarr_metadata.v3.definition import (
    ZARR_PYTHON_CONTEXT,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_REACH = settings(max_examples=2000, database=None)
_DEFINITIONS = sorted(DEFAULT_STRATEGIES, key=lambda definition: definition.name)
_SPELLINGS: tuple[Spellings, ...] = ("all", "canonical")
_SHORT_REPR = 10_000
"""Well short of the 30,000 characters past which Hypothesis warns about a strategy's repr."""


def _without(kind: type[Definition[Any]]) -> Strategies:
    """The default strategies, with none for a definition of `kind`."""
    return {d: s for d, s in DEFAULT_STRATEGIES.items() if not isinstance(d, kind)}


def _id(value: object) -> str:
    return value.name if isinstance(value, Definition) else str(value)


def _name(declaration: Any) -> str:  # noqa: ANN401
    return declaration if isinstance(declaration, str) else declaration["name"]


def _read_array(document: object) -> None:
    for written in (document, json.loads(json.dumps(document))):
        reading = read_array_metadata_v3(written, context=ZARR_PYTHON_CONTEXT)
        assert reading.problems == ()
        assert reading.metadata is not None
        assert reading.metadata.must_understand_fields == {}


def _read_group(document: object) -> None:
    for written in (document, json.loads(json.dumps(document))):
        reading = read_group_metadata_v3(written, context=ZARR_PYTHON_CONTEXT)
        assert reading.problems == ()
        assert reading.metadata is not None
        assert reading.metadata.must_understand_fields == {}


@pytest.mark.parametrize("spellings", _SPELLINGS)
@given(data=st.data())
def test_array_documents_read(spellings: Spellings, data: st.DataObject) -> None:
    _read_array(data.draw(array_documents(spellings=spellings)))


@given(document=group_documents())
def test_group_documents_read(document: Mapping[str, Any]) -> None:
    assert "consolidated_metadata" not in document
    _read_group(document)


@pytest.mark.parametrize("spellings", _SPELLINGS)
@given(data=st.data())
def test_consolidated_group_documents_read(spellings: Spellings, data: st.DataObject) -> None:
    document = data.draw(consolidated_group_documents(spellings=spellings))
    _read_group(document)
    listing: Any = document["consolidated_metadata"]
    reading = read_consolidated_metadata_v3(listing, context=ZARR_PYTHON_CONTEXT)
    assert reading.problems == ()
    for path, node in listing["metadata"].items():
        parent = path.rpartition("/")[0]
        assert parent == "" or listing["metadata"][parent]["node_type"] == "group"
        assert node["zarr_format"] == 3


@given(document=array_documents(shapes=st.just((3, 4))))
def test_array_documents_take_their_shape(document: Mapping[str, Any]) -> None:
    assert document["shape"] == (3, 4)


@given(document=consolidated_group_documents(max_nodes=2))
def test_consolidated_group_documents_take_their_max_nodes(document: Mapping[str, Any]) -> None:
    assert len(document["consolidated_metadata"]["metadata"]) <= 2


def _assert_canonical(declaration: object, kind: type[Definition[Any]]) -> None:
    canonical, problems = canonicalize(declaration, kind, ZARR_PYTHON_CONTEXT)
    assert problems == ()
    assert json.dumps(canonical) == json.dumps(declaration)


@given(document=array_documents(spellings="canonical"))
def test_canonical_array_documents_write_each_declaration_canonically(
    document: Mapping[str, Any],
) -> None:
    _assert_canonical(document["data_type"], DataTypeDefinition)
    _assert_canonical(document["chunk_grid"], ChunkGridDefinition)
    _assert_canonical(document["chunk_key_encoding"], ChunkKeyEncodingDefinition)
    for codec in document["codecs"]:
        _assert_canonical(codec, CodecDefinition)


_CANONICAL_PIPELINES = pipelines(spellings="canonical")
assert _CANONICAL_PIPELINES is not None


@given(pipeline=_CANONICAL_PIPELINES)
def test_canonical_pipelines_write_each_codec_canonically(pipeline: tuple[Any, ...]) -> None:
    for codec in pipeline:
        _assert_canonical(codec, CodecDefinition)


def _is_spelled_otherwise(document: Mapping[str, Any]) -> bool:
    return any(
        json.dumps(canonicalize(codec, CodecDefinition, ZARR_PYTHON_CONTEXT)[0])
        != json.dumps(codec)
        for codec in document["codecs"]
    )


def test_array_documents_reach_a_codec_not_in_its_canonical_spelling() -> None:
    find(array_documents(), _is_spelled_otherwise, settings=_REACH)


def _is_data_type(document: Mapping[str, Any], definition: Definition[Any]) -> bool:
    read = resolve(document["data_type"], DataTypeDefinition, ZARR_PYTHON_CONTEXT)[0]
    return isinstance(read, AcceptedDeclaration) and read.definition is definition


def _reach_of(definition: Definition[Any]) -> Callable[[Mapping[str, Any]], bool]:
    if isinstance(definition, DataTypeDefinition):
        return lambda document: _is_data_type(document, definition)
    if isinstance(definition, ChunkGridDefinition):
        return lambda document: _name(document["chunk_grid"]) == definition.name
    if isinstance(definition, ChunkKeyEncodingDefinition):
        return lambda document: _name(document["chunk_key_encoding"]) == definition.name
    return lambda document: any(_name(codec) == definition.name for codec in document["codecs"])


@pytest.mark.parametrize("definition", _DEFINITIONS, ids=_id)
def test_array_documents_reach_every_definition(definition: Definition[Any]) -> None:
    find(array_documents(), _reach_of(definition), settings=_REACH)


_ARRAY_REACH: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "no axes": lambda d: d["shape"] == (),
    "a zero-length axis": lambda d: 0 in d["shape"],
    "attributes": lambda d: "attributes" in d,
    "no attributes": lambda d: "attributes" not in d,
    "storage_transformers": lambda d: "storage_transformers" in d,
    "no storage_transformers": lambda d: "storage_transformers" not in d,
    "dimension_names": lambda d: len(d["shape"]) > 0 and "dimension_names" in d,
    "no dimension_names": lambda d: "dimension_names" not in d,
    "a member the spec does not define": lambda d: len(d.keys() - _ARRAY_MEMBERS) > 0,
    "a shard over a rectilinear grid": lambda d: (
        _name(d["chunk_grid"]) == "rectilinear"
        and any(_name(codec) == SHARDING_INDEXED_CODEC.name for codec in d["codecs"])
    ),
    "a pipeline of several codecs": lambda d: len(d["codecs"]) > 1,
}
_ARRAY_MEMBERS = {
    "zarr_format",
    "node_type",
    "shape",
    "data_type",
    "chunk_grid",
    "chunk_key_encoding",
    "fill_value",
    "codecs",
    "attributes",
    "storage_transformers",
    "dimension_names",
}


@pytest.mark.parametrize("target", _ARRAY_REACH)
def test_array_documents_reach(target: str) -> None:
    find(array_documents(), _ARRAY_REACH[target], settings=_REACH)


_GROUP_REACH: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "attributes": lambda d: "attributes" in d,
    "no attributes": lambda d: "attributes" not in d,
    "a member the spec does not define": lambda d: (
        len(d.keys() - {"zarr_format", "node_type", "attributes"}) > 0
    ),
}


@pytest.mark.parametrize("target", _GROUP_REACH)
def test_group_documents_reach(target: str) -> None:
    find(group_documents(), _GROUP_REACH[target], settings=_REACH)


def _listed(document: Mapping[str, Any]) -> Mapping[str, Any]:
    return document["consolidated_metadata"]["metadata"]


_LISTING_REACH: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "an empty listing": lambda d: len(_listed(d)) == 0,
    "an array": lambda d: any(node["node_type"] == "array" for node in _listed(d).values()),
    "a group": lambda d: any(node["node_type"] == "group" for node in _listed(d).values()),
    "a node below a listed group": lambda d: any("/" in path for path in _listed(d)),
}


@pytest.mark.parametrize("target", _LISTING_REACH)
def test_consolidated_group_documents_reach(target: str) -> None:
    find(consolidated_group_documents(), _LISTING_REACH[target], settings=_REACH)


def test_an_added_strategy_is_drawn_in_a_document() -> None:
    """A strategy joined to the defaults reaches the array document drawn from them."""
    level_three = CodecStrategy(configurations=lambda _chunk, _strategies: st.just({"level": 3}))
    strategies = DEFAULT_STRATEGIES | {GZIP_CODEC: level_three}
    document = find(
        array_documents(strategies=strategies),
        lambda d: any(_name(codec) == GZIP_CODEC.name for codec in d["codecs"]),
        settings=_REACH,
    )
    codecs: list[Any] = [codec for codec in document["codecs"] if _name(codec) == GZIP_CODEC.name]
    assert codecs[0]["configuration"] == {"level": 3}


def test_reprs_stay_short() -> None:
    """However the strategies are passed: a plain mapping is as long as every strategy in it."""
    plain: Strategies = dict(DEFAULT_STRATEGIES)
    assert len(repr(array_documents(strategies=plain))) < _SHORT_REPR
    assert len(repr(consolidated_group_documents(strategies=plain))) < _SHORT_REPR


def test_array_documents_without_a_data_type_is_a_lookup_error() -> None:
    with pytest.raises(LookupError, match="no data type"):
        array_documents(strategies=_without(DataTypeDefinition))


def test_array_documents_without_a_chunk_grid_is_a_lookup_error() -> None:
    with pytest.raises(LookupError, match="no chunk grid"):
        array_documents(strategies=_without(ChunkGridDefinition))


def test_array_documents_without_a_chunk_key_encoding_is_a_lookup_error() -> None:
    with pytest.raises(LookupError, match="no chunk key encoding"):
        array_documents(strategies=_without(ChunkKeyEncodingDefinition))


def test_array_documents_without_an_array_bytes_codec_is_a_lookup_error() -> None:
    with pytest.raises(LookupError, match=r"no array -> bytes codec"):
        array_documents(strategies=_without(CodecDefinition))


def test_consolidated_group_documents_without_a_chunk_grid_is_a_lookup_error() -> None:
    with pytest.raises(LookupError, match="no chunk grid"):
        consolidated_group_documents(strategies=_without(ChunkGridDefinition))


def test_a_chunk_grid_that_does_not_fit_the_shape_is_a_value_error() -> None:
    misfit = ChunkGridStrategy(configurations=lambda _shape, _s: st.just({"chunk_shape": (1, 1)}))
    strategies = {
        **{d: s for d, s in DEFAULT_STRATEGIES.items() if not isinstance(d, ChunkGridDefinition)},
        REGULAR_CHUNK_GRID: misfit,
    }
    with pytest.raises(ValueError, match="drawn over the shape"):
        find(array_documents(shapes=st.just((3,)), strategies=strategies), lambda _d: True)


def test_a_fill_value_its_data_type_refuses_is_a_value_error() -> None:
    wide = DataTypeStrategy(
        configurations=cast("DataTypeStrategy", DEFAULT_STRATEGIES[INT8_DATA_TYPE]).configurations,
        fill_values=lambda _configuration, _s: st.just(300),
    )
    strategies = {
        **{d: s for d, s in DEFAULT_STRATEGIES.items() if not isinstance(d, DataTypeDefinition)},
        INT8_DATA_TYPE: wide,
    }
    with pytest.raises(ValueError, match="drawn as an array document"):
        find(array_documents(strategies=strategies), lambda _d: True)


_NOTHING_ENCODES_STRINGS: Strategies = {
    d: s
    for d, s in DEFAULT_STRATEGIES.items()
    if isinstance(d, (ChunkGridDefinition, ChunkKeyEncodingDefinition))
    or d in (BYTES_CODEC, STRING_DATA_TYPE, INT8_DATA_TYPE)
}


@given(document=array_documents(strategies=_NOTHING_ENCODES_STRINGS))
def test_a_data_type_no_codec_takes_is_not_drawn(document: Mapping[str, Any]) -> None:
    """Without `vlen-utf8`, nothing encodes a `string`, so no document holds one."""
    assert document["data_type"] == "int8"
