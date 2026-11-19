"""`canonicalize`: each model with its document in its simplest equivalent spelling.

Two models are equal when their documents mean the same thing, and
`to_json` writes each document as it was given. `canonicalize` returns the
one document that every model equal to a given one writes, apart from the
order of the members of an object, which a `JSONWriter` sorts.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NotRequired, TypeVar, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from typing_extensions import TypedDict, TypeIs

from zarr_metadata._json import json_text
from zarr_metadata.definition._definition import canonicalize
from zarr_metadata.model import (
    JSONWriter,
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
)
from zarr_metadata.testing.strategies import (
    Spellings,
    array_documents,
    consolidated_group_documents,
    group_documents,
)
from zarr_metadata.v2.definition import CORE_V2_CONTEXT, ZarrV2Context
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS_CONTEXT,
    ZARR_PYTHON_CONTEXT,
    ArrayExtraFieldDefinition,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    GroupExtraFieldDefinition,
    StorageTransformerDefinition,
    ZarrV3Context,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata import JSONValue

Model = (
    ZarrV2ArrayMetadata
    | ZarrV2ConsolidatedMetadata
    | ZarrV2GroupMetadata
    | ZarrV3ArrayMetadata
    | ZarrV3ConsolidatedMetadata
    | ZarrV3GroupMetadata
)


class AcmeLevelConfiguration(TypedDict, closed=True):
    level: NotRequired[int]


class AcmeNoteConfiguration(TypedDict, closed=True):
    must_understand: NotRequired[bool]
    level: NotRequired[int]


C = TypeVar("C", AcmeLevelConfiguration, AcmeNoteConfiguration)


def _without_level_0(configuration: C) -> C:
    """Without a `level` of 0, which is what both definitions here do without a `level`."""
    if configuration.get("level") != 0:
        return configuration
    return cast("C", {key: value for key, value in configuration.items() if key != "level"})


_V3_CONTEXT = CORE_AND_EXTENSIONS_CONTEXT.union(
    ZarrV3Context(
        array_extra_fields=[
            ArrayExtraFieldDefinition(
                name="note", configuration=AcmeNoteConfiguration, canonical=_without_level_0
            )
        ],
        group_extra_fields=[
            GroupExtraFieldDefinition(
                name="note", configuration=AcmeNoteConfiguration, canonical=_without_level_0
            )
        ],
        storage_transformers=[
            StorageTransformerDefinition(
                name="acme.st", configuration=AcmeLevelConfiguration, canonical=_without_level_0
            )
        ],
    )
)
"""The scope the v3 documents here are read in. It is not the default scope: it also claims an extra field `note` and a storage transformer `acme.st`, each with a `canonical`."""

_V2_CONTEXT = CORE_V2_CONTEXT.union(ZarrV2Context())
"""The scope the v2 documents here are read in: the default scope as another object, which tells whether a model kept the scope it was read in."""

_LITTLE: dict[str, Any] = {"name": "bytes", "configuration": {"endian": "little"}}
_BIG: dict[str, Any] = {"name": "bytes", "configuration": {"endian": "big"}}
_NOSHUFFLE = {"cname": "lz4", "clevel": 5, "shuffle": "noshuffle", "blocksize": 0}
_WAIVED = {"name": "acme.note", "must_understand": False, "configuration": {"b": 1, "a": 2}}

_V3_ARRAY: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [4],
    "data_type": "uint8",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
    "chunk_key_encoding": {"name": "default"},
    "fill_value": 0,
    "codecs": [{"name": "bytes"}],
}
"""A v3 array document in its simplest spelling."""

_V3_ARRAY_SPELLED_OUT: dict[str, Any] = {
    "attributes": {},
    "zarr_format": 3,
    "node_type": "array",
    "shape": [4],
    "data_type": {"name": "uint8", "configuration": {}},
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
    "fill_value": 0,
    "codecs": [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [2],
                "index_location": "end",
                "codecs": [_BIG, {"name": "blosc", "configuration": {**_NOSHUFFLE, "typesize": 4}}],
                "index_codecs": [_LITTLE, {"name": "crc32c", "configuration": {}}],
            },
        }
    ],
    "storage_transformers": [],
    "dimension_names": [None],
    "acme": _WAIVED,
}
"""A v3 array document that says in more words what `_V3_ARRAY_SIMPLEST` says."""

_V3_ARRAY_SIMPLEST: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [4],
    "data_type": "uint8",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
    "chunk_key_encoding": {"name": "default"},
    "fill_value": 0,
    "codecs": [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [2],
                "codecs": [{"name": "bytes"}, {"name": "blosc", "configuration": _NOSHUFFLE}],
                "index_codecs": [_LITTLE, {"name": "crc32c"}],
            },
        }
    ],
    "dimension_names": [None],
    "acme": _WAIVED,
}

_V3_FLOATS: dict[str, Any] = {**_V3_ARRAY, "data_type": "float32", "codecs": [_LITTLE]}

_V3_UNCLAIMED: dict[str, Any] = {
    **_V3_ARRAY,
    "data_type": {"name": "acme.t", "configuration": {"bits": 8, "signed": False}},
    "fill_value": "0x00",
    "codecs": [{"name": "acme.codec", "configuration": {"level": 1}}, _LITTLE],
    "attributes": {"b": [1.0, {}], "a": None},
}
"""A v3 array document that holds names nothing in scope claims. Their configurations are kept as written, since only their own definitions could simplify them, and so is the `endian` of the `bytes` codec, which receives values nothing is known of."""

_V3_UNCLAIMED_ENVELOPES: dict[str, Any] = {
    **_V3_ARRAY,
    "data_type": {"name": "acme.t", "configuration": {}},
    "fill_value": "0x00",
    "codecs": [
        {"name": "acme.codec", "configuration": {}},
        "acme.other",
        {"name": "acme.third", "must_understand": True, "configuration": {"b": 1, "a": {}}},
        _LITTLE,
    ],
}
"""A v3 array document that holds names nothing in scope claims, each in an envelope that has a shorter form."""

_V3_UNCLAIMED_SHORTEST: dict[str, Any] = {
    **_V3_ARRAY,
    "data_type": "acme.t",
    "fill_value": "0x00",
    "codecs": [
        {"name": "acme.codec"},
        {"name": "acme.other"},
        {"name": "acme.third", "configuration": {"b": 1, "a": {}}},
        _LITTLE,
    ],
}


def _rectilinear(*chunk_shapes: list[int] | list[list[int]]) -> dict[str, Any]:
    return {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": list(chunk_shapes)},
    }


_V3_GROUP: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}


def _v3_listing(*documents: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "inline",
        "must_understand": False,
        "metadata": {f"n{index}": document for index, document in enumerate(documents)},
    }


_V2_ARRAY: dict[str, Any] = {
    "zarr_format": 2,
    "shape": [4],
    "chunks": [2],
    "dtype": "|i1",
    "compressor": {"id": "zstd", "level": 0, "checksum": False},
    "fill_value": 0,
    "order": "C",
    "filters": [{"id": "delta", "dtype": "|i1", "astype": "|i1"}],
}
"""A v2 array document in its canonical spelling, in which a codec writes each numcodecs default."""

_V2_ARRAY_SPELLED_OTHERWISE: dict[str, Any] = {
    **_V2_ARRAY,
    "dtype": "<i1",
    "compressor": {"id": "zstd", "level": 0},
    "filters": [{"id": "delta", "dtype": "<i1"}],
    "dimension_separator": ".",
    "attributes": {},
}

_V2_FLOATS: dict[str, Any] = {
    **_V2_ARRAY,
    "dtype": "<f4",
    "compressor": None,
    "filters": None,
    "fill_value": "NaN",
    "dimension_separator": "/",
    "attributes": {"a": 1},
}
"""A v2 array document with nothing to simplify: a `/` separator and attributes say something."""

_V2_GROUP: dict[str, Any] = {"zarr_format": 2}


def _v2_array_entry(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "attributes"}


@pytest.mark.parametrize(
    ("model", "document", "canonical"),
    [
        (ZarrV3ArrayMetadata, _V3_ARRAY, _V3_ARRAY),
        (ZarrV3ArrayMetadata, _V3_ARRAY_SPELLED_OUT, _V3_ARRAY_SIMPLEST),
        (
            ZarrV3ArrayMetadata,
            {**_V3_FLOATS, "fill_value": "0x7fc00000"},
            {**_V3_FLOATS, "fill_value": "NaN"},
        ),
        (ZarrV3ArrayMetadata, _V3_UNCLAIMED, _V3_UNCLAIMED),
        (ZarrV3ArrayMetadata, _V3_UNCLAIMED_ENVELOPES, _V3_UNCLAIMED_SHORTEST),
        (
            ZarrV3ArrayMetadata,
            {**_V3_ARRAY, "shape": [6], "chunk_grid": _rectilinear([2, 2, 2])},
            {**_V3_ARRAY, "shape": [6], "chunk_grid": _rectilinear([[2, 3]])},
        ),
        (
            ZarrV3ArrayMetadata,
            {
                **_V3_ARRAY,
                "storage_transformers": [{"name": "acme.st", "configuration": {"level": 0}}],
            },
            {**_V3_ARRAY, "storage_transformers": [{"name": "acme.st"}]},
        ),
        (ZarrV3ArrayMetadata, {**_V3_ARRAY, "note": {"level": 0}}, {**_V3_ARRAY, "note": {}}),
        (ZarrV3GroupMetadata, _V3_GROUP, _V3_GROUP),
        (
            ZarrV3GroupMetadata,
            {"attributes": {}, **_V3_GROUP, "acme": _WAIVED},
            {**_V3_GROUP, "acme": _WAIVED},
        ),
        (
            ZarrV3GroupMetadata,
            {**_V3_GROUP, "attributes": {"a": 1}},
            {**_V3_GROUP, "attributes": {"a": 1}},
        ),
        (ZarrV3GroupMetadata, {**_V3_GROUP, "note": {"level": 0}}, {**_V3_GROUP, "note": {}}),
        # A group does not read the documents its consolidated metadata lists.
        (
            ZarrV3GroupMetadata,
            {**_V3_GROUP, "consolidated_metadata": _v3_listing(_V3_ARRAY_SPELLED_OUT)},
            {**_V3_GROUP, "consolidated_metadata": _v3_listing(_V3_ARRAY_SPELLED_OUT)},
        ),
        (
            ZarrV3ConsolidatedMetadata,
            _v3_listing(_V3_ARRAY_SPELLED_OUT, {**_V3_GROUP, "attributes": {}}),
            _v3_listing(_V3_ARRAY_SIMPLEST, _V3_GROUP),
        ),
        (ZarrV2ArrayMetadata, _V2_ARRAY, _V2_ARRAY),
        (ZarrV2ArrayMetadata, _V2_ARRAY_SPELLED_OTHERWISE, _V2_ARRAY),
        (ZarrV2ArrayMetadata, _V2_FLOATS, _V2_FLOATS),
        (ZarrV2ArrayMetadata, {**_V2_FLOATS, "fill_value": 0}, {**_V2_FLOATS, "fill_value": 0.0}),
        (
            ZarrV2ArrayMetadata,
            {**_V2_FLOATS, "fill_value": 1e39},
            {**_V2_FLOATS, "fill_value": "Infinity"},
        ),
        (ZarrV2GroupMetadata, _V2_GROUP, _V2_GROUP),
        (ZarrV2GroupMetadata, {**_V2_GROUP, "attributes": {}}, _V2_GROUP),
        (
            ZarrV2GroupMetadata,
            {**_V2_GROUP, "attributes": {"a": 1}},
            {**_V2_GROUP, "attributes": {"a": 1}},
        ),
        (
            ZarrV2ConsolidatedMetadata,
            {
                "zarr_consolidated_format": 1,
                "metadata": {
                    ".zgroup": _V2_GROUP,
                    ".zattrs": {},
                    "a/.zarray": _v2_array_entry(_V2_ARRAY_SPELLED_OTHERWISE),
                    "a/.zattrs": {"x": 1},
                    "b/.zarray": _v2_array_entry(_V2_ARRAY_SPELLED_OTHERWISE),
                    "b/.zattrs": {},
                    "c/.zattrs": {},
                    "notes": [1],
                },
            },
            {
                "zarr_consolidated_format": 1,
                "metadata": {
                    ".zgroup": _V2_GROUP,
                    "a/.zarray": _V2_ARRAY,
                    "a/.zattrs": {"x": 1},
                    "b/.zarray": _V2_ARRAY,
                    # No node is read from these entries, so they are kept as written.
                    "c/.zattrs": {},
                    "notes": [1],
                },
            },
        ),
    ],
    ids=[
        "v3-array-simplest",
        "v3-array-spelled-out",
        "v3-array-fill-value",
        "v3-array-unclaimed",
        "v3-array-unclaimed-envelopes",
        "v3-array-chunk-grid",
        "v3-array-storage-transformer",
        "v3-array-claimed-extra-field",
        "v3-group-simplest",
        "v3-group-empty-attributes",
        "v3-group-attributes",
        "v3-group-claimed-extra-field",
        "v3-group-consolidated",
        "v3-consolidated",
        "v2-array-canonical",
        "v2-array-spelled-otherwise",
        "v2-array-nothing-to-simplify",
        "v2-array-fill-value",
        "v2-array-fill-value-out-of-range",
        "v2-group-simplest",
        "v2-group-empty-attributes",
        "v2-group-attributes",
        "v2-consolidated",
    ],
)
def test_a_model_canonicalizes_to_the_simplest_spelling_of_its_document(
    model: type[Model], document: dict[str, Any], canonical: dict[str, Any]
) -> None:
    """`canonicalize` returns an equal model of the same class, in the same scope, whose document is in its simplest equivalent spelling, with the members that remain in the order they were written. A document already in that spelling is returned as it is."""
    read = _read(model, document)
    result = read.canonicalize()
    assert json.dumps(result.to_json()) == json.dumps(canonical)
    assert type(result) is type(read)
    # Neither scope here is the default one, so a model that lost its scope would show.
    assert result.context is (_V2_CONTEXT if _is_v2(model) else _V3_CONTEXT)
    assert result == read
    assert hash(result) == hash(read)
    assert json.dumps(result.canonicalize().to_json()) == json.dumps(canonical)


def _is_v2(
    model: type[Model],
) -> TypeIs[type[ZarrV2ArrayMetadata | ZarrV2ConsolidatedMetadata | ZarrV2GroupMetadata]]:
    return issubclass(model, (ZarrV2ArrayMetadata, ZarrV2ConsolidatedMetadata, ZarrV2GroupMetadata))


def _read(model: type[Model], document: Mapping[str, Any]) -> Model:
    """The model of `document`, read in `_V2_CONTEXT` or `_V3_CONTEXT`."""
    if _is_v2(model):
        return model.from_json(document, context=_V2_CONTEXT)
    return model.from_json(document, context=_V3_CONTEXT)


def _reordered(document: Mapping[str, Any]) -> dict[str, Any]:
    """`document` with the members of every object it holds, at any depth, in the reverse order. The items of an array keep their order."""

    def reverse(value: object) -> object:
        if isinstance(value, dict):
            held = cast("dict[str, object]", value)
            return {key: reverse(held[key]) for key in reversed(held)}
        if isinstance(value, list):
            return [reverse(item) for item in cast("list[object]", value)]
        return value

    return cast("dict[str, Any]", reverse(dict(document)))


@pytest.mark.parametrize(
    ("model", "left", "right"),
    [
        (ZarrV3ArrayMetadata, _V3_ARRAY_SPELLED_OUT, _V3_ARRAY_SIMPLEST),
        (ZarrV3ArrayMetadata, _V3_ARRAY_SPELLED_OUT, _reordered(_V3_ARRAY_SIMPLEST)),
        (ZarrV3ArrayMetadata, _V3_UNCLAIMED_ENVELOPES, _reordered(_V3_UNCLAIMED_SHORTEST)),
        (
            ZarrV3ArrayMetadata,
            {**_V3_UNCLAIMED, "storage_transformers": []},
            _reordered(_V3_UNCLAIMED),
        ),
        (ZarrV3ArrayMetadata, {**_V3_ARRAY, "codecs": [_LITTLE]}, {**_V3_ARRAY, "codecs": [_BIG]}),
        (ZarrV3GroupMetadata, {**_V3_GROUP, "attributes": {}}, _V3_GROUP),
        (
            ZarrV3ConsolidatedMetadata,
            _v3_listing(_V3_ARRAY_SPELLED_OUT, _V3_GROUP),
            _v3_listing(_reordered(_V3_ARRAY_SIMPLEST), {**_V3_GROUP, "attributes": {}}),
        ),
        (ZarrV2ArrayMetadata, _V2_ARRAY_SPELLED_OTHERWISE, _reordered(_V2_ARRAY)),
        (ZarrV2GroupMetadata, {**_V2_GROUP, "attributes": {}}, _V2_GROUP),
    ],
)
def test_equal_models_canonicalize_to_one_document(
    model: type[Model], left: dict[str, Any], right: dict[str, Any]
) -> None:
    """Models that are equal give the same canonical document, apart from the order of the members of an object: a `JSONWriter` that sorts them writes the same bytes for each."""
    models = [_read(model, document) for document in (left, right)]
    assert models[0] == models[1]
    canonical = [each.canonicalize() for each in models]
    assert json_text(_json_of(canonical[0])) == json_text(_json_of(canonical[1]))
    assert _stored(canonical[0]) == _stored(canonical[1])


@pytest.mark.parametrize(
    ("model", "left", "right"),
    [
        (
            ZarrV3ArrayMetadata,
            {**_V3_ARRAY, "data_type": "uint16", "codecs": [_LITTLE]},
            {**_V3_ARRAY, "data_type": "uint16", "codecs": [_BIG]},
        ),
        (ZarrV3ArrayMetadata, _V3_ARRAY, {**_V3_ARRAY, "dimension_names": [None]}),
        (
            ZarrV3ArrayMetadata,
            {**_V3_FLOATS, "fill_value": 0.0},
            {**_V3_FLOATS, "fill_value": -0.0},
        ),
        (ZarrV3GroupMetadata, _V3_GROUP, {**_V3_GROUP, "attributes": {"a": 1}}),
        (ZarrV2ArrayMetadata, _V2_ARRAY, {**_V2_ARRAY, "dimension_separator": "/"}),
        (ZarrV2ArrayMetadata, {**_V2_ARRAY, "filters": None}, {**_V2_ARRAY, "filters": []}),
    ],
)
def test_models_that_differ_canonicalize_to_different_documents(
    model: type[Model], left: dict[str, Any], right: dict[str, Any]
) -> None:
    """Only spellings that mean the same thing are reduced to one: models that are not equal keep documents that differ."""
    models = [_read(model, document) for document in (left, right)]
    assert models[0] != models[1]
    written = [json_text(_json_of(each.canonicalize())) for each in models]
    assert written[0] != written[1]


def _json_of(model: Model) -> JSONValue:
    document: Any = model.to_json()
    return document


def _stored(model: Model) -> object:
    """The bytes a store holds for `model`, with the members of every object sorted. A v3 listing is stored in the document of a group."""
    if isinstance(model, ZarrV3ConsolidatedMetadata):
        group = ZarrV3GroupMetadata.from_json(_V3_GROUP, context=model.context)
        model = group.replace(consolidated_metadata=model)
    return model.to_key_value(JSONWriter(sort_keys=True))


def _assert_canonicalizes(
    model: ZarrV3ArrayMetadata | ZarrV3GroupMetadata | ZarrV3ConsolidatedMetadata,
) -> None:
    """`model` equals its canonical model, which is in the same scope, has nothing left to simplify and reads as the same model from the JSON text a store holds. The document of `model` with the members of every object in the reverse order gives the same canonical document."""
    canonical = model.canonicalize()
    assert canonical == model
    assert canonical.context is model.context
    assert hash(canonical) == hash(model)
    written = json.dumps(canonical.to_json())
    assert json.dumps(canonical.canonicalize().to_json()) == written
    assert type(model).from_json(json.loads(written), context=ZARR_PYTHON_CONTEXT) == model
    if not isinstance(canonical, ZarrV3ConsolidatedMetadata):
        document = cast("Mapping[str, Any]", _json_of(canonical))
        for member in ("attributes", "storage_transformers"):
            assert member not in document or len(document[member]) != 0
    reordered = type(model).from_json(
        _reordered(json.loads(json.dumps(model.to_json()))), context=ZARR_PYTHON_CONTEXT
    )
    assert reordered == model
    assert json_text(_json_of(reordered.canonicalize())) == json_text(_json_of(canonical))


@pytest.mark.parametrize("spellings", ["all", "canonical"])
@given(data=st.data())
def test_every_array_document_canonicalizes(spellings: Spellings, data: st.DataObject) -> None:
    document = data.draw(array_documents(spellings=spellings))
    _assert_canonicalizes(ZarrV3ArrayMetadata.from_json(document, context=ZARR_PYTHON_CONTEXT))


@given(document=array_documents(spellings="all"))
def test_every_array_document_canonicalizes_as_its_declarations_do(
    document: Mapping[str, Any],
) -> None:
    """A document, and the same document with each declaration in its own simplest spelling, are one array, and give one canonical document."""
    simpler = dict(document)
    for member, kind in (
        ("data_type", DataTypeDefinition),
        ("chunk_grid", ChunkGridDefinition),
        ("chunk_key_encoding", ChunkKeyEncodingDefinition),
    ):
        simpler[member], problems = canonicalize(document[member], kind, ZARR_PYTHON_CONTEXT)
        assert problems == ()
    simpler["codecs"] = [
        canonicalize(codec, CodecDefinition, ZARR_PYTHON_CONTEXT)[0] for codec in document["codecs"]
    ]
    models = [
        ZarrV3ArrayMetadata.from_json(each, context=ZARR_PYTHON_CONTEXT)
        for each in (document, simpler)
    ]
    assert models[0] == models[1]
    canonical = [json_text(_json_of(each.canonicalize())) for each in models]
    assert canonical[0] == canonical[1]


@given(document=group_documents())
def test_every_group_document_canonicalizes(document: Mapping[str, Any]) -> None:
    _assert_canonicalizes(ZarrV3GroupMetadata.from_json(document, context=ZARR_PYTHON_CONTEXT))


@given(document=consolidated_group_documents())
def test_every_consolidated_group_document_canonicalizes(
    document: Mapping[str, Any],
) -> None:
    group = ZarrV3GroupMetadata.from_json(document, context=ZARR_PYTHON_CONTEXT)
    _assert_canonicalizes(group)
    listing = group.consolidated()
    assert listing is not None
    _assert_canonicalizes(listing)
    # The listing in its simplest spelling can be written back into the group.
    _assert_canonicalizes(group.replace(consolidated_metadata=listing.canonicalize()))
