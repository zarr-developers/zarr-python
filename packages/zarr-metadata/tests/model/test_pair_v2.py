"""A v2 model is its document and the scope it was read in, as the v3 models are."""

from __future__ import annotations

import copy
import dataclasses
import pickle
from typing import Any, cast

import pytest

from zarr_metadata._sentinel import UNSET
from zarr_metadata.model import (
    MetadataValidationError,
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    read_array_metadata_v2,
)
from zarr_metadata.v2.codec.compression import ZLIB_V2
from zarr_metadata.v2.data_type.scalar import FLOAT_V2, UINT_V2
from zarr_metadata.v2.definition import (
    CORE_V2,
    Context,
    Read,
    ScopeConflictError,
    Unclaimed,
    ZarrV2CodecDefinition,
    ZarrV2DataTypeDefinition,
)
from zarr_metadata.v3.definition import (
    EmptyConfiguration,
)

Loc = tuple[str | int, ...]
ARRAY: dict[str, Any] = {
    "zarr_format": 2,
    "shape": [4],
    "chunks": [2],
    "dtype": "<f4",
    "fill_value": 0,
    "order": "C",
    "compressor": {"id": "zlib"},
    "filters": None,
    "dimension_separator": ".",
    "attributes": {"a": [1, 2]},
}
BARE_ZLIB = ZarrV2CodecDefinition(name="zlib", configuration=EmptyConfiguration)
SMALL = Context.of(FLOAT_V2, UINT_V2)
PRIVATE = CORE_V2.extended_with(BARE_ZLIB)


def _document(changes: dict[str, Any]) -> dict[str, Any]:
    """`ARRAY` with `changes`, a member given as `UNSET` left out."""
    return {key: value for key, value in {**ARRAY, **changes}.items() if value is not UNSET}


def test_a_model_is_its_document_read_in_its_scope() -> None:
    """The constructor reads the document in the scope given, `CORE_V2` by default; `to_json` is the document refined (arrays as tuples, `dimension_separator` put in when missing), sharing nothing with the model; `context` is the scope; the reading holds the model."""
    model = ZarrV2ArrayMetadata(ARRAY)
    assert model.context is CORE_V2
    document = model.to_json()
    assert document["shape"] == (4,)
    assert document.get("attributes") == {"a": (1, 2)}
    without = {k: v for k, v in ARRAY.items() if k != "dimension_separator"}
    assert ZarrV2ArrayMetadata(without).to_json().get("dimension_separator") == "."
    assert model.reading.metadata is model
    assert read_array_metadata_v2(ARRAY).metadata == model
    assert ZarrV2ArrayMetadata(ARRAY, SMALL).context is SMALL


def test_properties_are_what_the_reading_holds() -> None:
    """`dtype`, `compressor` and `filters` are the fields as the scope read them, None where `null` is written; `shape`, `chunks`, `fill_value`, `order`, `dimension_separator`, `attributes` and `extra_fields` are the members as the read refined them, read-only; `claims` is keyed as the scope files them."""
    model = ZarrV2ArrayMetadata({**ARRAY, "filters": [{"id": "x"}], "extra": [1]})
    assert isinstance(model.dtype, Read)
    assert model.dtype.definition is FLOAT_V2
    assert isinstance(model.compressor, Read)
    assert model.compressor.definition is ZLIB_V2
    assert model.filters is not None
    assert isinstance(model.filters[0], Unclaimed)
    members = (model.shape, model.chunks, model.fill_value, model.order, model.dimension_separator)
    assert members == ((4,), (2,), 0, "C", ".")
    assert model.attributes == {"a": (1, 2)}
    assert model.extra_fields == {"extra": (1,)}
    assert (ZarrV2DataTypeDefinition, "float") in model.claims
    with pytest.raises(TypeError):
        cast("dict[str, object]", model.attributes)["b"] = 1
    assert ZarrV2ArrayMetadata({**ARRAY, "compressor": None}).compressor is None
    assert ZarrV2ArrayMetadata(_document({"attributes": UNSET})).attributes is UNSET


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        ({}, {"shape": (4,), "attributes": {"a": (1, 2)}}, True),
        ({"dtype": "<f4"}, {"dtype": "<f4", "fill_value": 0.0}, True),
        ({"fill_value": "NaN"}, {"fill_value": "NaN"}, True),
        ({"dtype": "<b1", "fill_value": None}, {"dtype": "|b1", "fill_value": None}, True),
        (
            {"compressor": {"id": "zlib", "level": 1}},
            {"compressor": {"level": 1, "id": "zlib"}},
            True,
        ),
        ({"fill_value": 0.0}, {"fill_value": -0.0}, False),
        ({"dtype": "<f4"}, {"dtype": ">f4"}, False),
        ({"attributes": {"a": [1, 2]}}, {"attributes": {"a": [2, 1]}}, False),
        ({"attributes": UNSET}, {"attributes": {}}, False),
        ({"extra": 1}, {}, False),
    ],
    ids=[
        "same",
        "int-for-float",
        "nan",
        "spelling",
        "key-order",
        "signed-zero",
        "byte-order",
        "attributes",
        "zattrs-presence",
        "extra",
    ],
)
def test_models_are_equal_by_what_their_documents_mean(
    left: dict[str, Any], right: dict[str, Any], same: bool
) -> None:
    """Two models are one array when their documents mean the same in their scopes: a typestr spelled two ways or a fill value an integer or a float is one, a byte order or a `.zattrs` present or not is two; equal models hash alike."""
    one, other = ZarrV2ArrayMetadata(_document(left)), ZarrV2ArrayMetadata(_document(right))
    assert (one == other) is same
    if same:
        assert hash(one) == hash(other)


def test_a_model_read_in_two_scopes_that_read_it_alike_is_one_model() -> None:
    """Equality is by interpretation: the same document read by a private `zlib` with no parameters and by the core one are two models, and read in two scopes that file the same `zlib` are one."""
    document = {**ARRAY, "compressor": {"id": "zlib"}}
    assert ZarrV2ArrayMetadata(document) != ZarrV2ArrayMetadata(document, PRIVATE)
    same = Context.of(*CORE_V2.definitions())
    assert ZarrV2ArrayMetadata(document) == ZarrV2ArrayMetadata(document, same)


@pytest.mark.parametrize("context", [None, SMALL, PRIVATE], ids=["core", "small", "private"])
def test_a_model_round_trips_through_its_document_pickle_and_copy(context: Context | None) -> None:
    """`from_json(m.to_json(), context=m.context)`, `from_key_value(m.to_key_value())`, `pickle` and `copy.deepcopy` give an equal model in the same scope, and a pickled reading comes back as its model's own reading."""
    model = ZarrV2ArrayMetadata(ARRAY, context)
    assert ZarrV2ArrayMetadata.from_json(model.to_json(), context=model.context) == model
    restored = ZarrV2ArrayMetadata.from_key_value(model.to_key_value(), context=model.context)
    assert restored == model
    loaded = pickle.loads(pickle.dumps(model))
    assert loaded == model
    assert loaded.context == model.context
    assert copy.deepcopy(model) == model
    assert pickle.loads(pickle.dumps(model.reading)).metadata == model


@pytest.mark.parametrize(
    ("changes", "expect"),
    [
        ({"shape": [8], "chunks": [4]}, {"shape": (8,), "chunks": (4,)}),
        ({"dtype": "<i4", "fill_value": 3}, {"dtype": "<i4", "fill_value": 3}),
        ({"attributes": UNSET}, {}),
        ({"compressor": None}, {"compressor": None}),
        ({"extra": 1}, {"extra": 1}),
        ({"extra": UNSET}, {}),
    ],
    ids=["shape", "dtype", "unset-attributes", "null-compressor", "extra", "unset-extra"],
)
def test_update_reads_new_members_in_the_models_own_scope(
    changes: dict[str, Any], expect: dict[str, Any]
) -> None:
    """`update` puts JSON members in place of the document's, leaves out one given as `UNSET`, and reads the result in the model's own scope; the model is unchanged."""
    model = ZarrV2ArrayMetadata({**ARRAY, "extra": 0}, PRIVATE)
    changed = model.update(**changes)
    assert changed.context is PRIVATE
    document = changed.to_json()
    for key, value in expect.items():
        assert document[key] == value
    for key, value in changes.items():
        if value is UNSET:
            assert key not in document
    assert model.to_json()["extra"] == 0


@pytest.mark.parametrize(
    ("changes", "at"),
    [
        ({"dtype": "float32"}, ("dtype",)),
        ({"chunks": [1, 1]}, ("chunks",)),
        ({"dtype": "|b1"}, ("fill_value",)),
    ],
    ids=["dtype", "rank", "fill-no-longer-fits"],
)
def test_error_update_refuses_a_document_with_a_problem(changes: dict[str, Any], at: Loc) -> None:
    """A change that makes a document with a problem is refused at the change, with the problem where it is: a dtype the family's fill value no longer fits is reported at the fill value."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ArrayMetadata(ARRAY).update(**changes)
    assert raised.value.problems[0].loc == at


def test_with_context_and_refined_in_read_the_document_in_another_scope() -> None:
    """`with_context` reads the document in any scope (a loss is allowed), `refined_in` only up the order: a scope that claims what this one left unclaimed is a gain, one that reads a name by another definition or by none is a `ScopeConflictError` naming where the name sits, and a gain that surfaces a problem is a `MetadataValidationError`."""
    unclaimed = ZarrV2ArrayMetadata({**ARRAY, "compressor": {"id": "zlib"}}, SMALL)
    assert isinstance(unclaimed.compressor, Unclaimed)
    gained = unclaimed.refined_in(PRIVATE)
    assert isinstance(gained.compressor, Read)
    assert gained.compressor.definition is BARE_ZLIB
    assert unclaimed.refines(unclaimed)
    assert gained.refines(unclaimed)
    assert not unclaimed.refines(gained)
    with pytest.raises(ScopeConflictError) as conflict:
        gained.refined_in(CORE_V2)
    assert [c.loc for c in conflict.value.conflicts] == [("compressor",)]
    lost = gained.with_context(SMALL)
    assert isinstance(lost.compressor, Unclaimed)
    assert lost == unclaimed
    assert gained.with_context(PRIVATE) == gained
    with pytest.raises(MetadataValidationError):
        ZarrV2ArrayMetadata({**ARRAY, "compressor": {"id": "zlib", "level": 1}}, SMALL).refined_in(
            PRIVATE
        )


def test_refines_orders_models_by_information() -> None:
    """`refines` holds when each field refines its counterpart and every other member is the same, the fill value as the more informed dtype spells it; a fill value that dtype refuses is no refinement; a value of another type refines nothing."""
    core = ZarrV2ArrayMetadata(ARRAY)
    assert core.refines(ZarrV2ArrayMetadata({**ARRAY, "fill_value": 0.0}))
    assert not core.refines(ZarrV2ArrayMetadata({**ARRAY, "shape": [8], "chunks": [2]}))
    small = ZarrV2ArrayMetadata({**ARRAY, "dtype": "<e2", "fill_value": "garbage"}, SMALL)
    assert not ZarrV2ArrayMetadata({**ARRAY, "fill_value": 1}).refines(small)
    assert not core.refines(cast("ZarrV2ArrayMetadata", object()))


def test_error_a_document_with_a_problem_is_refused_at_construction() -> None:
    """No model is invalid: the constructor raises with every problem the read finds."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ArrayMetadata({**ARRAY, "dtype": "float32", "order": "Q"})
    assert [p.loc for p in raised.value.problems] == [("dtype",), ("order",)]


def test_the_dataclass_machinery_is_gone() -> None:
    """A v2 model is built only from a document: `dataclasses.replace` and `dataclasses.fields` do not apply, and the old `...Partial` names are gone."""
    import zarr_metadata

    assert not dataclasses.is_dataclass(ZarrV2ArrayMetadata)
    assert not hasattr(zarr_metadata, "ZarrV2ArrayMetadataPartial")
    assert "ZarrV2ArrayMetadataUpdate" in zarr_metadata.__all__


GROUP: dict[str, Any] = {"zarr_format": 2, "attributes": {"g": [1]}}


@pytest.mark.parametrize(
    ("document", "attributes"),
    [
        (GROUP, {"g": (1,)}),
        ({"zarr_format": 2}, UNSET),
        ({"zarr_format": 2, "attributes": {}}, {}),
    ],
    ids=["attributes", "no-zattrs", "empty-zattrs"],
)
def test_a_group_is_its_document_read_in_its_scope(
    document: dict[str, Any], attributes: object
) -> None:
    """A v2 group model is its document and scope: `attributes` as the read refined them, `UNSET` when no `.zattrs` exists; it round-trips through its document, its store keys, pickle and copy, in any scope."""
    model = ZarrV2GroupMetadata(document, SMALL)
    assert model.context is SMALL
    assert model.attributes == attributes
    assert model.to_json() == ZarrV2GroupMetadata.from_json(document).to_json()
    assert ZarrV2GroupMetadata.from_key_value(model.to_key_value(), context=SMALL) == model
    assert pickle.loads(pickle.dumps(model)) == model
    assert copy.deepcopy(model) == model
    assert dict(model.claims) == {}


def test_a_group_updates_moves_scope_and_refines_as_the_arrays_do() -> None:
    """`update` reads the new attributes in the model's own scope and `UNSET` leaves them out; `with_context` and `refined_in` read the document in another scope and conflict with nothing, since a group holds no field; two groups are one when their attributes are written alike, and `refines` is that equality."""
    model = ZarrV2GroupMetadata(GROUP)
    assert model.update(attributes={"h": 2}).attributes == {"h": 2}
    assert model.update(attributes=UNSET).attributes is UNSET
    assert model.refined_in(SMALL).context is SMALL
    assert model.with_context(PRIVATE) == model
    assert model == ZarrV2GroupMetadata({"zarr_format": 2, "attributes": {"g": (1,)}})
    assert model != ZarrV2GroupMetadata({"zarr_format": 2})
    assert model.refines(model)
    assert not model.refines(ZarrV2GroupMetadata({"zarr_format": 2}))
    assert not dataclasses.is_dataclass(ZarrV2GroupMetadata)


def test_error_a_group_document_with_a_problem_is_refused_at_construction() -> None:
    """A group document with a problem -- a member the spec does not define, an attribute key that is not a string -- is refused with every problem."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2GroupMetadata({"zarr_format": 2, "attributes": {1: "a"}, "extra": 1})
    assert sorted((p.loc, p.kind) for p in raised.value.problems) == [
        (("attributes",), "invalid_type"),
        (("extra",), "unknown_key"),
    ]


ZARRAY = {k: v for k, v in ARRAY.items() if k != "attributes"}
CONSOLIDATED: dict[str, Any] = {
    "zarr_consolidated_format": 1,
    "metadata": {
        ".zgroup": {"zarr_format": 2},
        ".zattrs": {"root": True},
        "a/.zarray": ZARRAY,
        "a/.zattrs": {"a": [1, 2]},
        "b/.zgroup": {"zarr_format": 2},
        "orphan/.zattrs": {"o": 1},
    },
}


def _consolidated(**entries: object) -> dict[str, Any]:
    return {**CONSOLIDATED, "metadata": {**CONSOLIDATED["metadata"], **entries}}


def test_consolidated_metadata_holds_its_entries_verbatim_and_each_node_as_a_model() -> None:
    """`metadata` is the flat file-keyed map as written, refined, read-only; `nodes` is each `.zarray`/`.zgroup` entry merged with its sibling `.zattrs` as a model of the consolidated scope, keyed by node path (`""` for the root); a `.zattrs` with no sibling is kept and makes no node; the whole round-trips through its document, store keys, pickle and copy."""
    model = ZarrV2ConsolidatedMetadata(CONSOLIDATED, PRIVATE)
    assert model.context is PRIVATE
    assert model.metadata["a/.zattrs"] == {"a": (1, 2)}
    assert set(model.nodes) == {"", "a", "b"}
    assert model.nodes["a"] == ZarrV2ArrayMetadata(ARRAY, PRIVATE)
    root = ZarrV2GroupMetadata({"zarr_format": 2, "attributes": {"root": True}}, PRIVATE)
    assert model.nodes[""] == root
    assert model.nodes["b"].attributes is UNSET
    entries = cast("dict[str, Any]", model.to_json()["metadata"])
    assert entries["orphan/.zattrs"] == {"o": 1}
    assert ZarrV2ConsolidatedMetadata.from_key_value(model.to_key_value(), context=PRIVATE) == model
    assert pickle.loads(pickle.dumps(model)) == model
    assert copy.deepcopy(model) == model
    with pytest.raises(TypeError):
        cast("dict[str, object]", model.metadata)["x"] = 1
    assert not dataclasses.is_dataclass(ZarrV2ConsolidatedMetadata)


def test_consolidated_metadata_is_equal_by_its_nodes_and_moves_scope_with_them() -> None:
    """Two consolidated documents are one when each node means the same and the other entries are written alike; `with_context`/`refined_in` read every node in the new scope, a conflict located at the node's entry; `refines` holds when each node refines its counterpart."""
    spelled = _consolidated(**{"a/.zarray": {**ZARRAY, "fill_value": 0.0}})
    assert ZarrV2ConsolidatedMetadata(CONSOLIDATED) == ZarrV2ConsolidatedMetadata(spelled)
    other = _consolidated(**{"orphan/.zattrs": {"o": 2}})
    assert ZarrV2ConsolidatedMetadata(CONSOLIDATED) != ZarrV2ConsolidatedMetadata(other)
    small = ZarrV2ConsolidatedMetadata(CONSOLIDATED, SMALL)
    assert isinstance(small.nodes["a"], ZarrV2ArrayMetadata)
    assert isinstance(small.nodes["a"].compressor, Unclaimed)
    gained = small.refined_in(PRIVATE)
    assert isinstance(gained.nodes["a"], ZarrV2ArrayMetadata)
    assert isinstance(gained.nodes["a"].compressor, Read)
    assert gained.refines(small)
    assert not small.refines(gained)
    with pytest.raises(ScopeConflictError) as conflict:
        gained.refined_in(CORE_V2)
    assert [c.loc for c in conflict.value.conflicts] == [("metadata", "a/.zarray", "compressor")]
    assert gained.with_context(SMALL) == small


@pytest.mark.parametrize(
    ("entries", "at"),
    [
        ({"a/.zarray": {**ZARRAY, "dtype": "float32"}}, ("metadata", "a/.zarray", "dtype")),
        ({"a/.zarray": ZARRAY, "a/.zattrs": {1: 2}}, ("metadata", "a/.zattrs")),
        ({"a/.zarray": ZARRAY, "a/.zgroup": {"zarr_format": 2}}, ("metadata", "a/.zgroup")),
        ({"a/.zgroup": {"zarr_format": 3}}, ("metadata", "a/.zgroup", "zarr_format")),
        ({"a/.zarray": 3}, ("metadata", "a/.zarray")),
    ],
    ids=["array-dtype", "zattrs-key", "array-and-group", "group-format", "not-an-object"],
)
def test_error_a_node_entry_with_a_problem_is_refused_at_the_entry(
    entries: dict[str, Any], at: Loc
) -> None:
    """A `.zarray` or `.zgroup` entry is read as the document it is, in the consolidated scope; its problems sit under the entry, a `.zattrs`'s under its own entry; a path that is both an array and a group is a problem at the `.zgroup`."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ConsolidatedMetadata({"zarr_consolidated_format": 1, "metadata": entries})
    assert raised.value.problems[0].loc == at


def test_error_two_entries_for_one_node_file_are_refused() -> None:
    """Two keys that name one file of one node -- `.zarray` and `/.zarray` -- are a problem at the second, since one would otherwise go unread; and a conflict is located at the key the document writes."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ConsolidatedMetadata(
            {"zarr_consolidated_format": 1, "metadata": {".zarray": ZARRAY, "/.zarray": ZARRAY}}
        )
    assert [(p.loc, p.kind) for p in raised.value.problems] == [
        (("metadata", "/.zarray"), "invalid_value")
    ]
    gained = ZarrV2ConsolidatedMetadata(
        {"zarr_consolidated_format": 1, "metadata": {"/.zarray": ZARRAY}}, PRIVATE
    )
    with pytest.raises(ScopeConflictError) as conflict:
        gained.refined_in(CORE_V2)
    assert [c.loc for c in conflict.value.conflicts] == [("metadata", "/.zarray", "compressor")]


def test_the_node_type_is_exported() -> None:
    """`ZarrV2NodeMetadata`, the type of each node `nodes` holds, is exported beside the models, as `ZarrV3NodeMetadata` is."""
    from zarr_metadata import model

    assert "ZarrV2NodeMetadata" in model.__all__


def test_error_a_second_key_for_a_node_file_hides_no_other_problem() -> None:
    """A second key naming one file of one node is one problem among the document's: every node is still read, and each node's problems reported, so a user sees everything at once; a leading or repeated `/` does not make a second node."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ConsolidatedMetadata(
            {
                "zarr_consolidated_format": 1,
                "metadata": {
                    "a/.zarray": ZARRAY,
                    "/a/.zarray": ZARRAY,
                    "b/.zgroup": {"zarr_format": 3},
                },
            }
        )
    assert [p.loc for p in raised.value.problems] == [
        ("metadata", "/a/.zarray"),
        ("metadata", "b/.zgroup", "zarr_format"),
    ]
    one = ZarrV2ConsolidatedMetadata(
        {"zarr_consolidated_format": 1, "metadata": {"/a/.zarray": ZARRAY}}
    )
    assert set(
        ZarrV2ConsolidatedMetadata(
            {"zarr_consolidated_format": 1, "metadata": {"x//y/.zarray": ZARRAY}}
        ).nodes
    ) == {"x/y"}
    assert set(one.nodes) == {"a"}
    assert one == ZarrV2ConsolidatedMetadata(
        {"zarr_consolidated_format": 1, "metadata": {"a/.zarray": ZARRAY}}
    )
