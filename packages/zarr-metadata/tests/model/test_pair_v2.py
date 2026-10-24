"""Tests that a v2 model is its document plus the scope it was read in, as the v3 models are."""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle
from collections.abc import Mapping
from typing import Any, cast, get_args

import pytest

from zarr_metadata.definition import (
    AcceptedField,
    EmptyConfiguration,
    ScopeConflictError,
    UnclaimedField,
)
from zarr_metadata.model import (
    UNSET,
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    read_array_metadata_v2,
    read_consolidated_metadata_v2,
)
from zarr_metadata.problems import MetadataValidationError
from zarr_metadata.v2.codec.compression import ZLIB_CODEC_V2
from zarr_metadata.v2.data_type.scalar import FLOAT_DATA_TYPE_V2, UINT_DATA_TYPE_V2
from zarr_metadata.v2.definition import (
    CORE_V2,
    ZarrV2CodecDefinition,
    ZarrV2Context,
    ZarrV2DataTypeDefinition,
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
SMALL = ZarrV2Context(data_types=[FLOAT_DATA_TYPE_V2, UINT_DATA_TYPE_V2])
PRIVATE = CORE_V2.extended_with(codecs=[BARE_ZLIB])


def _document(changes: dict[str, Any]) -> dict[str, Any]:
    """Return `ARRAY` with `changes` applied, leaving out any member given as `UNSET`."""
    return {key: value for key, value in {**ARRAY, **changes}.items() if value is not UNSET}


def test_a_model_is_its_document_read_in_its_scope() -> None:
    """The constructor reads the document in the given scope (`CORE_V2` by default). `to_json` returns the refined document (arrays as tuples), which shares nothing with the model. A missing `dimension_separator` stays missing. `context` is the scope, and the reading holds the model."""
    model = ZarrV2ArrayMetadata(ARRAY)
    assert model.context is CORE_V2
    document = model.to_json()
    assert document["shape"] == (4,)
    assert document.get("attributes") == {"a": (1, 2)}
    without = {k: v for k, v in ARRAY.items() if k != "dimension_separator"}
    assert "dimension_separator" not in ZarrV2ArrayMetadata(without).to_json()
    assert model.reading.metadata is model
    assert read_array_metadata_v2(ARRAY).metadata == model
    assert ZarrV2ArrayMetadata(ARRAY, SMALL).context is SMALL


def test_properties_are_what_the_reading_holds() -> None:
    """`dtype`, `compressor` and `filters` are the fields as the scope read them, or None where the document has `null`. `shape`, `chunks`, `fill_value`, `order`, `dimension_separator`, `attributes` and `extra_fields` are the members as the read refined them, and they are read-only. `zarr_format` and `node_type` are 2 and `"array"`, as a v3 array model has 3 and `"array"`. `claims` is keyed the same way the scope files its definitions."""
    model = ZarrV2ArrayMetadata({**ARRAY, "filters": [{"id": "x"}], "extra": [1]})
    assert isinstance(model.dtype, AcceptedField)
    assert model.dtype.definition is FLOAT_DATA_TYPE_V2
    assert isinstance(model.compressor, AcceptedField)
    assert model.compressor.definition is ZLIB_CODEC_V2
    assert model.filters is not None
    assert isinstance(model.filters[0], UnclaimedField)
    members = (model.shape, model.chunks, model.fill_value, model.order, model.dimension_separator)
    assert members == ((4,), (2,), 0, "C", ".")
    assert model.attributes == {"a": (1, 2)}
    assert model.extra_fields == {"extra": (1,)}
    assert (model.zarr_format, model.node_type) == (2, "array")
    assert (ZarrV2DataTypeDefinition, "float") in model.claims
    with pytest.raises(TypeError):
        cast("dict[str, object]", model.attributes)["b"] = 1
    assert ZarrV2ArrayMetadata({**ARRAY, "compressor": None}).compressor is None
    without = ZarrV2ArrayMetadata(_document({"attributes": UNSET}))
    assert without.attributes == {}
    assert isinstance(without.attributes, Mapping)
    assert "attributes" not in without.to_json()


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
        (
            {"compressor": {"id": "zstd", "level": 0}},
            {"compressor": {"id": "zstd", "level": 0, "checksum": False}},
            True,
        ),
        ({"filters": [{"id": "crc32"}]}, {"filters": [{"id": "crc32", "location": "start"}]}, True),
        ({"fill_value": 0.0}, {"fill_value": -0.0}, False),
        ({"dtype": "<f4"}, {"dtype": ">f4"}, False),
        ({"compressor": {"id": "zstd", "level": 0}}, {"compressor": {"id": "zstd"}}, True),
        ({"compressor": {"id": "zstd", "level": 1}}, {"compressor": {"id": "zstd"}}, False),
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
        "numcodecs-default",
        "numcodecs-default-filter",
        "signed-zero",
        "byte-order",
        "numcodecs-default-omitted",
        "codec-parameter",
        "attributes",
        "zattrs-presence",
        "extra",
    ],
)
def test_models_are_equal_by_what_their_documents_mean(
    left: dict[str, Any], right: dict[str, Any], same: bool
) -> None:
    """Two models are equal when their documents mean the same in their scopes. A typestr with two spellings, a fill value written as an integer or a float, or a codec parameter written or left to its numcodecs default, gives equal models. A different byte order, a codec parameter away from its default, or a `.zattrs` present in one and absent in the other, gives different models. Equal models have equal hashes."""
    one, other = ZarrV2ArrayMetadata(_document(left)), ZarrV2ArrayMetadata(_document(right))
    assert (one == other) is same
    if same:
        assert hash(one) == hash(other)


def test_a_model_read_in_two_scopes_that_read_it_alike_is_one_model() -> None:
    """Equality depends on interpretation. The same document read with a private `zlib` that takes no parameters and with the core `zlib` gives two different models. Read in two scopes that hold the same `zlib`, it gives equal models."""
    document = {**ARRAY, "compressor": {"id": "zlib"}}
    assert ZarrV2ArrayMetadata(document) != ZarrV2ArrayMetadata(document, PRIVATE)
    same = ZarrV2Context(data_types=CORE_V2.data_types.values(), codecs=CORE_V2.codecs.values())
    assert ZarrV2ArrayMetadata(document) == ZarrV2ArrayMetadata(document, same)


@pytest.mark.parametrize("context", [None, SMALL, PRIVATE], ids=["core", "small", "private"])
def test_a_model_round_trips_through_its_document_pickle_and_copy(
    context: ZarrV2Context | None,
) -> None:
    """`from_json(m.to_json(), context=m.context)`, `from_key_value(m.to_key_value())`, `pickle` and `copy.deepcopy` each give an equal model in the same scope. A pickled reading unpickles as its model's own reading."""
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
    ("changes", "expect", "absent"),
    [
        ({"shape": [8], "chunks": [4]}, {"shape": (8,), "chunks": (4,)}, ()),
        ({"dtype": "<i4", "fill_value": 3}, {"dtype": "<i4", "fill_value": 3}, ()),
        ({"attributes": UNSET}, {}, ("attributes",)),
        ({"compressor": None}, {"compressor": None}, ()),
        ({"extra_fields": {"extra": 1}}, {"extra": 1}, ()),
        ({"extra_fields": {"extra": UNSET}}, {}, ("extra",)),
    ],
    ids=["shape", "dtype", "unset-attributes", "null-compressor", "extra", "unset-extra"],
)
def test_update_reads_new_members_in_the_models_own_scope(
    changes: dict[str, Any], expect: dict[str, Any], absent: tuple[str, ...]
) -> None:
    """`update` replaces the document's members with the given JSON members, takes a member the spec does not define in `extra_fields`, leaves out any member given as `UNSET`, and reads the result in the model's own scope. The original model is unchanged."""
    model = ZarrV2ArrayMetadata({**ARRAY, "extra": 0}, PRIVATE)
    changed = model.update(**changes)
    assert changed.context is PRIVATE
    document = changed.to_json()
    for key, value in expect.items():
        assert document[key] == value
    for key in absent:
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
    """A change that produces a document with a problem is refused when the change is made, and the problem is reported at its location. For example, a dtype change that makes the fill value invalid is reported at the fill value."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ArrayMetadata(ARRAY).update(**changes)
    assert raised.value.problems[0].loc == at


def test_with_context_and_refined_in_read_the_document_in_another_scope() -> None:
    """`with_context` reads the document in any scope, and losing information is allowed. `refined_in` only moves up the order. A scope that claims what the current scope left unclaimed is a gain. A scope that reads a name with another definition, or with no definition, raises a `ScopeConflictError` that gives the name's location. A gain that reveals a problem raises a `MetadataValidationError`."""
    unclaimed = ZarrV2ArrayMetadata({**ARRAY, "compressor": {"id": "zlib"}}, SMALL)
    assert isinstance(unclaimed.compressor, UnclaimedField)
    gained = unclaimed.refined_in(PRIVATE)
    assert isinstance(gained.compressor, AcceptedField)
    assert gained.compressor.definition is BARE_ZLIB
    assert unclaimed.refines(unclaimed)
    assert gained.refines(unclaimed)
    assert not unclaimed.refines(gained)
    with pytest.raises(ScopeConflictError) as conflict:
        gained.refined_in(CORE_V2)
    assert [c.loc for c in conflict.value.conflicts] == [("compressor",)]
    lost = gained.with_context(SMALL)
    assert isinstance(lost.compressor, UnclaimedField)
    assert lost == unclaimed
    assert gained.with_context(PRIVATE) == gained
    with pytest.raises(MetadataValidationError):
        ZarrV2ArrayMetadata({**ARRAY, "compressor": {"id": "zlib", "level": 1}}, SMALL).refined_in(
            PRIVATE
        )


def test_refines_orders_models_by_information() -> None:
    """`refines` is true when each field refines its counterpart and every other member is the same, with the fill value compared as the more informed dtype spells it. A fill value that this dtype refuses is not a refinement. A value of another type refines nothing."""
    core = ZarrV2ArrayMetadata(ARRAY)
    assert core.refines(ZarrV2ArrayMetadata({**ARRAY, "fill_value": 0.0}))
    filled = ZarrV2ArrayMetadata({**ARRAY, "compressor": {"id": "zlib", "level": 1}})
    assert core.refines(filled)
    assert filled.refines(core)
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
    """A v2 model is built only from a document. `dataclasses.replace` and `dataclasses.fields` do not apply, and the old `...Partial` names are removed."""
    import zarr_metadata

    assert not dataclasses.is_dataclass(ZarrV2ArrayMetadata)
    assert not hasattr(zarr_metadata, "ZarrV2ArrayMetadataPartial")
    assert "ZarrV2ArrayMetadataUpdate" in zarr_metadata.__all__


GROUP: dict[str, Any] = {"zarr_format": 2, "attributes": {"g": [1]}}


@pytest.mark.parametrize(
    ("document", "attributes"),
    [
        (GROUP, {"g": (1,)}),
        ({"zarr_format": 2}, {}),
        ({"zarr_format": 2, "attributes": {}}, {}),
    ],
    ids=["attributes", "no-zattrs", "empty-zattrs"],
)
def test_a_group_is_its_document_read_in_its_scope(
    document: dict[str, Any], attributes: object
) -> None:
    """A v2 group model is its document plus its scope. `attributes` holds the attributes as the read refined them, and is empty when no `.zattrs` exists, as a v3 model's `attributes` is empty when the document writes none. `zarr_format` and `node_type` are 2 and `"group"`, as a v3 group model has 3 and `"group"`. The model round-trips through its document, its store keys, pickle and copy, in any scope."""
    model = ZarrV2GroupMetadata(document, SMALL)
    assert model.context is SMALL
    assert (model.zarr_format, model.node_type) == (2, "group")
    assert model.attributes == attributes
    assert model.to_json() == ZarrV2GroupMetadata.from_json(document).to_json()
    assert ZarrV2GroupMetadata.from_key_value(model.to_key_value(), context=SMALL) == model
    assert pickle.loads(pickle.dumps(model)) == model
    assert copy.deepcopy(model) == model
    assert dict(model.claims) == {}


def test_a_group_updates_moves_scope_and_refines_as_the_arrays_do() -> None:
    """`update` reads the new attributes in the model's own scope, and `UNSET` removes them. `with_context` and `refined_in` read the document in another scope and never conflict, since a group holds no field. Two groups are equal when their attributes are written the same way, and `refines` is that equality."""
    model = ZarrV2GroupMetadata(GROUP)
    assert model.update(attributes={"h": 2}).attributes == {"h": 2}
    removed = model.update(attributes=UNSET)
    assert removed.attributes == {}
    assert ".zattrs" not in removed.to_key_value()
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


def test_consolidated_metadata_maps_each_node_path_to_its_model() -> None:
    """`metadata` maps the path of each node, without the leading `/` (`""` for the root), to the model of its `.zarray` or `.zgroup` entry merged with its sibling `.zattrs`, read in the consolidated scope, as `ZarrV3ConsolidatedMetadata.metadata` maps each path to its model. It is read-only. The entries stay in `to_json()["metadata"]`, keyed by file, as written and refined. A `.zattrs` with no sibling is kept there but makes no node. The whole model round-trips through its document, store keys, pickle and copy."""
    model = ZarrV2ConsolidatedMetadata(CONSOLIDATED, PRIVATE)
    assert model.context is PRIVATE
    assert set(model.metadata) == {"", "a", "b"}
    assert model.metadata["a"] == ZarrV2ArrayMetadata(ARRAY, PRIVATE)
    root = ZarrV2GroupMetadata({"zarr_format": 2, "attributes": {"root": True}}, PRIVATE)
    assert model.metadata[""] == root
    assert model.metadata["b"].attributes == {}
    assert not hasattr(model, "nodes")
    entries = cast("dict[str, Any]", model.to_json()["metadata"])
    assert entries["a/.zattrs"] == {"a": (1, 2)}
    assert entries["orphan/.zattrs"] == {"o": 1}
    assert ZarrV2ConsolidatedMetadata.from_key_value(model.to_key_value(), context=PRIVATE) == model
    assert pickle.loads(pickle.dumps(model)) == model
    assert copy.deepcopy(model) == model
    with pytest.raises(TypeError):
        cast("dict[str, object]", model.metadata)["x"] = 1
    assert not dataclasses.is_dataclass(ZarrV2ConsolidatedMetadata)


@pytest.mark.parametrize(
    "document",
    [CONSOLIDATED, {"metadata": CONSOLIDATED["metadata"], "zarr_consolidated_format": 1}],
    ids=["format-first", "metadata-first"],
)
def test_consolidated_metadata_is_written_in_the_order_it_was_read(
    document: dict[str, Any],
) -> None:
    """`to_json` and `to_key_value` write the top-level keys of a `.zmetadata` in the order the document writes them, as the v3 models do. zarr-python 3 writes `metadata` first. A model built by `read_consolidated_metadata_v2` keeps the order too."""
    model = ZarrV2ConsolidatedMetadata(document)
    assert list(model.to_json()) == list(document)
    assert list(json.loads(model.to_key_value()[".zmetadata"])) == list(document)
    reading = read_consolidated_metadata_v2(document)
    assert reading.metadata is not None
    assert list(reading.metadata.to_json()) == list(document)


def test_consolidated_metadata_is_equal_by_its_nodes_and_moves_scope_with_them() -> None:
    """Two consolidated documents are equal when each node means the same and the other entries are written the same way. `with_context` and `refined_in` read every node in the new scope, and a conflict is located at the node's entry. `refines` is true when each node refines its counterpart."""
    spelled = _consolidated(**{"a/.zarray": {**ZARRAY, "fill_value": 0.0}})
    assert ZarrV2ConsolidatedMetadata(CONSOLIDATED) == ZarrV2ConsolidatedMetadata(spelled)
    other = _consolidated(**{"orphan/.zattrs": {"o": 2}})
    assert ZarrV2ConsolidatedMetadata(CONSOLIDATED) != ZarrV2ConsolidatedMetadata(other)
    small = ZarrV2ConsolidatedMetadata(CONSOLIDATED, SMALL)
    assert isinstance(small.metadata["a"], ZarrV2ArrayMetadata)
    assert isinstance(small.metadata["a"].compressor, UnclaimedField)
    gained = small.refined_in(PRIVATE)
    assert isinstance(gained.metadata["a"], ZarrV2ArrayMetadata)
    assert isinstance(gained.metadata["a"].compressor, AcceptedField)
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
    """A `.zarray` or `.zgroup` entry is read as its document type, in the consolidated scope. Its problems are located under the entry, and a `.zattrs` entry's problems under its own entry. A path that is both an array and a group is a problem at the `.zgroup`."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ConsolidatedMetadata({"zarr_consolidated_format": 1, "metadata": entries})
    assert raised.value.problems[0].loc == at


def test_error_two_entries_for_one_node_file_are_refused() -> None:
    """Two keys that name the same file of the same node (`.zarray` and `/.zarray`) are a problem at the second key, since one of them would otherwise not be read. A conflict is located at the key as the document writes it."""
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
    """`ZarrV2NodeMetadata`, the type of each node in a `.zmetadata`'s `metadata`, is exported alongside the models, as `ZarrV3NodeMetadata` is."""
    from zarr_metadata import model

    assert "ZarrV2NodeMetadata" in model.__all__


def test_the_node_type_is_a_type_alias_at_run_time() -> None:
    """`ZarrV2NodeMetadata` is a type alias at run time, as `ZarrV3NodeMetadata` is, so the two form a union, and its value is the union of the v2 array and group models."""
    from zarr_metadata.model import ZarrV2NodeMetadata, ZarrV3NodeMetadata

    either = ZarrV2NodeMetadata | ZarrV3NodeMetadata
    assert get_args(either) == (ZarrV2NodeMetadata, ZarrV3NodeMetadata)
    assert get_args(ZarrV2NodeMetadata.__value__) == (ZarrV2ArrayMetadata, ZarrV2GroupMetadata)


def test_error_a_second_key_for_a_node_file_hides_no_other_problem() -> None:
    """A second key that names the same file of the same node is one problem among the document's problems. Every node is still read and each node's problems are reported, so a user sees everything at once. A leading or repeated `/` does not make a second node."""
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
            {
                "zarr_consolidated_format": 1,
                "metadata": {"x/.zgroup": {"zarr_format": 2}, "x//y/.zarray": ZARRAY},
            }
        ).metadata
    ) == {"x", "x/y"}
    assert set(one.metadata) == {"a"}
    assert one == ZarrV2ConsolidatedMetadata(
        {"zarr_consolidated_format": 1, "metadata": {"a/.zarray": ZARRAY}}
    )


@pytest.mark.parametrize("name", ["__private", "zarr.json"])
def test_a_v2_node_may_have_a_name_that_only_v3_reserves(name: str) -> None:
    """Zarr v2 reserves no node names. A `.zmetadata` node named `__private` or `zarr.json`, which v3 reserves, is a valid v2 node."""
    model = ZarrV2ConsolidatedMetadata(
        {
            "zarr_consolidated_format": 1,
            "metadata": {".zgroup": {"zarr_format": 2}, f"{name}/.zarray": ZARRAY},
        }
    )
    assert set(model.metadata) == {"", name}


def test_error_a_key_with_a_dot_segment_is_no_node_s_path() -> None:
    """A `.zmetadata` key whose path has a `.` or `..` segment names no node, unlike a key with a leading or repeated `/`. It is a problem at the key, and it does not make a second node next to the one it would name."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ConsolidatedMetadata(
            {
                "zarr_consolidated_format": 1,
                "metadata": {
                    ".zgroup": {"zarr_format": 2},
                    "./a/.zarray": ZARRAY,
                    "b/../a/.zarray": ZARRAY,
                },
            }
        )
    assert [(p.loc, p.kind) for p in raised.value.problems] == [
        (("metadata", "./a/.zarray"), "invalid_value"),
        (("metadata", "b/../a/.zarray"), "invalid_value"),
    ]


@pytest.mark.parametrize(
    ("entries", "at", "kind"),
    [
        (
            {".zgroup": {"zarr_format": 2}, "a/.zarray": ZARRAY, "a/b/.zarray": ZARRAY},
            "a/b/.zarray",
            "invalid_value",
        ),
        ({".zarray": ZARRAY, "x/.zgroup": {"zarr_format": 2}}, "x/.zgroup", "invalid_value"),
        ({".zgroup": {"zarr_format": 2}, "x/y/.zarray": ZARRAY}, "x/.zgroup", "missing_key"),
    ],
    ids=["below-an-array", "below-the-root-array", "missing-parent"],
)
def test_error_the_nodes_of_a_zmetadata_make_a_hierarchy(
    entries: dict[str, Any], at: str, kind: str
) -> None:
    """The nodes in a `.zmetadata` form a hierarchy, as a v3 group's consolidated metadata does. A node below an array is a problem at its entry. A missing parent group is a `missing_key` problem at the entry that the group would have."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ConsolidatedMetadata({"zarr_consolidated_format": 1, "metadata": entries})
    assert [(p.loc, p.kind) for p in raised.value.problems] == [(("metadata", at), kind)]
