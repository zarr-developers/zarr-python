"""Tests for the group and consolidated metadata models in `zarr_metadata.model`."""

import copy
import dataclasses
import json
from collections import UserDict
from collections.abc import Callable, Iterator
from typing import cast

import pytest

from tests.model._cases import mutate_nested_containers
from zarr_metadata._common import JSONValue, ZarrV3NamedConfigJSON
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
)
from zarr_metadata.model import UNSET
from zarr_metadata.model._array import ZarrV3ArrayMetadata, ZarrV3ArrayMetadataUpdate
from zarr_metadata.model._group import (
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV2GroupMetadataPartial,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
    ZarrV3GroupMetadataReading,
    ZarrV3GroupMetadataUpdate,
    ZarrV3UnknownNodeReading,
    is_group_metadata_v3,
    node_metadata_from_json_v3,
    node_metadata_from_key_value_v3,
    parse_group_metadata_v3,
    read_group_metadata_v3,
    read_node_metadata_v3,
    validate_group_metadata_v3,
    validate_node_metadata_v3,
)
from zarr_metadata.model._validation import (
    ZarrV3ArrayMetadataReading,
    is_group_metadata_v2,
    parse_group_metadata_v2,
    validate_group_metadata_v2,
)
from zarr_metadata.v2.group import (
    ZarrV2GroupMetadataJSON,
    ZarrV2GroupMetadataJSONPartial,
    ZarrV2ZGroupJSON,
)
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSONPartial
from zarr_metadata.v3.codec.gzip import GZIP_CODEC, GzipCodecConfiguration
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    CodecDefinition,
    EmptyConfiguration,
    Nested,
    Refused,
    Unclaimed,
    resolve,
)
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSONPartial

# --- ZarrV3GroupMetadata ---------------------------------------------------


def test_group_v3_roundtrip() -> None:
    """A v3 group document round-trips through the model unchanged."""
    doc = {"zarr_format": 3, "node_type": "group", "attributes": {"a": (1, 2)}}
    model = ZarrV3GroupMetadata.from_json(doc)
    assert model.to_json() == doc


def test_group_v3_omits_empty_attributes() -> None:
    """to_json omits the attributes key when attributes is empty."""
    model = ZarrV3GroupMetadata.create_default()
    assert "attributes" not in model.to_json()


def test_group_v3_lists_become_tuples() -> None:
    """from_json converts JSON arrays in attributes to tuples."""
    doc = {"zarr_format": 3, "node_type": "group", "attributes": {"a": [1, 2]}}
    model = ZarrV3GroupMetadata.from_json(doc)
    assert model.attributes == {"a": (1, 2)}


def test_group_v3_extra_fields_roundtrip() -> None:
    """Unknown top-level keys land in extra_fields and reappear in to_json."""
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "my_extension": {"name": "thing", "must_understand": False},
    }
    model = ZarrV3GroupMetadata.from_json(doc)
    assert model.extra_fields == {"my_extension": {"name": "thing", "must_understand": False}}
    assert model.to_json() == doc


def test_group_v3_json_extra_field_roundtrips_as_must_understand() -> None:
    """A non-object extra field is preserved and implicitly requires understanding."""
    doc = {"zarr_format": 3, "node_type": "group", "ext": [1, 2]}
    model = ZarrV3GroupMetadata.from_json(doc)
    assert model.to_json()["ext"] == (1, 2)
    assert model.must_understand_fields == {"ext": (1, 2)}


def test_group_v3_extra_fields_overlap_rejected() -> None:
    """Constructing a v3 group model with extra_fields shadowing a standard key raises."""
    with pytest.raises(ValueError, match="Extra fields"):
        ZarrV3GroupMetadata(
            attributes={},
            consolidated_metadata=UNSET,
            extra_fields={"node_type": {"name": "x", "must_understand": False}},
        )


def test_group_v3_consolidated_extra_field_rejected() -> None:
    """extra_fields may not shadow the consolidated_metadata convention key."""
    with pytest.raises(ValueError, match="Extra fields"):
        ZarrV3GroupMetadata(
            attributes={},
            consolidated_metadata=UNSET,
            extra_fields={"consolidated_metadata": {"name": "x", "must_understand": False}},
        )


def test_group_v3_missing_required_key() -> None:
    """parse_group_metadata_v3 reports each missing required key."""
    with pytest.raises(MetadataValidationError, match="node_type"):
        parse_group_metadata_v3({"zarr_format": 3})


def test_group_v3_bad_attributes() -> None:
    """parse_group_metadata_v3 rejects a non-mapping attributes value."""
    with pytest.raises(MetadataValidationError, match="attributes"):
        parse_group_metadata_v3({"zarr_format": 3, "node_type": "group", "attributes": 5})


@pytest.mark.parametrize(
    ("document", "validate"),
    [
        pytest.param(
            {"zarr_format": 2.0},
            validate_group_metadata_v2,
            id="v2",
        ),
        pytest.param(
            {"zarr_format": 3.0, "node_type": "group"},
            validate_group_metadata_v3,
            id="v3",
        ),
    ],
)
def test_group_zarr_format_rejects_float(
    document: object, validate: Callable[[object], list[ValidationProblem]]
) -> None:
    """Integer-valued floats do not satisfy integer format literals."""
    assert [(p.loc, p.kind) for p in validate(document)] == [(("zarr_format",), "invalid_value")]


def _v2_consolidated_problems(document: object) -> list[tuple[tuple[str | int, ...], str]]:
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ConsolidatedMetadata.from_json(document)
    return [(p.loc, p.kind) for p in raised.value.problems]


def _v3_field_problems(field: object) -> list[tuple[tuple[str | int, ...], str]]:
    return [(p.loc, p.kind) for p in resolve(field, CodecDefinition, CORE_AND_EXTENSIONS)[1]]


@pytest.mark.parametrize(
    ("problems", "expected"),
    [
        (
            lambda: [
                (p.loc, p.kind) for p in validate_group_metadata_v2({"zarr_format": 2, "x": 1})
            ],
            [(("x",), "unknown_key")],
        ),
        (
            lambda: _v2_consolidated_problems(
                {"zarr_consolidated_format": 1, "metadata": {}, "x": 1}
            ),
            [(("x",), "unknown_key")],
        ),
        (
            lambda: [
                (p.loc, p.kind)
                for p in validate_group_metadata_v3(
                    _group(consolidated_metadata={**_inline(), "x": 1})
                )
            ],
            [(("consolidated_metadata", "x"), "unknown_key")],
        ),
        (
            lambda: _v3_field_problems({"name": "gzip", "configuration": {"level": 1}, "x": 1}),
            [(("x",), "unknown_key")],
        ),
        (
            lambda: _v3_field_problems({"name": "gzip", "configuration": {"level": 1, "x": 1}}),
            [(("configuration", "x"), "unknown_key")],
        ),
    ],
    ids=["v2-group", "v2-consolidated", "v3-consolidated", "v3-field", "v3-configuration"],
)
def test_a_member_a_closed_object_does_not_declare_is_an_unknown_key(
    problems: Callable[[], list[tuple[tuple[str | int, ...], str]]],
    expected: list[tuple[tuple[str | int, ...], str]],
) -> None:
    # Wherever it sits, so a reader that tolerates what another writer
    # added -- NCZarr's `_nczarr_*` keys -- filters by kind.
    assert problems() == expected


@pytest.mark.parametrize(
    "document_type",
    [ZarrV2ZGroupJSON, ZarrV2GroupMetadataJSON, ZarrV2GroupMetadataJSONPartial],
    ids=lambda document_type: document_type.__name__,
)
def test_v2_group_document_types_are_closed(document_type: type) -> None:
    """`.zgroup` "Other keys MUST NOT be present": the validator refuses them, and
    the types say so."""
    assert getattr(document_type, "__closed__", None) is True


@pytest.mark.parametrize(
    ("parse", "document"),
    [
        pytest.param(parse_group_metadata_v2, {"zarr_format": 2}, id="v2"),
        pytest.param(
            parse_group_metadata_v3,
            {"zarr_format": 3, "node_type": "group"},
            id="v3",
        ),
    ],
)
def test_group_parser_materializes_abstract_mapping(
    parse: Callable[[object], object], document: dict[str, object]
) -> None:
    """A successful group parser always returns the declared concrete TypedDict shape."""
    parsed = parse(UserDict(document))

    assert type(parsed) is dict
    assert parsed == document


def test_group_guards_reject_noncanonical_nested_json() -> None:
    """Document guards cannot narrow values that only parsers can materialize."""
    v3 = {"zarr_format": 3, "node_type": "group", "extension": range(2)}
    v2 = {"zarr_format": 2, "attributes": {"values": range(2)}}

    assert not is_group_metadata_v3(v3)
    assert not is_group_metadata_v2(v2)
    assert parse_group_metadata_v3(v3)["extension"] == (0, 1)
    assert parse_group_metadata_v2(v2).get("attributes") == {"values": (0, 1)}


def test_group_v3_extension_fields_are_validated() -> None:
    """Group extension payloads must be JSON values with a must-understand flag."""
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "ext": {"must_understand": False, "payload": object()},
    }
    assert [(problem.loc, problem.kind) for problem in validate_group_metadata_v3(doc)] == [
        (("ext", "payload"), "invalid_type")
    ]


def test_group_v3_key_value_roundtrip() -> None:
    """from_key_value(to_key_value()) is the identity for v3 groups."""
    model = ZarrV3GroupMetadata.create_default(attributes={"a": 1})
    assert ZarrV3GroupMetadata.from_key_value(model.to_key_value()) == model


def test_group_v3_update() -> None:
    """update replaces the given fields and returns a new instance."""
    base = ZarrV3GroupMetadata.create_default()
    updated = base.update(context=CORE_AND_EXTENSIONS, attributes={"a": 1})
    assert updated.attributes == {"a": 1}
    assert base.attributes == {}


# --- ZarrV2GroupMetadata ---------------------------------------------------


def test_group_v2_key_value_split() -> None:
    """v2 to_key_value writes .zgroup and .zattrs; from_key_value merges them."""
    model = ZarrV2GroupMetadata.create_default(attributes={"a": 1})
    kv = model.to_key_value()
    assert set(kv) == {".zgroup", ".zattrs"}
    assert json.loads(kv[".zgroup"]) == {"zarr_format": 2}
    assert ZarrV2GroupMetadata.from_key_value(kv) == model


@pytest.mark.parametrize("extra_key", ["attributes", "vendor_extension"])
def test_v2_group_from_key_value_rejects_zgroup_extra_members(extra_key: str) -> None:
    """Raw `.zgroup` documents reject every non-spec member."""
    doc: dict[str, object] = {"zarr_format": 2, extra_key: {}}

    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV2GroupMetadata.from_key_value({".zgroup": json.dumps(doc).encode()})

    assert [(problem.loc, problem.kind) for problem in exc_info.value.problems] == [
        ((extra_key,), "unknown_key")
    ]


def test_group_v2_zattrs_presence_round_trips() -> None:
    """A v2 group with no .zattrs file parses with UNSET attributes and emits
    no .zattrs; an explicit empty .zattrs stays a file — the stores remain
    distinct through a round-trip."""
    absent = ZarrV2GroupMetadata.from_key_value({".zgroup": b'{"zarr_format": 2}'})
    assert absent.attributes is UNSET
    assert ".zattrs" not in absent.to_key_value()
    explicit = ZarrV2GroupMetadata.from_key_value(
        {".zgroup": b'{"zarr_format": 2}', ".zattrs": b"{}"}
    )
    assert explicit.attributes == {}
    assert ".zattrs" in explicit.to_key_value()
    assert absent != explicit


def test_group_v2_json_roundtrip() -> None:
    """A merged-form v2 group document round-trips through the model unchanged."""
    doc = {"zarr_format": 2, "attributes": {"a": 1}}
    model = ZarrV2GroupMetadata.from_json(doc)
    assert model.to_json() == doc


def test_group_v2_omits_empty_attributes() -> None:
    """to_json omits the attributes key when attributes is empty."""
    assert "attributes" not in ZarrV2GroupMetadata.create_default().to_json()


def test_group_v2_not_a_mapping() -> None:
    """parse_group_metadata_v2 rejects a non-mapping document."""
    with pytest.raises(MetadataValidationError, match="expected an object"):
        parse_group_metadata_v2([1, 2, 3])


def test_group_v2_missing_required_key() -> None:
    """parse_group_metadata_v2 reports a missing zarr_format key."""
    with pytest.raises(MetadataValidationError, match="zarr_format"):
        parse_group_metadata_v2({})


# --- Partial TypedDict drift guards -----------------------------------------


def test_group_partial_keys_match_settable_model_fields() -> None:
    """The v2 group partial TypedDict lists exactly the settable model fields.

    Guards against drift: adding/removing a settable field on the model
    without updating its `*Partial` TypedDict fails here.
    """
    settable = {f.name for f in dataclasses.fields(ZarrV2GroupMetadata) if f.init}
    assert set(ZarrV2GroupMetadataPartial.__annotations__) == settable


def test_update_takes_every_member_of_the_document_it_may_change() -> None:
    """Each v3 model's `update` takes each member of its document but `zarr_format` and `node_type`, which it cannot change."""
    for update, partial in (
        (ZarrV3ArrayMetadataUpdate, ZarrV3ArrayMetadataJSONPartial),
        (ZarrV3GroupMetadataUpdate, ZarrV3GroupMetadataJSONPartial),
    ):
        fixed = {"zarr_format", "node_type"}
        assert set(update.__annotations__) == set(partial.__annotations__) - fixed


# --- ZarrV3ConsolidatedMetadata --------------------------------------------


def test_consolidated_v3_roundtrip() -> None:
    """A v3 group with inline consolidated metadata round-trips, with child
    entries parsed into array/group models."""
    child = ZarrV3ArrayMetadata.create_default(shape=(2,)).to_json()
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"a": child, "g": {"zarr_format": 3, "node_type": "group"}},
        },
    }
    model = ZarrV3GroupMetadata.from_json(doc)
    assert isinstance(model.consolidated_metadata, ZarrV3ConsolidatedMetadata)
    assert isinstance(model.consolidated_metadata.metadata["a"], ZarrV3ArrayMetadata)
    assert isinstance(model.consolidated_metadata.metadata["g"], ZarrV3GroupMetadata)
    assert model.to_json() == doc


def test_consolidated_v3_must_understand_true_rejected() -> None:
    """ZarrV3ConsolidatedMetadata enforces must_understand=False at runtime."""
    with pytest.raises(ValueError, match="must_understand"):
        ZarrV3ConsolidatedMetadata(must_understand=True, metadata={})


def test_consolidated_v3_from_json_must_understand_true_rejected() -> None:
    """from_json rejects a consolidated document carrying must_understand=true."""
    with pytest.raises(MetadataValidationError, match="must_understand"):
        ZarrV3ConsolidatedMetadata.from_json(
            {"kind": "inline", "must_understand": True, "metadata": {}}
        )


def test_consolidated_v3_entry_without_node_type_rejected() -> None:
    """from_json rejects a consolidated entry lacking a recognizable node_type."""
    with pytest.raises(MetadataValidationError, match="node_type"):
        ZarrV3ConsolidatedMetadata.from_json(
            {"kind": "inline", "must_understand": False, "metadata": {"a": {"zarr_format": 3}}}
        )


def test_consolidated_v3_not_a_mapping() -> None:
    """from_json rejects a non-mapping consolidated document."""
    with pytest.raises(MetadataValidationError, match="expected an object"):
        ZarrV3ConsolidatedMetadata.from_json(5)


# --- read_group_metadata_v3 ------------------------------------------------

LITTLE: ZarrV3NamedConfigJSON = {"name": "bytes", "configuration": {"endian": "little"}}
GZIP_99 = {"name": "gzip", "configuration": {"level": 99}}

A: tuple[str, ...] = ("consolidated_metadata", "metadata", "a")
"""Where the document at path `a` in a group's consolidated metadata sits in the group's."""


def _array(**members: object) -> dict[str, object]:
    return {**ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json(), **members}


def _inline(**documents: object) -> dict[str, object]:
    return {"kind": "inline", "must_understand": False, "metadata": documents}


def _group(**members: object) -> dict[str, object]:
    return {"zarr_format": 3, "node_type": "group", **members}


def test_error_a_consolidated_envelope_reports_the_members_it_lacks_in_its_order() -> None:
    document = _group(consolidated_metadata={"kind": "inline"})
    assert [(p.loc, p.kind) for p in validate_group_metadata_v3(document)] == [
        (("consolidated_metadata", "must_understand"), "missing_key"),
        (("consolidated_metadata", "metadata"), "missing_key"),
    ]


ACME_X = CodecDefinition(
    name="acme.x", configuration=EmptyConfiguration, kind="bytes_bytes", size="dynamic"
)


def test_group_update_keeps_the_documents_it_holds() -> None:
    """Read in no scope again: one read in a scope the call's does not claim keeps each field as it was read."""
    scope = CORE_AND_EXTENSIONS.extended_with(ACME_X)
    child = ZarrV3ArrayMetadata.create_default(
        context=scope, shape=(4,), codecs=(LITTLE, {"name": "acme.x"})
    )
    group = ZarrV3GroupMetadata(
        attributes={},
        consolidated_metadata=ZarrV3ConsolidatedMetadata(metadata={"a": child}),
        extra_fields={},
    )
    updated = group.update(context=CORE_AND_EXTENSIONS, attributes={"k": 1})
    assert updated.attributes == {"k": 1}
    assert updated.consolidated_metadata is group.consolidated_metadata


def test_group_update_reads_the_documents_it_is_given_in_its_scope() -> None:
    """And `UNSET` leaves them out. `zstd` is an extension, which `CORE` leaves unclaimed."""
    zstd = {"name": "zstd", "configuration": {"level": 3, "checksum": False}}
    member = cast("JSONValue", _inline(a=_array(codecs=[LITTLE, zstd])))
    updated = ZarrV3GroupMetadata.create_default().update(
        context=CORE, consolidated_metadata=member
    )
    assert updated.consolidated_metadata is not UNSET
    child = updated.consolidated_metadata.metadata["a"]
    assert isinstance(child, ZarrV3ArrayMetadata)
    assert isinstance(child.codecs[1], Unclaimed)
    removed = updated.update(context=CORE, consolidated_metadata=UNSET)
    assert removed.consolidated_metadata is UNSET


def _fields_of_an_array(*at: str | int) -> list[tuple[str | int, ...]]:
    """Where a default array's fields sit, under `at`."""
    points = ("data_type", "chunk_grid", "chunk_key_encoding")
    return [*((*at, point) for point in points), (*at, "codecs", 0)]


@pytest.mark.parametrize(
    ("document", "paths", "locs"),
    [
        (_group(), [], []),
        # A null, which a historical zarr-python bug wrote, holds nothing.
        (_group(consolidated_metadata=None), [], []),
        (
            _group(consolidated_metadata=_inline(a=_array(), g=_group())),
            ["a", "g"],
            _fields_of_an_array(*A),
        ),
        # Every node below the group, each below a group, at its path.
        (
            _group(consolidated_metadata=_inline(g=_group(), **{"g/b": _array()})),
            ["g", "g/b"],
            _fields_of_an_array("consolidated_metadata", "metadata", "g/b"),
        ),
        # A group's consolidated metadata in a group's, listing what the group
        # lists too: each field located from the root of the outer document.
        (
            _group(
                consolidated_metadata=_inline(
                    g=_group(consolidated_metadata=_inline(b=_array())), **{"g/b": _array()}
                )
            ),
            ["g", "g/b"],
            [
                *_fields_of_an_array(
                    "consolidated_metadata",
                    "metadata",
                    "g",
                    "consolidated_metadata",
                    "metadata",
                    "b",
                ),
                *_fields_of_an_array("consolidated_metadata", "metadata", "g/b"),
            ],
        ),
    ],
    ids=["no-consolidated-metadata", "null", "an-array-and-a-group", "paths", "nested"],
)
def test_a_group_reads_each_document_its_consolidated_metadata_holds(
    document: dict[str, object], paths: list[str], locs: list[tuple[str | int, ...]]
) -> None:
    reading = read_group_metadata_v3(document)
    assert reading.problems == ()
    assert list(reading.consolidated) == paths
    assert [loc for loc, _ in reading.fields()] == locs
    model = reading.metadata
    assert model is not None
    assert model == ZarrV3GroupMetadata.from_json(document)
    # It holds the model each document's own reading built.
    consolidated = model.consolidated_metadata
    held = {} if consolidated is UNSET else consolidated.metadata
    assert all(held[path] is reading.consolidated[path].metadata for path in paths)


@pytest.mark.parametrize(
    ("path", "fault"),
    [
        ("", "is the group's own"),
        ("/a", 'starts with "/"'),
        ("a/", 'ends with "/"'),
        ("a//b", 'holds an empty name between two "/"'),
        (".", 'holds ".", a name that is periods alone'),
        ("a/../b", 'holds "..", a name that is periods alone'),
        ("__a", 'holds "__a", a name that starts with the reserved "__"'),
        ("zarr.json", 'holds "zarr.json", a name that is the reserved "zarr.json"'),
    ],
)
def test_error_a_document_in_consolidated_metadata_is_at_a_node_s_path_below_the_group(
    path: str, fault: str
) -> None:
    # Its node names, joined by "/", as the reference implementation keeps
    # it: the group's own path, "/", and it make the node's.
    document = _group(consolidated_metadata=_inline(**{path: _group()}))
    message = f"expected the path of a node below the group, got {json.dumps(path)}, which {fault}"
    assert validate_group_metadata_v3(document) == (
        ValidationProblem(("consolidated_metadata", "metadata", path), message, "invalid_value"),
    )


@pytest.mark.parametrize(
    ("documents", "path"),
    [
        ({"a": _array(), "a/b": _group()}, "a/b"),
        # However many groups are missing between them: none would help.
        ({"a": _array(), "a/b/c": _array()}, "a/b/c"),
    ],
    ids=["child", "descendant"],
)
def test_error_no_document_in_consolidated_metadata_is_below_an_array(
    documents: dict[str, object], path: str
) -> None:
    # "Group nodes may have children but array nodes may not." A message
    # names each node by its path in the hierarchy below the group.
    document = _group(consolidated_metadata=_inline(**documents))
    message = f'expected a node below a group, got "/{path}", below the array "/a"'
    assert validate_group_metadata_v3(document) == (
        ValidationProblem(("consolidated_metadata", "metadata", path), message, "invalid_value"),
    )


def test_a_document_of_no_node_type_is_taken_as_a_group_s() -> None:
    # Its own problem is reported where it sits, and the nodes below it are
    # not refused for it.
    document = _group(
        consolidated_metadata=_inline(a={"zarr_format": 3, "node_type": "x"}, **{"a/b": _array()})
    )
    assert [(p.loc, p.kind) for p in validate_group_metadata_v3(document)] == [
        (("consolidated_metadata", "metadata", "a", "node_type"), "invalid_value")
    ]


def test_error_consolidated_metadata_holds_the_group_holding_each_document() -> None:
    # The nearest group missing above a node, once, counting those above it
    # up to the group holding the documents.
    document = _group(consolidated_metadata=_inline(**{"a/b/c": _array(), "a/b/d": _array()}))
    assert [(p.loc, p.message, p.kind) for p in validate_group_metadata_v3(document)] == [
        (
            ("consolidated_metadata", "metadata", "a/b"),
            'missing the group holding "/a/b/c", and 1 group above it',
            "missing_key",
        ),
    ]


def test_error_a_consolidated_metadata_key_that_is_not_a_string() -> None:
    listing: dict[object, object] = {"a": _array(), 1: _array()}
    document = _group(consolidated_metadata={**_inline(), "metadata": listing})
    assert [(p.loc, p.kind) for p in validate_group_metadata_v3(document)] == [
        (("consolidated_metadata", "metadata"), "invalid_type")
    ]


NESTED = ("consolidated_metadata", "metadata", "g", "consolidated_metadata", "metadata")
"""Where the own listing of the group at `g` sits in the outer document."""


@pytest.mark.parametrize(
    ("listed", "flat", "expected"),
    [
        # What a listed group lists itself is what the group lists, too.
        ({"b": _array()}, {"g/b": _array()}, []),
        # A node the listed group lists alone would be dropped by the
        # reference reader, which keeps the flat listing.
        (
            {"b": _array()},
            {},
            [
                (
                    (*NESTED, "b"),
                    'expected a node the group lists, got "/g/b", which "/g" lists alone',
                    "invalid_value",
                )
            ],
        ),
        # Nor may the two listings disagree on what a node is.
        (
            {"b": _array()},
            {"g/b": _group()},
            [
                (
                    (*NESTED, "b"),
                    'expected a group, as the group lists "/g/b", got an array',
                    "invalid_value",
                )
            ],
        ),
        # A deeper listing is the listed group's own to judge, when its
        # document is read: its problem is that document's, at its place.
        (
            {"h": _group(consolidated_metadata=_inline(x=_array()))},
            {"g/h": _group()},
            [
                (
                    (*NESTED, "h", "consolidated_metadata", "metadata", "x"),
                    'expected a node the group lists, got "/h/x", which "/h" lists alone',
                    "invalid_value",
                )
            ],
        ),
    ],
    ids=["agreeing", "listed-alone", "contradicting", "deeper"],
)
def test_a_listed_group_s_own_listing_lists_what_the_group_lists(
    listed: dict[str, object],
    flat: dict[str, object],
    expected: list[tuple[tuple[str, ...], str, str]],
) -> None:
    document = _group(
        consolidated_metadata=_inline(g=_group(consolidated_metadata=_inline(**listed)), **flat)
    )
    problems = validate_group_metadata_v3(document)
    assert [(p.loc, p.message, p.kind) for p in problems] == expected
    # The constructor refuses what the reader reports, built of models: a
    # listed group whose own listing is wrong is refused as it is built.
    listing = {"g": _group(consolidated_metadata=_inline(**listed)), **flat}
    deeper = [(loc, message, kind) for loc, message, kind in expected if len(loc) > len(NESTED) + 1]
    if len(deeper) != 0:
        with pytest.raises(MetadataValidationError) as inner:
            node_metadata_from_json_v3(listing["g"])
        assert [(p.loc, p.message, p.kind) for p in inner.value.problems] == [
            (loc[3:], message, kind) for loc, message, kind in deeper
        ]
        return
    members = {key: node_metadata_from_json_v3(value) for key, value in listing.items()}
    if expected == []:
        assert ZarrV3ConsolidatedMetadata(metadata=members).metadata.keys() == members.keys()
        return
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ConsolidatedMetadata(metadata=members)
    assert [(p.loc[1:], p.message, p.kind) for p in raised.value.problems] == [
        (loc[2:], message, kind) for loc, message, kind in expected
    ]


def test_each_document_its_consolidated_metadata_holds_is_read_once() -> None:
    reads: list[GzipCodecConfiguration] = []

    def counted(
        configuration: GzipCodecConfiguration, nested: Nested
    ) -> Iterator[ValidationProblem]:
        reads.append(configuration)
        yield from ()

    scope = CORE_AND_EXTENSIONS.extended_with(dataclasses.replace(GZIP_CODEC, rules=counted))
    array = _array(codecs=[LITTLE, {"name": "gzip", "configuration": {"level": 5}}])
    group = _group(consolidated_metadata=_inline(a=array))
    assert read_group_metadata_v3(group, context=scope).problems == ()
    assert reads == [{"level": 5}]
    ZarrV3GroupMetadata.from_json(group, context=scope)
    assert len(reads) == 2


def test_error_a_group_document_that_is_not_an_object_reads_as_nothing() -> None:
    not_an_object = ValidationProblem((), "expected an object", "invalid_type")
    assert read_group_metadata_v3([1]) == ZarrV3GroupMetadataReading(problems=(not_an_object,))


@pytest.mark.parametrize(
    ("document", "problems", "models", "refused"),
    [
        # The group's own member: each document it holds still has its model.
        (
            _group(attributes=5, consolidated_metadata=_inline(a=_array())),
            [(("attributes",), "invalid_type")],
            {"a": True},
            [],
        ),
        # A document it holds: that one has none, and its sibling has one;
        # the field refused is found where it sits, like every field.
        (
            _group(consolidated_metadata=_inline(a=_array(codecs=[LITTLE, GZIP_99]), b=_array())),
            [((*A, "codecs", 1, "configuration", "level"), "invalid_value")],
            {"a": False, "b": True},
            [(*A, "codecs", 1)],
        ),
        # One of no node type is read as none, and nothing else of it is.
        (
            _group(consolidated_metadata=_inline(a={"zarr_format": 3})),
            [((*A, "node_type"), "missing_key")],
            {"a": False},
            [],
        ),
    ],
    ids=["group", "consolidated-document", "no-node-type"],
)
def test_error_a_group_document_with_a_problem_reads_as_no_model(
    document: dict[str, object],
    problems: list[tuple[tuple[str | int, ...], str]],
    models: dict[str, bool],
    refused: list[tuple[str | int, ...]],
) -> None:
    reading = read_group_metadata_v3(document)
    assert [(p.loc, p.kind) for p in reading.problems] == problems
    assert reading.metadata is None
    assert {
        path: read.metadata is not None for path, read in reading.consolidated.items()
    } == models
    assert [loc for loc, field in reading.fields() if isinstance(field, Refused)] == refused


# --- read_node_metadata_v3 -------------------------------------------------


@pytest.mark.parametrize(
    ("document", "reading", "problems"),
    [
        (_array(), ZarrV3ArrayMetadataReading, []),
        (_group(attributes={"a": 1}), ZarrV3GroupMetadataReading, []),
        (_array(fill_value=300), ZarrV3ArrayMetadataReading, [(("fill_value",), "invalid_value")]),
        (_group(attributes=5), ZarrV3GroupMetadataReading, [(("attributes",), "invalid_type")]),
    ],
    ids=["array", "group", "array-with-a-problem", "group-with-a-problem"],
)
def test_a_node_is_read_as_the_node_its_node_type_says(
    document: dict[str, object],
    reading: type[ZarrV3ArrayMetadataReading | ZarrV3GroupMetadataReading],
    problems: list[tuple[tuple[str | int, ...], str]],
) -> None:
    """As that node's own read reads it, and its model only when it has no problem."""
    read = read_node_metadata_v3(document)
    assert type(read) is reading
    assert [(p.loc, p.kind) for p in read.problems] == problems
    assert [(p.loc, p.kind) for p in validate_node_metadata_v3(document)] == problems
    assert (read.metadata is not None) is (len(problems) == 0)


@pytest.mark.parametrize(
    ("node_type", "kind"),
    [("dataset", "invalid_value"), (5, "invalid_type"), (None, "invalid_type")],
    ids=["another-kind", "a-number", "null"],
)
def test_error_a_node_type_the_spec_does_not_define(node_type: object, kind: str) -> None:
    """Nothing else of the document is read but its `zarr_format`, which is 3 here, so nothing else is judged."""
    read = read_node_metadata_v3({**_array(), "node_type": node_type, "shape": "not a shape"})
    assert isinstance(read, ZarrV3UnknownNodeReading)
    assert [(p.loc, p.kind) for p in read.problems] == [(("node_type",), kind)]
    assert read.metadata is None
    assert list(read.fields()) == []


def test_error_a_document_without_a_node_type() -> None:
    document = {key: value for key, value in _array().items() if key != "node_type"}
    read = read_node_metadata_v3(document)
    assert isinstance(read, ZarrV3UnknownNodeReading)
    assert [(p.loc, p.kind) for p in read.problems] == [(("node_type",), "missing_key")]


@pytest.mark.parametrize(
    ("document", "problems"),
    [
        # zarr-python 2's draft of v3 (zarr-python#2982): a root `zarr.json`
        # naming its format by URL, and no node type.
        (
            {
                "zarr_format": "https://purl.org/zarr/spec/protocol/core/3.0",
                "metadata_encoding": "https://purl.org/zarr/spec/protocol/core/3.0",
                "metadata_key_suffix": ".json",
                "extensions": [],
            },
            [(("zarr_format",), "invalid_type"), (("node_type",), "missing_key")],
        ),
        (
            {"zarr_format": 2, "shape": [4], "chunks": [4], "dtype": "|u1"},
            [(("zarr_format",), "invalid_value"), (("node_type",), "missing_key")],
        ),
        (
            {"node_type": "dataset"},
            [(("zarr_format",), "missing_key"), (("node_type",), "invalid_value")],
        ),
    ],
    ids=["v3-draft", "v2", "no-format"],
)
def test_error_a_document_of_another_format_says_so(
    document: dict[str, object], problems: list[tuple[tuple[str | int, ...], str]]
) -> None:
    read = read_node_metadata_v3(document)
    assert isinstance(read, ZarrV3UnknownNodeReading)
    assert [(p.loc, p.kind) for p in read.problems] == problems
    # The validator reads a node's type as the reader does.
    assert [(p.loc, p.kind) for p in validate_node_metadata_v3(document)] == problems


def test_error_a_node_that_is_not_an_object() -> None:
    read = read_node_metadata_v3([_array()])
    assert isinstance(read, ZarrV3UnknownNodeReading)
    assert [(p.loc, p.kind) for p in read.problems] == [((), "invalid_type")]


@pytest.mark.parametrize(
    "model",
    [
        ZarrV3ArrayMetadata.create_default(shape=(4,)),
        ZarrV3GroupMetadata.create_default(attributes={"a": 1}),
    ],
    ids=["array", "group"],
)
def test_a_node_is_built_as_the_model_its_node_type_says(
    model: ZarrV3ArrayMetadata | ZarrV3GroupMetadata,
) -> None:
    """From its JSON and from a store's bytes alike, as the model's own class builds it."""
    built = node_metadata_from_key_value_v3(model.to_key_value())
    assert type(built) is type(model)
    assert built == model
    assert node_metadata_from_json_v3(model.to_json()) == model


def test_error_a_node_built_from_a_store_without_its_document() -> None:
    with pytest.raises(MetadataValidationError) as raised:
        node_metadata_from_key_value_v3({})
    assert [(p.loc, p.kind) for p in raised.value.problems] == [(("zarr.json",), "missing_key")]


def test_error_a_node_built_from_bytes_that_are_not_json() -> None:
    with pytest.raises(MetadataValidationError) as raised:
        node_metadata_from_key_value_v3({"zarr.json": b"{"})
    assert [(p.loc, p.kind) for p in raised.value.problems] == [(("zarr.json",), "invalid_json")]


def test_error_a_node_built_from_a_document_of_no_node_type() -> None:
    document = json.dumps({**_array(), "node_type": "dataset"}).encode()
    with pytest.raises(MetadataValidationError) as raised:
        node_metadata_from_key_value_v3({"zarr.json": document})
    assert [(p.loc, p.kind) for p in raised.value.problems] == [(("node_type",), "invalid_value")]


def test_error_a_node_built_from_a_document_with_a_problem() -> None:
    """The problems of the node its `node_type` says it is."""
    with pytest.raises(MetadataValidationError) as raised:
        node_metadata_from_json_v3(_array(fill_value=300))
    assert [(p.loc, p.kind) for p in raised.value.problems] == [(("fill_value",), "invalid_value")]


# --- ZarrV2ConsolidatedMetadata --------------------------------------------


def test_consolidated_v2_verbatim_roundtrip() -> None:
    """The v2 .zmetadata model holds the flat file-keyed map verbatim,
    including nodes that have no .zattrs entry."""
    doc = {
        "zarr_consolidated_format": 1,
        "metadata": {
            ".zgroup": {"zarr_format": 2},
            "a/.zarray": {
                "zarr_format": 2,
                "shape": (2,),
                "chunks": (2,),
                "dtype": "|u1",
                "fill_value": 0,
                "order": "C",
                "compressor": None,
                "filters": None,
            },
        },
    }
    model = ZarrV2ConsolidatedMetadata.from_json(doc)
    assert model.to_json() == doc


@pytest.mark.parametrize(("value", "kind"), [("1", "invalid_type"), (2, "invalid_value")])
def test_error_consolidated_v2_format_other_than_1(value: object, kind: str) -> None:
    document = {"zarr_consolidated_format": value, "metadata": {}}
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ConsolidatedMetadata.from_json(document)
    assert [(p.loc, p.kind) for p in raised.value.problems] == [
        (("zarr_consolidated_format",), kind)
    ]


def test_consolidated_v2_key_value_roundtrip() -> None:
    """from_key_value(to_key_value()) is the identity for .zmetadata documents."""
    model = ZarrV2ConsolidatedMetadata.from_json(
        {"zarr_consolidated_format": 1, "metadata": {".zgroup": {"zarr_format": 2}}}
    )
    assert ZarrV2ConsolidatedMetadata.from_key_value(model.to_key_value()) == model


def test_consolidated_v2_lists_become_tuples() -> None:
    """from_json converts JSON arrays inside entries to tuples."""
    doc = {
        "zarr_consolidated_format": 1,
        "metadata": {"a/.zarray": {"shape": [2, 3]}},
    }
    model = ZarrV2ConsolidatedMetadata.from_json(doc)
    assert model.metadata == {"a/.zarray": {"shape": (2, 3)}}


def test_consolidated_v2_envelope_validation() -> None:
    """from_json rejects a .zmetadata document missing the metadata key."""
    with pytest.raises(MetadataValidationError, match="metadata"):
        ZarrV2ConsolidatedMetadata.from_json({"zarr_consolidated_format": 1})


def test_consolidated_v2_not_a_mapping() -> None:
    """from_json rejects a non-mapping .zmetadata document."""
    with pytest.raises(MetadataValidationError, match="expected an object"):
        ZarrV2ConsolidatedMetadata.from_json([1])


def test_consolidated_v2_format_literal_enforced() -> None:
    """A .zmetadata document must declare consolidated format 1."""
    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV2ConsolidatedMetadata.from_json({"zarr_consolidated_format": 2, "metadata": {}})
    assert [(problem.loc, problem.kind) for problem in exc_info.value.problems] == [
        (("zarr_consolidated_format",), "invalid_value")
    ]


def test_consolidated_v2_metadata_values_must_be_json() -> None:
    """Non-JSON values in the flat metadata map are rejected during ingestion."""
    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV2ConsolidatedMetadata.from_json(
            {"zarr_consolidated_format": 1, "metadata": {".zgroup": object()}}
        )
    assert [(problem.loc, problem.kind) for problem in exc_info.value.problems] == [
        (("metadata", ".zgroup"), "invalid_type")
    ]


@pytest.mark.parametrize(
    "stored",
    [
        '{"zarr_format": 2}',
        None,
        memoryview(b'{"zarr_format": 2}'),
        bytearray(b'{"zarr_format": 2}'),
    ],
    ids=["str", "none", "memoryview", "bytearray"],
)
def test_error_a_store_value_that_is_not_bytes_is_refused(stored: object) -> None:
    """A store maps keys to bytes, and a reader checks that it read bytes."""
    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV2GroupMetadata.from_key_value({".zgroup": stored})  # pyright: ignore[reportArgumentType]
    assert [(problem.loc, problem.kind) for problem in exc_info.value.problems] == [
        ((".zgroup",), "invalid_type")
    ]


def test_group_v2_from_key_value_scalar_root_raises_metadata_error() -> None:
    """A scalar .zgroup document fails through the unified metadata error channel."""
    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV2GroupMetadata.from_key_value({".zgroup": b"null"})
    assert [(problem.loc, problem.kind) for problem in exc_info.value.problems] == [
        ((), "invalid_type")
    ]


# --- Literal-value enforcement -----------------------------------------------


def test_group_v3_literals_enforced() -> None:
    """A v3 group doc with wrong zarr_format or node_type is rejected with invalid_value."""
    base = ZarrV3GroupMetadata.create_default().to_json()
    for key, bad in (("zarr_format", 2), ("node_type", "array")):
        problems = validate_group_metadata_v3(dict(base) | {key: bad})
        assert [(p.loc, p.kind) for p in problems] == [((key,), "invalid_value")], key


def test_group_v2_zarr_format_literal_enforced() -> None:
    """A v2 group doc claiming zarr_format 3 is rejected with invalid_value."""
    problems = validate_group_metadata_v2({"zarr_format": 3})
    assert [(p.loc, p.kind) for p in problems] == [(("zarr_format",), "invalid_value")]


# --- Consolidated envelope validated by the group validator ------------------


def test_group_v3_validator_agrees_with_from_json_on_consolidated() -> None:
    """The group validator validates the consolidated envelope and entries, so
    is_group_metadata_v3 never vouches for a document from_json would reject."""
    bad_docs = (
        # empty envelope: missing kind/must_understand/metadata
        {"zarr_format": 3, "node_type": "group", "consolidated_metadata": {}},
        # entry without a recognizable node_type
        {
            "zarr_format": 3,
            "node_type": "group",
            "consolidated_metadata": {
                "kind": "inline",
                "must_understand": False,
                "metadata": {"a": {"zarr_format": 3}},
            },
        },
        # must_understand: true
        {
            "zarr_format": 3,
            "node_type": "group",
            "consolidated_metadata": {
                "kind": "inline",
                "must_understand": True,
                "metadata": {},
            },
        },
    )
    for doc in bad_docs:
        assert validate_group_metadata_v3(doc) != (), doc
        with pytest.raises(MetadataValidationError):
            ZarrV3GroupMetadata.from_json(doc)


def test_group_v3_valid_consolidated_passes_validator() -> None:
    """A well-formed consolidated group validates cleanly (control case)."""
    child = ZarrV3ArrayMetadata.create_default(shape=(2,)).to_json()
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"a": child, "g": {"zarr_format": 3, "node_type": "group"}},
        },
    }
    assert validate_group_metadata_v3(doc) == ()


def test_v3_consolidated_rejects_unknown_envelope_member() -> None:
    """The inline consolidated envelope is closed and never drops accepted members."""
    doc = {
        "kind": "inline",
        "must_understand": False,
        "metadata": {},
        "unexpected": 1,
    }

    with pytest.raises(MetadataValidationError, match="unexpected"):
        ZarrV3ConsolidatedMetadata.from_json(doc)


def test_v2_consolidated_rejects_unknown_document_member() -> None:
    """The v2 consolidated document is closed and never drops accepted members."""
    doc = {"zarr_consolidated_format": 1, "metadata": {}, "unexpected": 1}

    with pytest.raises(MetadataValidationError, match="unexpected"):
        ZarrV2ConsolidatedMetadata.from_json(doc)


@pytest.mark.parametrize("key", [1, None], ids=["int", "none"])
def test_v2_consolidated_rejects_non_string_document_key(key: object) -> None:
    """A non-string key is a problem at the document: not a member at a
    location that reads as an index, nor a `TypeError`."""
    doc = {"zarr_consolidated_format": 1, "metadata": {}, key: "x"}

    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV2ConsolidatedMetadata.from_json(doc)
    assert [(problem.loc, problem.kind) for problem in exc_info.value.problems] == [
        ((), "invalid_type")
    ]


# --- must_understand partition ------------------------------------------------


def test_group_must_understand_fields_partition() -> None:
    """The group model partitions extra fields by the spec's implicit-true rule,
    like the array model.

    https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1573
    """
    model = ZarrV3GroupMetadata.create_default(
        waived={"name": "w", "must_understand": False}, implicit={"name": "i"}
    )
    assert set(model.must_understand_fields) == {"implicit"}


def test_group_v3_null_consolidated_metadata_repaired_to_absence() -> None:
    """consolidated_metadata: null was written by a historical zarr-python bug.
    Those stores must remain readable, but the bug spelling is not honored:
    it is read as absence (UNSET) and never written back — the round-trip
    deliberately repairs the document rather than preserving the bug."""
    null_doc = {"zarr_format": 3, "node_type": "group", "consolidated_metadata": None}
    assert validate_group_metadata_v3(null_doc) == ()
    model = ZarrV3GroupMetadata.from_json(null_doc)
    assert model.consolidated_metadata is UNSET
    assert "consolidated_metadata" not in model.to_json()
    assert model == ZarrV3GroupMetadata.from_json({"zarr_format": 3, "node_type": "group"})


# --- to_json shares no mutable state with the model ------------------------

TO_JSON_NO_ALIASING_PARAMS = [
    pytest.param(
        ZarrV3GroupMetadata(
            attributes={"a": {"b": [1]}},
            consolidated_metadata=ZarrV3ConsolidatedMetadata(
                metadata={
                    "child": ZarrV3ArrayMetadata.create_default(attributes={"x": {"y": 1}}),
                    "grp": ZarrV3GroupMetadata.create_default(attributes={"x": {"y": 1}}),
                }
            ),
            extra_fields={"ext": {"must_understand": False, "cfg": {"x": [1]}}},
        ),
        id="v3-group",
    ),
    pytest.param(
        ZarrV2GroupMetadata.create_default(attributes={"a": {"b": [1]}}),
        id="v2-group",
    ),
    pytest.param(
        ZarrV3ConsolidatedMetadata(
            metadata={"child": ZarrV3ArrayMetadata.create_default(attributes={"x": {"y": 1}})}
        ),
        id="v3-consolidated",
    ),
    pytest.param(
        ZarrV2ConsolidatedMetadata(metadata={"a/.zarray": {"nested": {"x": [1]}}}),
        id="v2-consolidated",
    ),
]


@pytest.mark.parametrize("model", TO_JSON_NO_ALIASING_PARAMS)
def test_to_json_shares_no_mutable_state_with_model(
    model: ZarrV3GroupMetadata
    | ZarrV2GroupMetadata
    | ZarrV3ConsolidatedMetadata
    | ZarrV2ConsolidatedMetadata,
) -> None:
    """Mutating a document returned by to_json leaves the model unchanged."""
    baseline = copy.deepcopy(model.to_json())
    mutate_nested_containers(model.to_json())
    assert model.to_json() == baseline


@pytest.mark.parametrize("model", TO_JSON_NO_ALIASING_PARAMS)
def test_from_json_shares_no_mutable_state_with_its_input(
    model: ZarrV3GroupMetadata
    | ZarrV2GroupMetadata
    | ZarrV3ConsolidatedMetadata
    | ZarrV2ConsolidatedMetadata,
) -> None:
    """Mutating the document a model was read from leaves the model unchanged."""
    # Arrays as tuples: the reader has nothing to rebuild, so only a copy
    # keeps the model apart from its input.
    document = arrays_to_tuples(model.to_json())
    read = type(model).from_json(document)
    baseline = copy.deepcopy(read.to_json())
    mutate_nested_containers(document)
    assert read.to_json() == baseline
