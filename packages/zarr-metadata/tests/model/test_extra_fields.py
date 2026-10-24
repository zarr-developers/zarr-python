"""Members of a v3 document that the spec does not define: extra fields, read by the definitions a scope holds for them."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, NotRequired

import pytest
from annotated_types import Ge
from jsonschema import Draft202012Validator
from typing_extensions import TypedDict

from zarr_metadata.definition import AcceptedField, UnclaimedField, resolve
from zarr_metadata.json_schema import field_json_schema_v3
from zarr_metadata.model import ZarrV3ArrayMetadata, ZarrV3GroupMetadata
from zarr_metadata.problems import MetadataValidationError
from zarr_metadata.typed_json import UserJSONValue
from zarr_metadata.v3.definition import (
    CORE,
    ArrayExtraFieldDefinition,
    GroupExtraFieldDefinition,
    ZarrV3Context,
)


class AcmeTilingConfiguration(TypedDict, closed=True):
    """The value of an `acme.tiling` member: a tile edge, and the waiver that lets other readers skip it."""

    edge: Annotated[int, Ge(1)]
    must_understand: bool


ACME_TILING = ArrayExtraFieldDefinition(name="acme.tiling", configuration=AcmeTilingConfiguration)
"""A reader's own definition of the `acme.tiling` member of an array."""


class AcmeNoteConfiguration(TypedDict, closed=True):
    """The value of an `acme.note` member: any JSON, and the waiver."""

    note: UserJSONValue
    must_understand: NotRequired[bool]


ACME_NOTE = ArrayExtraFieldDefinition(name="acme.note", configuration=AcmeNoteConfiguration)
"""A reader's own definition of the `acme.note` member of an array, whose numbers must be finite."""

CONTEXT = CORE.extended_with(array_extra_fields=[ACME_TILING, ACME_NOTE])

ARRAY: dict[str, Any] = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
GROUP: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}
TILING = {"edge": 2, "must_understand": True}
INLINE = {"kind": "inline", "must_understand": False, "metadata": {}}
NAN_LISTING = {**INLINE, "metadata": {"a": {**ARRAY, "attributes": {"x": math.nan}}}}


@pytest.mark.parametrize(
    ("document", "context", "claimed", "must_understand"),
    [
        ({**ARRAY, "acme.tiling": TILING}, CONTEXT, {"acme.tiling"}, {}),
        ({**ARRAY, "acme.tiling": TILING}, CORE, set(), {"acme.tiling": TILING}),
        ({**ARRAY, "acme.other": {"must_understand": False}}, CONTEXT, set(), {}),
        ({**ARRAY, "acme.other": 5}, CONTEXT, set(), {"acme.other": 5}),
        ({**GROUP, "consolidated_metadata": INLINE}, CORE, {"consolidated_metadata"}, {}),
        ({**GROUP, "consolidated_metadata": INLINE}, ZarrV3Context(), set(), {}),
        (
            {**GROUP, "consolidated_metadata": None},
            ZarrV3Context(),
            set(),
            {"consolidated_metadata": None},
        ),
        ({**ARRAY, "consolidated_metadata": INLINE}, CORE, set(), {}),
        ({**GROUP, "consolidated_metadata": NAN_LISTING}, CORE, {"consolidated_metadata"}, {}),
        ({**GROUP, "consolidated_metadata": NAN_LISTING}, ZarrV3Context(), set(), {}),
        ({**ARRAY, "acme.other": {"x": math.inf}}, CONTEXT, set(), {"acme.other": {"x": math.inf}}),
    ],
    ids=[
        "claimed",
        "unclaimed",
        "unclaimed-waived",
        "unclaimed-not-an-object",
        "consolidated-in-core",
        "consolidated-unclaimed",
        "null-consolidated-unclaimed",
        "a-group-field-on-an-array",
        "non-finite-listing-claimed",
        "non-finite-listing-unclaimed",
        "non-finite-unclaimed",
    ],
)
def test_a_scope_reads_the_extra_fields_it_claims(
    document: dict[str, Any],
    context: ZarrV3Context,
    claimed: set[str],
    must_understand: dict[str, object],
) -> None:
    """Each member the spec does not define is an extra field. A member whose key a definition in scope claims is an `AcceptedField`, and a reader understands it. Any other member is an `UnclaimedField`, kept as written, and `must_understand_fields` lists it unless it says `"must_understand": false`. Nothing reads an unclaimed member, so like attributes it may hold any number, finite or not. A definition of a group's extra field does not read a member of an array, and `CORE` reads a group's `consolidated_metadata`. The document is written back as it was read."""
    model: ZarrV3ArrayMetadata | ZarrV3GroupMetadata
    if document["node_type"] == "array":
        model = ZarrV3ArrayMetadata(document, context=context)
    else:
        model = ZarrV3GroupMetadata(document, context=context)
    accepted = {
        key for key, extra in model.extra_fields.items() if isinstance(extra, AcceptedField)
    }
    assert accepted == claimed
    assert all(
        isinstance(extra, UnclaimedField)
        for key, extra in model.extra_fields.items()
        if key not in claimed
    )
    assert _same_json(model.must_understand_fields, must_understand)
    assert _same_json(model.to_json(), type(model)(model.to_json(), context=context).to_json())
    for key, extra in model.extra_fields.items():
        assert _same_json(extra.to_json(), {key: model.to_json()[key]})
        read, problems = resolve({key: model.to_json()[key]}, extra.read_as, context)
        assert problems == ()
        assert type(read) is type(extra)


def _same_json(a: object, b: object) -> bool:
    """Whether `a` and `b` are the same JSON, counting NaN as equal to itself and a read-only view as the object it shows."""
    return json.dumps(a, default=dict, sort_keys=True) == json.dumps(
        b, default=dict, sort_keys=True
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [(1, 2), ("a", "b"), (None, 0), ((1,), (2,)), ({"x": 1}, {"x": 2}), ({}, None)],
    ids=["numbers", "strings", "null", "arrays", "objects", "empty-object-or-null"],
)
def test_unclaimed_extra_fields_compare_as_written(first: object, second: object) -> None:
    """Nothing reads an unclaimed member, so documents that write it differently are different documents: they compare unequal, and neither refines the other, whatever JSON the member holds."""
    a = ZarrV3ArrayMetadata({**ARRAY, "acme.flag": first})
    b = ZarrV3ArrayMetadata({**ARRAY, "acme.flag": second})
    assert a != b
    assert not a.refines(b)
    assert not b.refines(a)
    assert a == ZarrV3ArrayMetadata({**ARRAY, "acme.flag": first})


def test_an_extra_field_is_a_field_of_the_reading() -> None:
    """A claimed extra field takes part in what the model's reading holds: it is a field at its key, a claim of its kind and name, and `resolve` reads it as the member alone, `{key: value}`. The JSON Schema of the kind accepts the member."""
    model = ZarrV3ArrayMetadata({**ARRAY, "acme.tiling": TILING}, context=CONTEXT)
    assert dict(model.reading.fields())[("acme.tiling",)] == model.extra_fields["acme.tiling"]
    assert model.claims[ArrayExtraFieldDefinition, "acme.tiling"] is ACME_TILING
    read, problems = resolve({"acme.tiling": TILING}, ArrayExtraFieldDefinition, CONTEXT)
    assert isinstance(read, AcceptedField)
    assert problems == ()
    schema = field_json_schema_v3(ArrayExtraFieldDefinition, context=CONTEXT)
    assert Draft202012Validator(schema).is_valid({"acme.tiling": TILING})
    assert not Draft202012Validator(schema).is_valid({"acme.tiling": {"edge": 0}})


def test_error_a_claimed_extra_field_that_is_not_an_object() -> None:
    """A definition reads a member whose value is an object. Any other value of a claimed member is a problem at the member's key, as `CORE` refuses a `consolidated_metadata` of null."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3GroupMetadata({**GROUP, "consolidated_metadata": None}, context=CORE)
    assert [(p.loc, p.kind) for p in raised.value.problems] == [
        (("consolidated_metadata",), "invalid_type")
    ]


def test_error_a_claimed_extra_field_with_a_bad_configuration() -> None:
    """A claimed member's value is checked against its definition's configuration, and each problem is located in the member, under its key."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ArrayMetadata(
            {**ARRAY, "acme.tiling": {"edge": 0, "must_understand": True}}, context=CONTEXT
        )
    assert [(p.loc, p.kind) for p in raised.value.problems] == [
        (("acme.tiling", "edge"), "invalid_value")
    ]


def test_error_a_non_finite_number_in_an_extra_field_that_holds_no_user_data() -> None:
    """Every number in a claimed extra field must be finite unless its definition reads user data, as `CONSOLIDATED_METADATA_EXTRA_FIELD` does for the attributes of the documents it lists."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ArrayMetadata({**ARRAY, "acme.note": {"note": [math.nan]}}, context=CONTEXT)
    assert [(p.loc, p.kind) for p in raised.value.problems] == [
        (("acme.note", "note", 0), "invalid_value")
    ]


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        (ArrayExtraFieldDefinition, "shape"),
        (ArrayExtraFieldDefinition, "attributes"),
        (GroupExtraFieldDefinition, "attributes"),
        (GroupExtraFieldDefinition, "node_type"),
    ],
)
def test_error_an_extra_field_named_after_a_member_the_spec_defines(kind: Any, name: str) -> None:  # noqa: ANN401 - either extra-field kind
    """A member the spec defines is never an extra field, so a definition named after one would never read anything: it is refused when it is built, as a data type named like raw bits is."""
    with pytest.raises(TypeError, match="the spec defines"):
        kind(name=name, configuration=AcmeTilingConfiguration)


def test_error_an_extra_field_configuration_that_does_not_declare_must_understand() -> None:
    """The spec lets every extension member carry `must_understand`, so the configuration of an extra field declares it. A definition whose configuration does not is refused when it is built, rather than refusing the documents that waive it."""

    class Bare(TypedDict, closed=True):
        tool: str

    with pytest.raises(TypeError, match="must_understand"):
        ArrayExtraFieldDefinition(name="acme.bare", configuration=Bare)


def test_error_finite_is_a_boolean() -> None:
    """`finite` decides whether a member may hold non-finite numbers, so it must be a boolean: anything else is refused when the definition is built."""
    with pytest.raises(TypeError, match="finite"):
        ArrayExtraFieldDefinition(
            name="acme.tiling",
            configuration=AcmeTilingConfiguration,
            finite=0,  # pyright: ignore[reportArgumentType]
        )


def test_error_an_extra_field_of_a_group_is_not_an_extra_field_of_an_array() -> None:
    """The slots of a scope hold extra fields by node type, so a group's extra field given in the array slot is refused, as any definition of another kind is."""
    tiling = GroupExtraFieldDefinition(name="acme.tiling", configuration=AcmeTilingConfiguration)
    with pytest.raises(TypeError, match="array extra field definitions"):
        ZarrV3Context(array_extra_fields=[tiling])  # pyright: ignore[reportArgumentType]
