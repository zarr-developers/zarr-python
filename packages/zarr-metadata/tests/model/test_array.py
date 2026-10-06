"""Tests for the metadata models in ``zarr_metadata.model``."""

import copy
import dataclasses
import json
import math
import pickle
from collections import UserDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeGuard, cast, get_args, get_origin, get_type_hints

import pytest
from typing_extensions import Unpack

from tests.model._cases import Expect, ExpectFail, mutate_nested_containers
from zarr_metadata._json import JSON_DEPTH, arrays_to_tuples, json_text, prefixed
from zarr_metadata.model import (
    ARRAY_METADATA_OPTIONAL_KEYS_V3,
    ARRAY_METADATA_REQUIRED_KEYS_V3,
    ARRAY_METADATA_STANDARD_KEYS_V3,
    UNSET,
    MetadataValidationError,
    ValidationProblem,
    ZarrV2ArrayMetadata,
    ZarrV2ArrayMetadataPartial,
    ZarrV3ArrayMetadata,
    ZarrV3GroupMetadata,
    is_array_metadata_v2,
    is_array_metadata_v3,
    is_group_metadata_v2,
    is_group_metadata_v3,
    is_json,
    is_metadata_field_v3,
    parse_array_metadata_v2,
    parse_array_metadata_v3,
    parse_json,
    parse_metadata_field_v3,
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_json,
    validate_metadata_field_v3,
)
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSONPartial
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    Read,
    Unclaimed,
    canonical_of,
    configuration_of,
    fields_of,
    with_problems,
)

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue, ZarrV3NamedConfigJSON
    from zarr_metadata.v2 import ZarrV2CodecMetadata

# --- public exports --------------------------------------------------------


def test_guards_exported_from_package() -> None:
    """The wire-type guard/parser functions are exported from the package."""
    import zarr_metadata.model

    for name in (
        "is_json",
        "parse_json",
        "is_metadata_field_v3",
        "parse_metadata_field_v3",
        "is_array_metadata_v3",
        "parse_array_metadata_v3",
        "is_array_metadata_v2",
        "parse_array_metadata_v2",
    ):
        assert name in zarr_metadata.model.__all__
        assert hasattr(zarr_metadata.model, name)


# `ZARR_V3_CONSOLIDATED_METADATA_KEY` is deliberately absent: it names a key
# *inside* a v3 group document, not a store key, so it has no paired `Literal`
# and no `to_key_value` signature to appear in. See `test_v3_consolidated_key_
# is_not_a_store_key`, which pins that distinction.
STORE_KEY_PAIRS = [
    ("ZARR_V2_ARRAY_METADATA_STORE_KEY", "ZarrV2ArrayMetadataStoreKey", "zarr_metadata.v2.array"),
    ("ZARR_V3_ARRAY_METADATA_STORE_KEY", "ZarrV3ArrayMetadataStoreKey", "zarr_metadata.v3.array"),
    ("ZARR_V2_ATTRIBUTES_STORE_KEY", "ZarrV2AttributesStoreKey", "zarr_metadata.v2.attributes"),
    ("ZARR_V2_GROUP_METADATA_STORE_KEY", "ZarrV2GroupMetadataStoreKey", "zarr_metadata.v2.group"),
    ("ZARR_V3_GROUP_METADATA_STORE_KEY", "ZarrV3GroupMetadataStoreKey", "zarr_metadata.v3.group"),
    (
        "ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY",
        "ZarrV2ConsolidatedMetadataStoreKey",
        "zarr_metadata.v2.consolidated",
    ),
]


def test_store_key_pairs_exported_from_package() -> None:
    """Each store-key constant is exported together with its Literal type
    alias, and the pair cannot drift apart."""
    import zarr_metadata.model as m

    for const_name, alias_name, _ in STORE_KEY_PAIRS:
        assert const_name in m.__all__
        assert alias_name in m.__all__
        assert (getattr(m, const_name),) == get_args(getattr(m, alias_name))


def test_store_keys_are_defined_in_their_spec_modules() -> None:
    """Store keys are facts about the on-disk specs, so each is defined in the
    `v2`/`v3` module describing that document — not in the model layer, which
    only re-exports them."""
    import importlib

    for const_name, alias_name, module_name in STORE_KEY_PAIRS:
        module = importlib.import_module(module_name)
        for name in (const_name, alias_name):
            assert name in module.__all__, f"{name} should be exported by {module_name}"


def test_v3_consolidated_key_is_not_a_store_key() -> None:
    """v3 consolidated metadata is embedded as a field inside the group's own
    `zarr.json`, not persisted under its own store key. It therefore has no
    paired `Literal` alias, unlike every true store key — which is why it is
    excluded from `STORE_KEY_PAIRS` rather than merely forgotten."""
    import zarr_metadata.model as m

    assert "ZARR_V3_CONSOLIDATED_METADATA_KEY" in m.__all__
    assert not hasattr(m, "ZarrV3ConsolidatedMetadataKey")
    assert m.ZARR_V3_CONSOLIDATED_METADATA_KEY not in {
        getattr(m, const_name) for const_name, _, _ in STORE_KEY_PAIRS
    }


def test_v3_node_store_keys_agree() -> None:
    """v3 keys both node types' metadata under one store key, distinguished by
    the document's `node_type`. The array and group constants are separately
    typed but must name the same file; adjacency used to make that obvious, and
    they now live in different modules."""
    import zarr_metadata.model as m

    assert m.ZARR_V3_ARRAY_METADATA_STORE_KEY == m.ZARR_V3_GROUP_METADATA_STORE_KEY


def test_validation_diagnostics_exported_from_package() -> None:
    """The validation-diagnostic types and validators are exported from the package."""
    import zarr_metadata.model

    for name in (
        "ValidationProblem",
        "MetadataValidationError",
        "validate_json",
        "validate_metadata_field_v3",
        "validate_array_metadata_v3",
        "validate_array_metadata_v2",
    ):
        assert name in zarr_metadata.model.__all__
        assert hasattr(zarr_metadata.model, name)


def test_expect_expectfail_smoke() -> None:
    """The Expect/ExpectFail test-case dataclasses behave as expected."""
    e = Expect(input=1, output=2, id="x")
    assert (e.input, e.output, e.id) == (1, 2, "x")
    f = ExpectFail(input=1, exception=ValueError, id="y", msg="boom")
    with f.raises():
        raise ValueError("boom")


def test_v3_from_json_error_lists_all_problems() -> None:
    """A malformed v3 document surfaces every problem via MetadataValidationError.problems."""
    doc: dict[str, object] = dict(ZarrV3ArrayMetadata.create_default().to_json())
    del doc["shape"]
    doc["data_type"] = 5
    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV3ArrayMetadata.from_json(doc)
    locs = {p.loc for p in exc_info.value.problems}
    assert ("shape",) in locs
    assert ("data_type",) in locs


# --- JSON type / fill_value contract ---------------------------------------


def test_json_value_type_accepts_json_shapes() -> None:
    # JSONValue is the package's public JSON type alias; assigning JSON-shaped
    # values to it is valid.
    """The JSONValue type alias accepts JSON-shaped values."""
    value: JSONValue = {"a": [1, 2.0, "x", True, None]}
    assert value == {"a": [1, 2.0, "x", True, None]}


def test_string_nan_fill_value_roundtrips() -> None:
    # A float's non-finite fill values are the spec strings ("NaN",
    # "Infinity", "-Infinity"):
    #   https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L63-L79
    # The string form round-trips cleanly under default dataclass equality,
    # unlike a raw float('nan'), which is not JSON.
    """A float array's string 'NaN' fill_value round-trips cleanly."""
    m = ZarrV3ArrayMetadata.create_default(fill_value="NaN", data_type="float32")
    assert ZarrV3ArrayMetadata.from_json(m.to_json()) == m
    assert ZarrV3ArrayMetadata.from_json(m.to_json()).fill_value == "NaN"


# --- V3 baseline -----------------------------------------------------------


def test_v3_to_json_emits_canonical_document() -> None:
    """V3 to_json emits exactly the expected document (which covers every
    spec-required key by construction)."""
    out = ZarrV3ArrayMetadata.create_default(shape=(10,), data_type="int32").to_json()
    assert out == {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (10,),
        "fill_value": 0,
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (10,)}},
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
        "chunk_key_encoding": {"name": "default"},
    }


@pytest.mark.parametrize(
    "data_type",
    ["int32", {"name": "int32"}, {"name": "int32", "configuration": {}}],
    ids=["bare-name", "object", "empty-configuration"],
)
def test_v3_a_data_type_with_nothing_to_configure_is_written_by_its_bare_name(
    data_type: object,
) -> None:
    # As core data types have been written since Zarr v3.0, which is how
    # zarr-python reads them; every other extension point is an object.
    document = {
        **ZarrV3ArrayMetadata.create_default(shape=(10,)).to_json(),
        "data_type": data_type,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    model = ZarrV3ArrayMetadata.from_json(document)
    # The document is written as it was written; the field alone, by its
    # bare name, which every reader takes.
    assert model.to_json()["data_type"] == data_type
    assert model.data_type.to_json() == "int32"


def test_v3_dimension_names_included_when_present() -> None:
    """V3 to_json includes dimension_names when they are set."""
    out: dict[str, object] = dict(
        ZarrV3ArrayMetadata.create_default(shape=(4,), dimension_names=("x",)).to_json()
    )
    assert out["dimension_names"] == ("x",)


def test_v3_dimension_names_omitted_when_none() -> None:
    """V3 to_json omits dimension_names when they are UNSET."""
    model = ZarrV3ArrayMetadata.create_default()
    assert model.dimension_names is UNSET
    assert "dimension_names" not in model.to_json()


# --- BUG 1: attributes gated on dimension_names ----------------------------


def test_v3_attributes_included_when_dimension_names_is_none() -> None:
    """Attributes must be emitted regardless of dimension_names.

    Regression: attributes were gated on ``dimension_names is not None``,
    so non-empty attributes were silently dropped when there were no
    dimension names.
    """
    model = ZarrV3ArrayMetadata.create_default(attributes={"foo": "bar"})
    assert model.dimension_names is UNSET
    out: dict[str, object] = dict(model.to_json())
    assert out["attributes"] == {"foo": "bar"}


# --- BUG 2: single storage transformer dropped -----------------------------


def test_v3_single_storage_transformer_included() -> None:
    """A single storage transformer must be emitted.

    Regression: the guard used ``> 1`` instead of ``> 0``, dropping a
    lone storage transformer.
    """
    st: ZarrV3NamedConfigJSON = {"name": "some_transformer"}
    out: dict[str, object] = dict(
        ZarrV3ArrayMetadata.create_default(storage_transformers=(st,)).to_json()
    )
    assert out["storage_transformers"] == ({"name": "some_transformer"},)


def test_v3_no_storage_transformers_omitted() -> None:
    """V3 to_json omits storage_transformers when the document wrote none, and writes an empty one as written."""
    assert "storage_transformers" not in ZarrV3ArrayMetadata.create_default().to_json()
    written = dict(ZarrV3ArrayMetadata.create_default(storage_transformers=()).to_json())
    assert written["storage_transformers"] == ()


# --- V3 extra fields -------------------------------------------------------


def test_v3_extra_fields_merged() -> None:
    """V3 to_json merges extra_fields into the top-level document."""
    model = ZarrV3ArrayMetadata.create_default(my_ext={"must_understand": False})
    assert model.extra_fields == {"my_ext": {"must_understand": False}}
    assert model.to_json()["my_ext"] == {"must_understand": False}


# --- V3 key/value ----------------------------------------------------------


def test_v3_to_key_value_is_valid_json_under_zarr_json() -> None:
    """V3 to_key_value produces valid JSON bytes under the zarr.json key."""
    kv = ZarrV3ArrayMetadata.create_default(attributes={"a": 1}).to_key_value()
    assert set(kv) == {"zarr.json"}
    parsed = json.loads(kv["zarr.json"].decode("utf-8"))
    assert parsed["zarr_format"] == 3
    assert parsed["attributes"] == {"a": 1}


# --- V3 standard-key sets --------------------------------------------------


def test_standard_keys_is_union_of_required_and_optional() -> None:
    """The standard-key set is the union of the required and optional key sets."""
    assert (
        ARRAY_METADATA_STANDARD_KEYS_V3
        == ARRAY_METADATA_REQUIRED_KEYS_V3 | ARRAY_METADATA_OPTIONAL_KEYS_V3
    )


def test_standard_keys_contains_known_fields_and_excludes_extensions() -> None:
    """The standard-key set contains known fields and excludes extension keys."""
    assert {
        "zarr_format",
        "node_type",
        "shape",
        "codecs",
    } <= ARRAY_METADATA_STANDARD_KEYS_V3
    assert "my_ext" not in ARRAY_METADATA_STANDARD_KEYS_V3


# --- create_default --------------------------------------------------------


def test_v3_create_default_is_valid_empty_array() -> None:
    """V3 create_default builds a structurally valid empty array that round-trips."""
    m = ZarrV3ArrayMetadata.create_default()
    assert m.shape == ()
    assert m.data_type.to_json() == "uint8"
    assert m.fill_value == 0
    assert m.attributes == {}
    assert m.extra_fields == {}
    # the default document is structurally valid and round-trips
    assert validate_array_metadata_v3(m.to_json()) == ()
    assert ZarrV3ArrayMetadata.from_json(m.to_json()) == m


def test_v3_create_default_applies_overrides() -> None:
    """V3 create_default applies keyword overrides over the defaults."""
    m = ZarrV3ArrayMetadata.create_default(shape=(4, 4), attributes={"a": 1})
    assert m.shape == (4, 4)
    assert m.attributes == {"a": 1}
    # un-overridden fields keep their defaults
    assert m.data_type.to_json() == "uint8"


def test_v2_create_default_is_valid_empty_array() -> None:
    """V2 create_default builds a structurally valid empty array that round-trips."""
    m = ZarrV2ArrayMetadata.create_default()
    assert m.shape == ()
    assert m.chunks == ()
    assert m.fill_value == 0
    assert m.compressor is None
    assert m.filters is None
    assert m.attributes is UNSET
    assert validate_array_metadata_v2(m.to_json()) == ()
    assert ZarrV2ArrayMetadata.from_json(m.to_json()) == m


def test_v2_create_default_applies_overrides() -> None:
    """V2 create_default applies keyword overrides over the defaults."""
    m = ZarrV2ArrayMetadata.create_default(shape=(8,), attributes={"k": "v"})
    assert m.shape == (8,)
    assert m.attributes == {"k": "v"}
    assert m.dtype == "|u1"  # default dtype unchanged


# --- V3 update -------------------------------------------------------------

# Cluster 3: update same-shape pairs across versions — parametrized

UPDATE_NEW_INSTANCE_PARAMS = [
    pytest.param(ZarrV3ArrayMetadata, id="v3"),
    pytest.param(ZarrV2ArrayMetadata, id="v2"),
]


@pytest.mark.parametrize("model_cls", UPDATE_NEW_INSTANCE_PARAMS)
def test_update_returns_new_instance(
    model_cls: type[ZarrV3ArrayMetadata | ZarrV2ArrayMetadata],
) -> None:
    """update returns a new instance with the field replaced, leaving the original unchanged."""
    base = model_cls.create_default(shape=(10,))
    updated = base.update(shape=(20,))
    assert updated.shape == (20,)
    assert base.shape == (10,)  # original unchanged
    assert isinstance(updated, model_cls)


UPDATE_NO_ARGS_PARAMS = [
    pytest.param(ZarrV3ArrayMetadata, id="v3"),
    pytest.param(ZarrV2ArrayMetadata, id="v2"),
]


@pytest.mark.parametrize("model_cls", UPDATE_NO_ARGS_PARAMS)
def test_update_no_args_returns_equal_model(
    model_cls: type[ZarrV3ArrayMetadata | ZarrV2ArrayMetadata],
) -> None:
    """update with no arguments returns a model equal to the original."""
    base = model_cls.create_default()
    updated = base.update()
    assert updated == base


# V3-only update tests — kept direct (extra_fields is v3-specific)


def test_update_can_add_an_extension_member() -> None:
    """update can add a member the spec does not define, which the model holds in extra_fields."""
    base = ZarrV3ArrayMetadata.create_default()
    updated = base.update(my_ext={"must_understand": False})
    assert updated.extra_fields == {"my_ext": {"must_understand": False}}


def test_update_reads_every_member_in_the_models_own_scope() -> None:
    """`update` reads the document it makes in the scope the model was read in, so a name that scope leaves unclaimed stays unclaimed; `with_context` is how another scope reads it.

    `zstd` is an extension, which `CORE` leaves unclaimed and unjudged.
    """
    little: ZarrV3NamedConfigJSON = {"name": "bytes", "configuration": {"endian": "little"}}
    zstd: ZarrV3NamedConfigJSON = {"name": "zstd", "configuration": {"level": 3, "checksum": False}}
    base = ZarrV3ArrayMetadata.create_default(context=CORE, codecs=(little, zstd))
    assert isinstance(base.codecs[1], Unclaimed)
    kept = base.update(attributes={"k": 1})
    assert kept.codecs[1] == base.codecs[1]
    assert kept.context == CORE
    given = base.with_context(CORE_AND_EXTENSIONS).update(codecs=(little, zstd))
    assert isinstance(given.codecs[1], Read)


def test_update_leaves_out_a_member_given_as_unset() -> None:
    """Each member a document may leave out: `dimension_names`, `attributes`, one the spec does not define."""
    base = ZarrV3ArrayMetadata.create_default(
        shape=(2,), dimension_names=("x",), attributes={"a": 1}, my_ext={"must_understand": False}
    )
    for updated, member in (
        (base.update(dimension_names=UNSET), "dimension_names"),
        (base.update(attributes=UNSET), "attributes"),
        (base.update(my_ext=UNSET), "my_ext"),
    ):
        written = updated.to_json()
        assert member not in written
        assert {key: value for key, value in base.to_json().items() if key != member} == written


@pytest.mark.parametrize(
    "members",
    [
        {"data_type": {"name": "uint8"}},
        {"codecs": ("bytes",)},
        {"codecs": ({"name": "bytes", "configuration": {}, "must_understand": True},)},
        {"chunk_key_encoding": {"name": "default", "configuration": {}}},
        {"codecs": ({"name": "bytes"}, {"name": "acme.codec", "configuration": {}})},
        {
            "shape": (4,),
            "codecs": (
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": (2,),
                        "codecs": ({"name": "bytes"},),
                        "index_codecs": (
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            "crc32c",
                        ),
                    },
                },
            ),
        },
    ],
    ids=[
        "data-type-object",
        "codec-bare",
        "codec-verbose",
        "empty-configuration",
        "unclaimed",
        "a-field-a-codec-holds-bare",
    ],
)
def test_a_model_is_what_its_document_says_not_how_it_is_spelled(
    members: ZarrV3ArrayMetadataJSONPartial,
) -> None:
    """A model reads back from its own document as itself, and equals the model of any spelling of it."""
    model = ZarrV3ArrayMetadata.create_default(**members)
    assert ZarrV3ArrayMetadata.from_json(model.to_json()) == model
    written = {**model.to_json(), **members}
    assert ZarrV3ArrayMetadata.from_json(written) == model


_DATETIME = {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 1}}
_LITTLE = {"name": "bytes", "configuration": {"endian": "little"}}


@pytest.mark.parametrize(
    ("data_type", "left", "right", "same"),
    [
        ("float32", "NaN", "0x7fc00000", True),
        ("float32", 1, 1.0, True),
        ("float32", 0.1, 0.10000000149011612, True),
        ("float32", 0.0, -0.0, False),
        ("float32", "NaN", "0xffc00000", False),
        (_DATETIME, "NaT", -(2**63), True),
        ("bytes", [65], "QR==", True),
        # The fill value of a data type nothing in scope claims is not
        # interpreted, and is compared as JSON text.
        ("acme.decimal", {"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
        ("acme.decimal", [1, 2], [2, 1], False),
        ("acme.decimal", True, 1, False),
        ("acme.decimal", 0.0, -0.0, False),
    ],
)
def test_two_models_are_one_array_when_their_fill_values_are_one_value(
    data_type: object, left: object, right: object, same: bool
) -> None:
    """As two fields are one when they read the same, however each is spelled."""
    codecs = [{"name": "vlen-bytes"} if data_type == "bytes" else _LITTLE]
    document = {**ZarrV3ArrayMetadata.create_default(shape=(2,)).to_json(), "codecs": codecs}
    models = [
        ZarrV3ArrayMetadata.from_json({**document, "data_type": data_type, "fill_value": value})
        for value in (left, right)
    ]
    assert (models[0] == models[1]) is same
    assert (models[1] == models[0]) is same
    assert models[0] == ZarrV3ArrayMetadata.from_json(models[0].to_json())
    # Equal models hash alike.
    if same:
        assert hash(models[0]) == hash(models[1])
    # A group holding the arrays says so too.
    groups = [
        ZarrV3GroupMetadata.create_default(
            consolidated_metadata={**_INLINE, "metadata": {"a": model.to_json()}}
        )
        for model in models
    ]
    assert (groups[0] == groups[1]) is same


_INLINE: dict[str, Any] = {"kind": "inline", "must_understand": False}
_NOSHUFFLE = {"cname": "lz4", "clevel": 5, "shuffle": "noshuffle", "blocksize": 0}


@pytest.mark.parametrize(
    ("member", "left", "right"),
    [
        (
            "codecs",
            [_LITTLE, {"name": "blosc", "configuration": _NOSHUFFLE}],
            [_LITTLE, {"name": "blosc", "configuration": {**_NOSHUFFLE, "typesize": 4}}],
        ),
        (
            "codecs",
            [_LITTLE, {"name": "zstd", "configuration": {"level": 1}}],
            [_LITTLE, {"name": "zstd", "configuration": {"level": 1, "checksum": False}}],
        ),
        (
            "chunk_key_encoding",
            "default",
            {"name": "default", "configuration": {"separator": "/"}},
        ),
    ],
    ids=["blosc-typesize-noshuffle", "zstd-checksum-false", "default-separator"],
)
def test_two_models_are_one_array_when_their_fields_read_the_same(
    member: str, left: object, right: object
) -> None:
    """The spec's equivalences, which each definition's `canonical` folds, hold in a model's `==`, though `to_json` writes each as given."""
    document = ZarrV3ArrayMetadata.create_default(shape=(2,)).to_json()
    models = [ZarrV3ArrayMetadata.from_json({**document, member: value}) for value in (left, right)]
    assert models[0] == models[1]
    assert hash(models[0]) == hash(models[1])
    assert models[0].to_json() != models[1].to_json()


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        ({"a": math.nan}, {"a": math.nan}, True),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
        ({"a": True}, {"a": 1}, False),
        ({"a": 0.0}, {"a": -0.0}, False),
        ({"a": 1}, {"a": 1.0}, False),
    ],
    ids=["nan", "key-order", "bool-vs-int", "signed-zero", "int-vs-float"],
)
def test_user_json_compares_as_text(
    left: "dict[str, JSONValue]", right: "dict[str, JSONValue]", same: bool
) -> None:
    """Attributes, which nothing interprets, compare as a document writes them: `NaN` is itself, `true` is not `1`, `-0.0` is not `0.0`."""
    arrays = [ZarrV3ArrayMetadata.create_default(attributes=held) for held in (left, right)]
    groups = [ZarrV3GroupMetadata.create_default(attributes=held) for held in (left, right)]
    v2 = [ZarrV2ArrayMetadata.create_default(attributes=held) for held in (left, right)]
    for models in (arrays, groups, v2):
        assert (models[0] == models[1]) is same
        if same:
            assert hash(models[0]) == hash(models[1])


def test_a_model_holding_nan_user_data_equals_its_copies() -> None:
    # What Python's `==` on the value denies: `nan != nan`.
    model = ZarrV3ArrayMetadata.create_default(attributes={"_FillValue": math.nan})
    assert ZarrV3ArrayMetadata.from_key_value(model.to_key_value()) == model
    assert pickle.loads(pickle.dumps(model)) == model
    assert ZarrV3ArrayMetadata.from_json(json.loads(json.dumps(model.to_json()))) == model


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        (0.0, 0.0, True),
        ("NaN", "NaN", True),
        (0.0, -0.0, False),
        (1, 1.0, False),
        ("Infinity", "NaN", False),
    ],
)
def test_two_v2_models_are_one_array_when_their_documents_are_written_alike(
    left: object, right: object, same: bool
) -> None:
    """A v2 model compares by its document as text: a fill value spelled two ways is two models."""
    model = ZarrV2ArrayMetadata.create_default(shape=(2,), chunks=(2,), dtype="<f4")
    models = [dataclasses.replace(model, fill_value=value) for value in (left, right)]
    assert (models[0] == models[1]) is same
    if same:
        assert hash(models[0]) == hash(models[1])


def test_a_model_pickles_and_copies_with_its_definitions() -> None:
    model = ZarrV3ArrayMetadata.create_default(
        codecs=(
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "gzip", "configuration": {"level": 1}},
        ),
        context=CORE,
    )
    for again in (pickle.loads(pickle.dumps(model)), copy.copy(model), copy.deepcopy(model)):
        assert again == model
        assert configuration_of(again.codecs[1], GZIP_CODEC) == {"level": 1}


def test_update_replaces_a_member_rather_than_merging_into_it() -> None:
    """update replaces each member it is given whole, and keeps the others."""
    base = ZarrV3ArrayMetadata.create_default(
        a={"must_understand": False, "x": 1}, b={"must_understand": False}
    )
    updated = base.update(a={"must_understand": False})
    assert updated.extra_fields == {
        "a": {"must_understand": False},
        "b": {"must_understand": False},
    }


# --- V2 model --------------------------------------------------------------


def test_v2_partial_keys_match_settable_model_fields() -> None:
    """The v2 partial TypedDict must list exactly the settable fields."""
    settable = {f.name for f in dataclasses.fields(ZarrV2ArrayMetadata) if f.init}
    assert set(ZarrV2ArrayMetadataPartial.__annotations__) == settable


def test_v2_to_key_value_splits_zarray_and_zattrs() -> None:
    """V2 to_key_value splits the document into .zarray and .zattrs."""
    kv = ZarrV2ArrayMetadata.create_default(attributes={"a": 1}).to_key_value()
    assert set(kv) == {".zarray", ".zattrs"}
    zarray = json.loads(kv[".zarray"].decode("utf-8"))
    zattrs = json.loads(kv[".zattrs"].decode("utf-8"))
    assert zarray["zarr_format"] == 2
    assert zattrs == {"a": 1}


def test_v2_zarray_excludes_attributes() -> None:
    """The on-disk ``.zarray`` document must not contain user attributes.

    In v2, attributes live only in the sibling ``.zattrs`` file. The bundled
    ``ZarrV2ArrayMetadataJSON`` / ``to_json()`` carry attributes for convenience, but
    ``to_key_value()`` must split them out.
    """
    kv = ZarrV2ArrayMetadata.create_default(attributes={"a": 1}).to_key_value()
    zarray = json.loads(kv[".zarray"].decode("utf-8"))
    assert "attributes" not in zarray


def test_v2_to_json_still_includes_attributes() -> None:
    """``to_json()`` is the bundled in-memory form and keeps attributes."""
    out: dict[str, object] = dict(ZarrV2ArrayMetadata.create_default(attributes={"a": 1}).to_json())
    assert out["attributes"] == {"a": 1}


# --- arrays_to_tuples helper ----------------------------------------------

ARRAYS_TO_TUPLES_CASES = [
    Expect([1, 2, 3], (1, 2, 3), id="top-level-list"),
    Expect({"a": [1, [2, 3]], "b": "x"}, {"a": (1, (2, 3)), "b": "x"}, id="nested-in-dict"),
    Expect(5, 5, id="scalar-int"),
    Expect("s", "s", id="scalar-str"),
    Expect(None, None, id="scalar-none"),
    Expect(
        {"name": "bytes", "configuration": {"nums": [1, 2]}},
        {"name": "bytes", "configuration": {"nums": (1, 2)}},
        id="dict-keys-preserved",
    ),
]


@pytest.mark.parametrize("case", ARRAYS_TO_TUPLES_CASES, ids=lambda c: c.id)
def test_arrays_to_tuples(case: Expect[object, object]) -> None:
    """arrays_to_tuples recursively converts JSON arrays to tuples."""
    assert arrays_to_tuples(case.input) == case.output


# --- ZarrV3ArrayMetadata.from_json ----------------------------------------


def test_v3_from_json_reconstructs_required_fields() -> None:
    """V3 from_json reconstructs the required fields from a document."""
    doc = ZarrV3ArrayMetadata.create_default(
        shape=(7,), attributes={"a": 1}, data_type="int32"
    ).to_json()
    model = ZarrV3ArrayMetadata.from_json(doc)
    assert model.shape == (7,)
    assert model.data_type.to_json() == "int32"
    assert model.attributes == {"a": 1}


def test_v3_from_json_defaults_for_omitted_optionals() -> None:
    """V3 from_json supplies defaults for omitted optional fields."""
    doc = ZarrV3ArrayMetadata.create_default(attributes={}, storage_transformers=()).to_json()
    # to_json omits these entirely; from_json must restore defaults
    model = ZarrV3ArrayMetadata.from_json(doc)
    assert model.attributes == {}
    assert model.storage_transformers == ()
    assert model.dimension_names is UNSET


def test_v3_from_json_routes_unknown_keys_to_extra_fields() -> None:
    """V3 from_json routes unknown top-level keys into extra_fields."""
    doc = ZarrV3ArrayMetadata.create_default(my_ext={"must_understand": False}).to_json()
    model = ZarrV3ArrayMetadata.from_json(doc)
    assert model.extra_fields == {"my_ext": {"must_understand": False}}


def test_v3_from_json_standard_keys_not_in_extra_fields() -> None:
    """V3 from_json keeps standard keys out of extra_fields."""
    doc = ZarrV3ArrayMetadata.create_default(
        shape=(10,), attributes={"a": 1}, dimension_names=("x",)
    ).to_json()
    model = ZarrV3ArrayMetadata.from_json(doc)
    assert model.extra_fields == {}


def test_v3_from_json_nested_arrays_in_attributes_become_tuples() -> None:
    """V3 from_json converts nested arrays in attributes into tuples."""
    doc = ZarrV3ArrayMetadata.create_default(attributes={"scale": [[1, 2], [3, 4]]}).to_json()
    model = ZarrV3ArrayMetadata.from_json(doc)
    assert model.attributes == {"scale": ((1, 2), (3, 4))}


# --- ZarrV3ArrayMetadata.from_key_value ----------------------------------


def test_v3_from_key_value_parses_zarr_json() -> None:
    """V3 from_key_value parses the zarr.json entry into a model."""
    kv = ZarrV3ArrayMetadata.create_default(shape=(3,)).to_key_value()
    model = ZarrV3ArrayMetadata.from_key_value(kv)
    assert model.shape == (3,)


# --- Cluster 2: from_key_value missing-key raises (parametrized) -----------

FROM_KEY_VALUE_MISSING_PARAMS = [
    pytest.param(
        ZarrV3ArrayMetadata,
        ExpectFail({}, MetadataValidationError, id="v3-missing-zarr-json", msg="missing store key"),
        id="v3-missing-zarr-json",
    ),
    pytest.param(
        ZarrV2ArrayMetadata,
        ExpectFail({}, MetadataValidationError, id="v2-missing-zarray", msg="missing store key"),
        id="v2-missing-zarray",
    ),
]


@pytest.mark.parametrize(("model_cls", "case"), FROM_KEY_VALUE_MISSING_PARAMS)
def test_from_key_value_missing_key_raises(
    model_cls: type[ZarrV3ArrayMetadata | ZarrV2ArrayMetadata],
    case: ExpectFail[dict[str, bytes]],
) -> None:
    """from_key_value raises MetadataValidationError when the required store key is absent."""
    with case.raises():
        model_cls.from_key_value(case.input)


# --- Cluster 1: round-trips (model → json → model, parametrized) -----------

ROUNDTRIP_MODEL_JSON_PARAMS = [
    pytest.param(
        ZarrV3ArrayMetadata,
        ZarrV3ArrayMetadata.create_default(
            shape=(10,),
            attributes={"a": 1},
            dimension_names=("x",),
            storage_transformers=({"name": "acme.t"},),
            ext={"must_understand": False},
        ),
        id="v3-full",
    ),
    pytest.param(
        ZarrV3ArrayMetadata,
        ZarrV3ArrayMetadata.create_default(attributes={}, storage_transformers=()),
        id="v3-empty-optionals",
    ),
    pytest.param(
        ZarrV2ArrayMetadata,
        ZarrV2ArrayMetadata.create_default(attributes={"a": 1}, filters=None, compressor=None),
        id="v2-basic",
    ),
]


@pytest.mark.parametrize(("model_cls", "model"), ROUNDTRIP_MODEL_JSON_PARAMS)
def test_roundtrip_model_json_model(
    model_cls: type[ZarrV3ArrayMetadata | ZarrV2ArrayMetadata],
    model: ZarrV3ArrayMetadata | ZarrV2ArrayMetadata,
) -> None:
    """A model round-trips through to_json/from_json back to an equal model."""
    assert model_cls.from_json(model.to_json()) == model


# --- Round-trips (model → key_value → model) --------------------------------


def test_roundtrip_via_key_value() -> None:
    """A model round-trips through to_key_value/from_key_value back to an equal model."""
    # Not parametrized over the two classes: a mapping's key type is
    # invariant, so a reader cannot take the union of what they write.
    v3 = ZarrV3ArrayMetadata.create_default(attributes={"a": 1})
    v2 = ZarrV2ArrayMetadata.create_default(attributes={"a": 1})
    assert ZarrV3ArrayMetadata.from_key_value(v3.to_key_value()) == v3
    assert ZarrV2ArrayMetadata.from_key_value(v2.to_key_value()) == v2


# --- Round-trips (json → model → json, direction distinct — kept direct) ---


def test_v3_roundtrip_json_model_json() -> None:
    """A v3 document round-trips through from_json/to_json back to an equal document."""
    doc = ZarrV3ArrayMetadata.create_default(
        shape=(10,), attributes={"a": 1}, dimension_names=("x",)
    ).to_json()
    assert ZarrV3ArrayMetadata.from_json(doc).to_json() == doc


def test_v2_roundtrip_json_model_json() -> None:
    """A v2 document round-trips through from_json/to_json back to an equal document."""
    doc = ZarrV2ArrayMetadata.create_default(attributes={"a": 1}).to_json()
    assert ZarrV2ArrayMetadata.from_json(doc).to_json() == doc


# --- to_json shares no mutable state with the model ------------------------

TO_JSON_NO_ALIASING_PARAMS = [
    pytest.param(
        ZarrV3ArrayMetadata.create_default(
            shape=(2,),
            attributes={"a": {"b": [1]}},
            # A name nothing in the scope claims, so reading it judges only
            # the document, and its configuration can nest.
            codecs=({"name": "acme.nested", "configuration": {"opts": {"level": 1}}},),
            ext={"must_understand": False, "cfg": {"x": [1]}},
        ),
        id="v3",
    ),
    pytest.param(
        ZarrV2ArrayMetadata.create_default(
            attributes={"a": {"b": [1]}},
            # Ids nothing in the scope claims, so their parameters can nest;
            # a complex type, whose fill value is a pair.
            dtype="<c8",
            compressor={"id": "acme.zstd", "opts": {"level": 1}},
            filters=({"id": "acme.delta", "cfg": [1]},),
            fill_value=[0, 0],
        ),
        id="v2",
    ),
]


@pytest.mark.parametrize("model", TO_JSON_NO_ALIASING_PARAMS)
def test_to_json_shares_no_mutable_state_with_model(
    model: ZarrV3ArrayMetadata | ZarrV2ArrayMetadata,
) -> None:
    """Mutating a document returned by to_json leaves the model unchanged."""
    baseline = copy.deepcopy(model.to_json())
    mutate_nested_containers(model.to_json())
    assert model.to_json() == baseline


@pytest.mark.parametrize("model", TO_JSON_NO_ALIASING_PARAMS)
def test_from_json_shares_no_mutable_state_with_its_input(
    model: ZarrV3ArrayMetadata | ZarrV2ArrayMetadata,
) -> None:
    """Mutating the document a model was read from leaves the model unchanged."""
    # Arrays as tuples: the reader has nothing to rebuild, so only a copy
    # keeps the model apart from its input.
    document = arrays_to_tuples(model.to_json())
    read = type(model).from_json(document)
    baseline = copy.deepcopy(read.to_json())
    mutate_nested_containers(document)
    assert read.to_json() == baseline


def test_v3_parser_accepts_bare_string_data_type() -> None:
    """V3 from_json accepts a bare-string data_type and re-serializes it canonically."""
    doc = ZarrV3ArrayMetadata.create_default().to_json()
    doc["data_type"] = "int32"
    doc["codecs"] = ({"name": "bytes", "configuration": {"endian": "little"}},)
    model = ZarrV3ArrayMetadata.from_json(doc)
    assert (model.data_type.json, model.data_type.name) == ("int32", "int32")
    assert model.to_json()["data_type"] == "int32"


@pytest.mark.parametrize("name", ["bytes", "acme.codec", "urn:example:codec"])
def test_metadata_field_accepts_a_name_as_the_spec_names_one(name: str) -> None:
    """The structural layer checks the name is one the spec gives an extension, not that anything registered it."""
    assert validate_metadata_field_v3({"name": name}) == ()


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_metadata_field_must_understand_must_be_boolean(value: object) -> None:
    """must_understand is a JSON boolean, not a truthy scalar."""
    problems = validate_metadata_field_v3({"name": "acme.x", "must_understand": value})
    assert [(problem.loc, problem.kind) for problem in problems] == [
        (("must_understand",), "invalid_type")
    ]


def test_metadata_field_rejects_unknown_envelope_member() -> None:
    """Unknown envelope keys cannot be silently discarded during normalization."""
    problems = validate_metadata_field_v3({"name": "acme.x", "typo": 1})
    assert [(problem.loc, problem.kind) for problem in problems] == [(("typo",), "unknown_key")]


@pytest.mark.parametrize("field", ["codecs", "storage_transformers"])
def test_every_extension_point_rejects_must_understand_false(field: str) -> None:
    """No extension point may be declared ignorable.

    Ignoring a codec gives wrong bytes as surely as ignoring a data type
    gives wrong values, so `must_understand` is a property of the kind of
    metadata rather than a per-occurrence choice. The spec names only the
    three required points; this package reads that as an oversight.
    """
    doc: dict[str, object] = dict(ZarrV3ArrayMetadata.create_default().to_json())
    doc[field] = ({"name": "optional", "must_understand": False},)
    assert [problem.loc for problem in validate_array_metadata_v3(doc)] == [
        (field, 0, "must_understand")
    ]


@pytest.mark.parametrize("field", ["data_type", "chunk_grid", "chunk_key_encoding"])
def test_required_extension_points_reject_must_understand_false(field: str) -> None:
    """Core extension points needed to locate or decode chunks cannot be ignored."""
    doc: dict[str, object] = dict(ZarrV3ArrayMetadata.create_default().to_json())
    doc[field] = {"name": "optional", "must_understand": False}
    assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v3(doc)] == [
        ((field, "must_understand"), "invalid_value")
    ]


def test_v3_codecs_cannot_be_empty() -> None:
    """The core document requires at least one array-to-bytes codec."""
    doc: dict[str, object] = dict(ZarrV3ArrayMetadata.create_default().to_json())
    doc["codecs"] = ()
    assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v3(doc)] == [
        (("codecs",), "invalid_value")
    ]


def test_v2_roundtrip_with_compressor_and_filters() -> None:
    # Non-None compressor/filters must round-trip; extra assertion on .compressor.
    """A v2 model with non-None compressor and filters round-trips."""
    compressor: ZarrV2CodecMetadata = {"id": "blosc", "clevel": 5}
    filters: tuple[ZarrV2CodecMetadata, ...] = ({"id": "delta", "dtype": "<i4"},)
    m = ZarrV2ArrayMetadata.create_default(compressor=compressor, filters=filters)
    restored = ZarrV2ArrayMetadata.from_json(m.to_json())
    assert restored == m
    assert restored.compressor == {"id": "blosc", "clevel": 5}


# --- ZarrV2ArrayMetadata.from_json ----------------------------------------


def test_v2_from_json_reconstructs_fields() -> None:
    """V2 from_json reconstructs the fields from a document."""
    doc = ZarrV2ArrayMetadata.create_default(shape=(4,), attributes={"a": 1}, dtype="<i4").to_json()
    model = ZarrV2ArrayMetadata.from_json(doc)
    assert model.shape == (4,)
    assert model.dtype == "<i4"
    assert model.attributes == {"a": 1}


def test_v2_from_json_attributes_absent_is_unset() -> None:
    """V2 from_json reads an absent attributes key as UNSET, distinct from an
    explicit empty mapping."""
    absent = ZarrV2ArrayMetadata.from_json(ZarrV2ArrayMetadata.create_default().to_json())
    explicit = ZarrV2ArrayMetadata.from_json(
        ZarrV2ArrayMetadata.create_default(attributes={}).to_json()
    )
    assert absent.attributes is UNSET
    assert explicit.attributes == {}
    assert absent != explicit


# --- ZarrV2ArrayMetadata.from_key_value --------------------------------


def test_v2_from_key_value_remerges_zattrs() -> None:
    """V2 from_key_value re-merges .zattrs back into attributes."""
    kv = ZarrV2ArrayMetadata.create_default(attributes={"a": 1}, shape=(10,)).to_key_value()
    model = ZarrV2ArrayMetadata.from_key_value(kv)
    assert model.attributes == {"a": 1}
    assert model.shape == (10,)


def test_v2_from_key_value_rejects_zarray_attributes() -> None:
    """A raw `.zarray` document must not carry `attributes`: they live in `.zattrs`."""
    doc: dict[str, object] = dict(ZarrV2ArrayMetadata.create_default().to_json())
    doc.pop("attributes", None)
    doc["attributes"] = {}

    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV2ArrayMetadata.from_key_value({".zarray": json.dumps(doc).encode()})

    assert [(problem.loc, problem.kind) for problem in exc_info.value.problems] == [
        (("attributes",), "invalid_value")
    ]


def test_v2_from_key_value_ignores_zarray_extra_members() -> None:
    """Other raw `.zarray` members "SHOULD be ignored by implementations" (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L91-L92)."""
    doc: dict[str, object] = dict(ZarrV2ArrayMetadata.create_default().to_json())
    doc.pop("attributes", None)
    doc["vendor_extension"] = {}

    model = ZarrV2ArrayMetadata.from_key_value({".zarray": json.dumps(doc).encode()})

    assert "vendor_extension" not in model.to_json()


def test_v2_zattrs_presence_round_trips() -> None:
    """The .zattrs file's presence is part of the store: an absent file reads
    as UNSET and emits no .zattrs; an explicit empty file reads as {} and
    emits .zattrs — the two stores stay distinct through a round-trip."""
    explicit_kv = dict(ZarrV2ArrayMetadata.create_default(attributes={}).to_key_value())
    assert ".zattrs" in explicit_kv
    absent_kv = dict(explicit_kv)
    del absent_kv[".zattrs"]

    absent = ZarrV2ArrayMetadata.from_key_value(absent_kv)
    explicit = ZarrV2ArrayMetadata.from_key_value(explicit_kv)
    assert absent.attributes is UNSET
    assert explicit.attributes == {}
    assert ".zattrs" not in absent.to_key_value()
    assert ".zattrs" in explicit.to_key_value()


def test_v2_from_json_nested_arrays_in_attributes_become_tuples() -> None:
    """V2 from_json converts nested arrays in attributes into tuples."""
    doc = ZarrV2ArrayMetadata.create_default(attributes={"axes": [[0, 1], [2, 3]]}).to_json()
    model = ZarrV2ArrayMetadata.from_json(doc)
    assert model.attributes == {"axes": ((0, 1), (2, 3))}


# --- scalar wire-type guards (is_/validate_/parse_) ------------------------
#
# Each value is modelled once as Expect[object, frozenset[tuple[str | int, ...]]]
# where `output` is the set of expected problem locs validate_* must report —
# frozenset() means VALID. Valid iff output == frozenset().

JSON_VALIDATE_CASES: list[Expect[object, frozenset[tuple[str | int, ...]]]] = [
    Expect("s", frozenset(), id="str"),
    Expect(1, frozenset(), id="int"),
    Expect(1.5, frozenset(), id="float"),
    Expect(True, frozenset(), id="bool"),
    Expect(None, frozenset(), id="none"),
    Expect({"a": [1, {"b": None}], "c": "x"}, frozenset(), id="nested-containers"),
    Expect((1, 2, 3), frozenset(), id="tuple-array"),
    Expect(float("nan"), frozenset({()}), id="nan"),
    Expect(float("inf"), frozenset({()}), id="inf"),
    Expect(float("-inf"), frozenset({()}), id="negative-inf"),
    Expect(object(), frozenset({()}), id="object"),
    Expect(b"abc", frozenset({()}), id="bytes"),
    Expect(bytearray(b"abc"), frozenset({()}), id="bytearray"),
    Expect({1: "x"}, frozenset({()}), id="non-str-key"),
    Expect([1, object()], frozenset({(1,)}), id="non-json-list-item"),
    Expect({"ok": object()}, frozenset({("ok",)}), id="non-json-value"),
]


@pytest.mark.parametrize("case", JSON_VALIDATE_CASES, ids=lambda c: c.id)
def test_is_json(case: Expect[object, frozenset[tuple[str | int, ...]]]) -> None:
    """is_json reports whether a value is JSON-serializable."""
    assert is_json(case.input) is (case.output == frozenset())


@pytest.mark.parametrize("case", JSON_VALIDATE_CASES, ids=lambda c: c.id)
def test_validate_json(case: Expect[object, frozenset[tuple[str | int, ...]]]) -> None:
    """validate_json reports the problems (and their locs) for a value."""
    problems = validate_json(case.input)
    assert (problems == ()) is (case.output == frozenset())
    assert {p.loc for p in problems} >= case.output


@pytest.mark.parametrize("case", JSON_VALIDATE_CASES, ids=lambda c: c.id)
def test_parse_json(case: Expect[object, frozenset[tuple[str | int, ...]]]) -> None:
    """parse_json returns valid JSON values and raises on invalid ones."""
    if case.output == frozenset():
        parsed = parse_json(case.input)
        assert arrays_to_tuples(parsed) == arrays_to_tuples(case.input)
    else:
        with pytest.raises(MetadataValidationError):
            parse_json(case.input)


def test_parse_json_materializes_abstract_containers() -> None:
    """Accepted Mapping and Sequence values normalize to JSON encoder containers."""
    value = UserDict({"values": range(3)})

    parsed = parse_json(value)

    assert parsed == {"values": (0, 1, 2)}
    assert type(parsed) is dict
    assert type(parsed["values"]) is tuple
    json.dumps(parsed, allow_nan=False)


@pytest.mark.parametrize(
    "guard",
    [
        is_json,
        is_metadata_field_v3,
        is_array_metadata_v3,
        is_array_metadata_v2,
        is_group_metadata_v3,
        is_group_metadata_v2,
    ],
    ids=lambda guard: guard.__name__,
)
def test_a_guard_narrows_only_when_it_says_yes(guard: Callable[[object], bool]) -> None:
    """Each guard is False for some values of its type -- `is_json(math.nan)` is,
    and a NaN is a `float` -- so it is a `TypeGuard`: a `TypeIs` would tell a
    type checker to narrow such a value away when the guard says no."""
    assert get_origin(get_type_hints(guard)["return"]) is TypeGuard


def test_json_type_guard_rejects_abstract_sequence() -> None:
    """A guard cannot narrow an abstract sequence that only the parser materializes."""
    assert not is_json(range(3))
    assert parse_json(range(3)) == (0, 1, 2)


def test_parse_metadata_field_materializes_abstract_containers() -> None:
    """Named-config parsing produces canonical containers at every nesting level."""
    value = UserDict({"name": "example", "configuration": UserDict({"values": range(2)})})

    parsed = parse_metadata_field_v3(value)

    assert isinstance(parsed, dict)
    assert parsed == {"name": "example", "configuration": {"values": (0, 1)}}
    assert "configuration" in parsed
    assert type(parsed["configuration"]) is dict


def test_metadata_field_type_guard_rejects_abstract_mapping() -> None:
    """A metadata-field guard only narrows concrete TypedDict-shaped objects."""
    value = UserDict({"name": "bytes"})

    assert not is_metadata_field_v3(value)
    assert parse_metadata_field_v3(value) == {"name": "bytes"}


def test_validate_json_reports_json_in_message() -> None:
    """validate_json's message for a non-JSON value mentions JSON."""
    problems = validate_json(object())
    assert problems[0].loc == ()
    assert "JSON" in problems[0].message


METADATA_FIELD_VALIDATE_CASES: list[Expect[object, frozenset[tuple[str | int, ...]]]] = [
    Expect("bytes", frozenset(), id="bare-string"),
    Expect({"name": "acme.x", "configuration": {"a": 1}}, frozenset(), id="named-config"),
    Expect({"name": "bytes"}, frozenset(), id="name-only"),
    Expect(5, frozenset({()}), id="not-str-or-mapping"),
    Expect({"configuration": {}}, frozenset({("name",)}), id="missing-name"),
    Expect({"name": 3}, frozenset({("name",)}), id="non-str-name"),
    Expect(
        {"name": "x", "configuration": [1]},
        frozenset({("configuration",)}),
        id="config-not-mapping",
    ),
    Expect(
        {"name": "x", "configuration": {1: "y"}},
        frozenset({("configuration",)}),
        id="config-non-str-key",
    ),
]


@pytest.mark.parametrize("case", METADATA_FIELD_VALIDATE_CASES, ids=lambda c: c.id)
def test_is_metadata_field_v3(case: Expect[object, frozenset[tuple[str | int, ...]]]) -> None:
    """is_metadata_field_v3 reports whether a value is a v3 metadata field."""
    assert is_metadata_field_v3(case.input) is (case.output == frozenset())


@pytest.mark.parametrize("case", METADATA_FIELD_VALIDATE_CASES, ids=lambda c: c.id)
def test_validate_metadata_field_v3(
    case: Expect[object, frozenset[tuple[str | int, ...]]],
) -> None:
    """validate_metadata_field_v3 reports the problems for a metadata-field value."""
    problems = validate_metadata_field_v3(case.input)
    assert (problems == ()) is (case.output == frozenset())
    assert {p.loc for p in problems} >= case.output


@pytest.mark.parametrize("case", METADATA_FIELD_VALIDATE_CASES, ids=lambda c: c.id)
def test_parse_metadata_field_v3(
    case: Expect[object, frozenset[tuple[str | int, ...]]],
) -> None:
    """parse_metadata_field_v3 returns valid fields and raises on invalid ones."""
    if case.output == frozenset():
        assert parse_metadata_field_v3(case.input) is case.input
    else:
        with pytest.raises(MetadataValidationError):
            parse_metadata_field_v3(case.input)


# --- array-document wire-type guards (is_/validate_/parse_) ----------------
#
# Each case starts from a valid document (built by `make`) and applies a
# mutation. `expected_locs` are loc paths `validate_*` must report for the
# invalid cases (a subset check, so accumulation of OTHER problems is allowed).


def _build_v3(**overrides: Unpack[ZarrV3ArrayMetadataJSONPartial]) -> dict[str, object]:
    return dict(ZarrV3ArrayMetadata.create_default(**overrides).to_json())


def _build_v2(**overrides: Unpack[ZarrV2ArrayMetadataPartial]) -> dict[str, object]:
    return dict(ZarrV2ArrayMetadata.create_default(**overrides).to_json())


def _mutate(build: Callable[[], dict], mutate: Callable[[dict], object]) -> Callable[[], dict]:
    def _factory() -> dict:
        doc = build()
        mutate(doc)
        return doc

    return _factory


def _del(key: str) -> Callable[[dict], object]:
    return lambda doc: doc.pop(key)


def _set(key: str, value: object) -> Callable[[dict], object]:
    return lambda doc: doc.__setitem__(key, value)


V3_DOC_CASES: list[Expect[Callable[[], object], frozenset[tuple[str | int, ...]]]] = [
    Expect(_build_v3, frozenset(), id="valid"),
    Expect(
        lambda: _build_v3(shape=(10,), attributes={"a": 1}, dimension_names=("x",)),
        frozenset(),
        id="valid-with-attributes-and-dim-names",
    ),
    Expect(
        lambda: _build_v3(my_ext={"must_understand": False}),
        frozenset(),
        id="valid-with-extra-fields",
    ),
    Expect(_mutate(_build_v3, _del("shape")), frozenset({("shape",)}), id="missing-shape"),
    Expect(
        _mutate(_build_v3, _set("data_type", 5)),
        frozenset({("data_type",)}),
        id="bad-data-type",
    ),
    Expect(
        _mutate(_build_v3, _set("shape", "not-a-shape")),
        frozenset({("shape",)}),
        id="shape-not-sequence",
    ),
    Expect(
        _mutate(_build_v3, _set("shape", [1, "x"])),
        frozenset({("shape",)}),
        id="shape-non-int-item",
    ),
    Expect(
        _mutate(_build_v3, _set("codecs", (5,))),
        frozenset({("codecs", 0)}),
        id="bad-codec-entry",
    ),
    Expect(lambda: [1, 2, 3], frozenset({()}), id="non-mapping-list"),
    Expect(lambda: "nope", frozenset({()}), id="non-mapping-str"),
    Expect(
        _mutate(_mutate(_build_v3, _del("shape")), _set("data_type", 5)),
        frozenset({("shape",), ("data_type",)}),
        id="missing-shape-and-bad-data-type",
    ),
]

V2_DOC_CASES: list[Expect[Callable[[], object], frozenset[tuple[str | int, ...]]]] = [
    Expect(_build_v2, frozenset(), id="valid"),
    Expect(lambda: _build_v2(attributes={"a": 1}), frozenset(), id="valid-with-attributes"),
    Expect(
        lambda: _build_v2(compressor=None, filters=None),
        frozenset(),
        id="valid-none-compressor-filters",
    ),
    Expect(
        _mutate(_build_v2, _del("chunks")),
        frozenset({("chunks",)}),
        id="missing-chunks",
    ),
    Expect(
        _mutate(_build_v2, _set("shape", [1, "x"])),
        frozenset({("shape",)}),
        id="bad-shape",
    ),
    Expect(
        _mutate(_mutate(_build_v2, _del("chunks")), _set("shape", [1, "x"])),
        frozenset({("chunks",), ("shape",)}),
        id="missing-chunks-and-bad-shape",
    ),
]

ALL_DOC_CASES = [
    *(
        pytest.param(
            is_array_metadata_v3,
            validate_array_metadata_v3,
            parse_array_metadata_v3,
            c,
            id=f"v3-{c.id}",
        )
        for c in V3_DOC_CASES
    ),
    *(
        pytest.param(
            is_array_metadata_v2,
            validate_array_metadata_v2,
            parse_array_metadata_v2,
            c,
            id=f"v2-{c.id}",
        )
        for c in V2_DOC_CASES
    ),
]


@pytest.mark.parametrize(("is_fn", "validate_fn", "parse_fn", "case"), ALL_DOC_CASES)
def test_array_metadata_guards(
    is_fn: Callable[[object], bool],
    validate_fn: Callable[[object], list[ValidationProblem]],
    parse_fn: Callable[[object], object],
    case: Expect[Callable[[], object], frozenset[tuple[str | int, ...]]],
) -> None:
    """is_/validate_/parse_ array-metadata guards agree on validity and locs for each case."""
    doc = case.input()
    valid = case.output == frozenset()
    assert is_fn(doc) is valid
    problems = validate_fn(doc)
    assert (problems == ()) is valid
    assert {p.loc for p in problems} >= case.output
    if valid:
        assert parse_fn(doc) is doc
    else:
        with pytest.raises(MetadataValidationError):
            parse_fn(doc)


# --- strict from_json validation -------------------------------------------


FROM_JSON_REJECT_PARAMS = [
    pytest.param(
        ZarrV3ArrayMetadata,
        ExpectFail(lambda: {"zarr_format": 3}, MetadataValidationError, id="x"),
        id="v3-missing-required",
    ),
    pytest.param(
        ZarrV3ArrayMetadata,
        ExpectFail(_mutate(_build_v3, _set("data_type", 5)), MetadataValidationError, id="x"),
        id="v3-bad-field-type",
    ),
    pytest.param(
        ZarrV2ArrayMetadata,
        ExpectFail(lambda: {"zarr_format": 2}, MetadataValidationError, id="x"),
        id="v2-missing-required",
    ),
]


@pytest.mark.parametrize(("model", "case"), FROM_JSON_REJECT_PARAMS)
def test_from_json_rejects_malformed(
    model: type[ZarrV3ArrayMetadata | ZarrV2ArrayMetadata],
    case: ExpectFail[Callable[[], object]],
) -> None:
    """from_json raises MetadataValidationError on a malformed document."""
    with case.raises():
        model.from_json(case.input())


# --- ValidationProblem / MetadataValidationError / _prefix -----------------
# Small structural tests — not "parametrize over inputs" shaped, kept direct.


def test_validation_problem_str_with_loc() -> None:
    """ValidationProblem.__str__ renders a non-empty loc as a dotted path."""
    p = ValidationProblem(loc=("codecs", 0, "name"), message="expected str", kind="invalid_type")
    assert str(p) == "codecs.0.name: expected str"


def test_validation_problem_str_empty_loc() -> None:
    """ValidationProblem.__str__ renders an empty loc as <root>."""
    p = ValidationProblem(loc=(), message="not a mapping", kind="invalid_type")
    assert str(p) == "<root>: not a mapping"


def test_validation_problem_is_frozen() -> None:
    """ValidationProblem is immutable (frozen dataclass)."""
    p = ValidationProblem(loc=("shape",), message="x", kind="invalid_type")
    with pytest.raises(dataclasses.FrozenInstanceError):
        # setattr: assigning to a frozen field is an intentional runtime error,
        # spelled dynamically so it is not also a static type error.
        setattr(p, "message", "y")  # noqa: B010


def test_metadata_validation_error_holds_problems() -> None:
    """MetadataValidationError carries its problem list and renders them in its message."""
    problems = [
        ValidationProblem(loc=("shape",), message="missing required key", kind="missing_key"),
        ValidationProblem(
            loc=("data_type",), message="expected a metadata field", kind="invalid_type"
        ),
    ]
    err = MetadataValidationError(problems)
    assert err.problems == tuple(problems)
    assert "shape: missing required key" in str(err)
    assert "data_type: expected a metadata field" in str(err)


def test_the_error_pickles_and_copies_as_its_problems() -> None:
    error = MetadataValidationError([ValidationProblem(("a",), "bad a", "invalid_value")])
    error.add_note("while reading a")
    for again in (pickle.loads(pickle.dumps(error)), copy.copy(error), copy.deepcopy(error)):
        assert type(again) is MetadataValidationError
        assert again.problems == error.problems
        assert str(again) == str(error)
        assert again.__notes__ == ["while reading a"]


def test_error_a_problem_refuses_a_loc_that_is_not_a_tuple() -> None:
    # The missing comma: `("level")` is a string, and read as a location
    # it would be the path through each of its characters.
    with pytest.raises(TypeError, match="loc is a tuple of keys and indices"):
        ValidationProblem(("level"), "bad level", "invalid_value")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("part", [True, 1.5], ids=["bool", "float"])
def test_error_a_problem_refuses_a_loc_part_that_is_not_a_key_or_an_index(part: object) -> None:
    # `True` passes as an `int`, and would index a sequence as 1.
    with pytest.raises(TypeError, match="loc is a tuple of keys and indices"):
        ValidationProblem(("a", part), "bad a", "invalid_value")  # pyright: ignore[reportArgumentType]


def test_error_a_problem_refuses_a_message_that_is_not_a_string() -> None:
    with pytest.raises(TypeError, match="message is a string"):
        ValidationProblem(("level",), 7, "invalid_value")  # pyright: ignore[reportArgumentType]


def test_error_a_problem_refuses_a_kind_that_is_not_one() -> None:
    # A kind outside the set is one a consumer that dispatches on kinds
    # never sees.
    with pytest.raises(TypeError, match="kind is one of"):
        ValidationProblem(("level",), "bad level", "invalid")  # pyright: ignore[reportArgumentType]


class _EqualToEveryKind:
    """Not a kind, though it compares equal to each."""

    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return 0


def test_error_a_problem_refuses_a_kind_that_only_compares_equal_to_one() -> None:
    # `in` tests equality, which any object can claim.
    with pytest.raises(TypeError, match="kind is one of"):
        ValidationProblem(("level",), "bad level", _EqualToEveryKind())  # pyright: ignore[reportArgumentType]


def test_error_the_error_refuses_what_is_not_a_problem() -> None:
    # A list of one-element tuples of problems is the likely slip: it
    # would otherwise fail far away, where a `loc` is read off an entry.
    with pytest.raises(TypeError, match="takes ValidationProblem values, got tuple"):
        MetadataValidationError([(ValidationProblem(("a",), "bad a", "invalid_value"),)])  # pyright: ignore[reportArgumentType]


def test_prefix_prepends_loc_head() -> None:
    """`prefixed` prepends a loc head to each problem's loc."""
    problems = [ValidationProblem(loc=("name",), message="expected str", kind="invalid_type")]
    located = prefixed(0, problems)
    assert located == (
        ValidationProblem(loc=(0, "name"), message="expected str", kind="invalid_type"),
    )


# --- Stricter v2/v3 field validation and error kinds -------------------------


def test_v2_dtype_must_be_string_or_records() -> None:
    """A non-string, non-records v2 dtype is rejected with an invalid_type problem."""
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"dtype": 42}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("dtype",), "invalid_type")]


@pytest.mark.parametrize(
    ("validate", "document", "loc"),
    [
        (
            validate_array_metadata_v3,
            {**ZarrV3ArrayMetadata.create_default().to_json(), "codecs": b""},
            ("codecs",),
        ),
        (
            validate_array_metadata_v3,
            {**ZarrV3ArrayMetadata.create_default().to_json(), "storage_transformers": bytearray()},
            ("storage_transformers",),
        ),
        (
            validate_array_metadata_v3,
            {**ZarrV3ArrayMetadata.create_default(shape=()).to_json(), "dimension_names": b""},
            ("dimension_names",),
        ),
        (
            validate_array_metadata_v2,
            {**ZarrV2ArrayMetadata.create_default().to_json(), "dtype": b""},
            ("dtype",),
        ),
        (
            validate_array_metadata_v2,
            {**ZarrV2ArrayMetadata.create_default().to_json(), "dtype": (("f0", b""),)},
            ("dtype", 0, 1),
        ),
        (
            validate_array_metadata_v2,
            {**ZarrV2ArrayMetadata.create_default().to_json(), "filters": b""},
            ("filters",),
        ),
    ],
    ids=["codecs", "storage-transformers", "dimension-names", "dtype", "dtype-record", "filters"],
)
def test_error_bytes_are_not_an_array(
    validate: Callable[[object], tuple[ValidationProblem, ...]],
    document: dict[str, object],
    loc: tuple[str | int, ...],
) -> None:
    """`bytes` is a sequence to Python and not an array to JSON: where a document
    expects an array, an empty one no longer passes as one with no items."""
    assert [(p.loc, p.kind) for p in validate(document)] == [(loc, "invalid_type")]


def test_v2_structured_dtype_records_accepted() -> None:
    """A structured v2 dtype (field records, optionally nested/shaped) validates."""
    dtype = (("a", "<i4"), ("b", (("c", "|u1"),)), ("d", "<f8", (2, 2)))
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {
        "dtype": dtype,
        "fill_value": None,
    }
    assert validate_array_metadata_v2(doc) == ()


def test_v2_structured_dtype_malformed_record_rejected() -> None:
    """A field record with the wrong arity is rejected, at the record."""
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"dtype": (("a",),)}
    problems = validate_array_metadata_v2(doc)
    assert [p.loc for p in problems] == [("dtype", "fields", 0)]


def test_v2_order_literal_enforced() -> None:
    """An order other than 'C' or 'F' is rejected with an invalid_value problem."""
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"order": "Q"}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("order",), "invalid_value")]


def test_v2_compressor_must_be_codec_or_none() -> None:
    """A compressor that is not null or a codec config mapping is rejected."""
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"compressor": "zlib"}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("compressor",), "invalid_type")]


def test_v2_compressor_requires_string_id() -> None:
    """A compressor mapping without a string id is rejected, at the id."""
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"compressor": {"level": 3}}
    problems = validate_array_metadata_v2(doc)
    assert [p.loc for p in problems] == [("compressor", "id")]


def test_v2_filters_must_be_codec_sequence_or_none() -> None:
    """Filters that are not null or a sequence of codec configs are rejected: an item that is no codec at the item, anything else at the field."""
    for bad, at in ((7, ("filters",)), ((5,), ("filters", 0)), ("gzip", ("filters",))):
        doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"filters": bad}
        problems = validate_array_metadata_v2(doc)
        assert [(p.loc, p.kind) for p in problems] == [(at, "invalid_type")], bad


def test_v2_shape_and_chunks_must_have_equal_rank() -> None:
    """Raw v2 metadata requires one chunk length per array dimension."""
    doc = dict(ZarrV2ArrayMetadata.create_default(shape=(2, 3)).to_json())
    doc["chunks"] = (1,)

    assert [(p.loc, p.kind) for p in validate_array_metadata_v2(doc)] == [
        (("chunks",), "invalid_value")
    ]
    with pytest.raises(MetadataValidationError, match="same number of dimensions"):
        ZarrV2ArrayMetadata.from_key_value({".zarray": json.dumps(doc).encode()})


def test_v2_filters_may_be_empty() -> None:
    """An empty filter list is a list: the spec says "a list ... or null", with no minimum.

    https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L76-L79
    """
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json())
    doc["filters"] = ()

    assert validate_array_metadata_v2(doc) == ()
    assert ZarrV2ArrayMetadata.from_key_value({".zarray": json.dumps(doc).encode()}).filters == ()


def test_v2_dimension_separator_literal_enforced() -> None:
    """A dimension_separator other than '.' or '/' is rejected."""
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"dimension_separator": "-"}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("dimension_separator",), "invalid_value")]


def test_v2_zarr_format_literal_enforced() -> None:
    """A v2 document claiming zarr_format 3 is rejected with an invalid_value problem."""
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"zarr_format": 3}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("zarr_format",), "invalid_value")]


def test_v3_zarr_format_literal_enforced() -> None:
    """A v3 document claiming zarr_format 2 is rejected with an invalid_value problem."""
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {"zarr_format": 2}
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("zarr_format",), "invalid_value")]


@pytest.mark.parametrize(
    ("document", "validate"),
    [
        pytest.param(
            dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"zarr_format": 2.0},
            validate_array_metadata_v2,
            id="v2",
        ),
        pytest.param(
            dict(ZarrV3ArrayMetadata.create_default().to_json()) | {"zarr_format": 3.0},
            validate_array_metadata_v3,
            id="v3",
        ),
    ],
)
def test_array_zarr_format_rejects_float(
    document: object, validate: Callable[[object], list[ValidationProblem]]
) -> None:
    """Integer-valued floats do not satisfy integer format literals."""
    assert [(p.loc, p.kind) for p in validate(document)] == [(("zarr_format",), "invalid_value")]


def test_array_v2_ignores_unknown_document_member() -> None:
    """Other .zarray keys "SHOULD NOT be present ... and SHOULD be ignored": tolerated, dropped.

    https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L91-L92
    """
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"unexpected": 1}

    assert validate_array_metadata_v2(doc) == ()
    assert "unexpected" not in ZarrV2ArrayMetadata.from_json(doc).to_json()


@pytest.mark.parametrize(
    ("member", "kind"),
    [
        (object(), "invalid_type"),
        ({1, 2}, "invalid_type"),
        (b"\x00", "invalid_type"),
        (math.nan, "invalid_value"),
    ],
    ids=["object", "set", "bytes", "nan"],
)
def test_error_array_v2_unknown_member_that_is_not_json(member: object, kind: str) -> None:
    """Ignored is not unchecked: an unknown member is a JSON value, as in v3."""
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"unexpected": member}

    assert [(p.loc, p.kind) for p in validate_array_metadata_v2(doc)] == [(("unexpected",), kind)]


@pytest.mark.parametrize("key", [7, None, True, (1, 2)], ids=["int", "none", "bool", "tuple"])
def test_error_array_v2_key_that_is_not_a_string(key: object) -> None:
    """A document's keys are strings: `parse_array_metadata_v2` returned this one."""
    doc: dict[object, object] = {**ZarrV2ArrayMetadata.create_default().to_json(), key: "x"}

    assert [(p.loc, p.kind) for p in validate_array_metadata_v2(doc)] == [((), "invalid_type")]


def test_array_v3_from_json_materializes_abstract_containers() -> None:
    """A flexible input mapping becomes the canonical dict/tuple model shape."""
    # `range(2)` is the shape (0, 1), which the default grid for it fits.
    doc = UserDict(dict(ZarrV3ArrayMetadata.create_default(shape=(0, 1)).to_json()))
    doc["shape"] = range(2)

    model = ZarrV3ArrayMetadata.from_json(doc)

    assert model.shape == (0, 1)
    assert type(model.shape) is tuple


def test_from_key_value_rejects_non_standard_json_constant() -> None:
    """A JavaScript NaN constant outside the attributes is located, not decoded as a fill value."""
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json())
    doc["fill_value"] = float("nan")
    raw = json.dumps(doc)

    with pytest.raises(MetadataValidationError, match="fill_value: non-finite float nan"):
        ZarrV3ArrayMetadata.from_key_value({"zarr.json": raw.encode()})


@pytest.mark.parametrize(
    ("members", "problems"),
    [
        ({"fill_value": math.nan}, [(("fill_value",), "invalid_value")]),
        # The default fill value, 0, is not a boolean: a data type goes with
        # a fill value of it.
        ({"data_type": "bool"}, [(("fill_value",), "invalid_type")]),
        # Values of several bytes take an `endian`.
        (
            {"data_type": "int16", "codecs": ({"name": "bytes"},)},
            [(("codecs", 0, "configuration", "endian"), "missing_key")],
        ),
        ({"shape": (2,), "dimension_names": ("x", "y")}, [(("dimension_names",), "invalid_value")]),
        # The shape is not derived from a grid, which the scalar default
        # shape does not fit: a grid goes with the shape it fits.
        (
            {"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (10, 10)}}},
            [(("chunk_grid", "configuration", "chunk_shape"), "invalid_value")],
        ),
    ],
    ids=[
        "non-finite-fill-value",
        "fill-value-of-another-type",
        "no-endian",
        "names-past-shape",
        "grid-of-another-rank",
    ],
)
def test_error_create_default_refuses_a_document_with_a_problem(
    members: ZarrV3ArrayMetadataJSONPartial, problems: list[tuple[tuple[str | int, ...], str]]
) -> None:
    """A model comes from a read, so a document its read refuses makes none, and nothing is written."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ArrayMetadata.create_default(**members)
    assert [(p.loc, p.kind) for p in raised.value.problems] == problems


@pytest.mark.parametrize(
    ("members", "problems"),
    [
        ({"fill_value": 300}, [(("fill_value",), "invalid_value")]),
        # A shape goes with a grid that fits it.
        ({"shape": (4, 4)}, [(("chunk_grid", "configuration", "chunk_shape"), "invalid_value")]),
    ],
    ids=["fill-value-out-of-range", "shape-without-its-grid"],
)
def test_error_update_refuses_a_document_with_a_problem(
    members: ZarrV3ArrayMetadataJSONPartial, problems: list[tuple[tuple[str | int, ...], str]]
) -> None:
    base = ZarrV3ArrayMetadata.create_default(shape=(4,))
    with pytest.raises(MetadataValidationError) as raised:
        base.update(**members)
    assert [(p.loc, p.kind) for p in raised.value.problems] == problems


def test_error_update_refuses_to_leave_out_a_member_a_document_holds() -> None:
    base = ZarrV3ArrayMetadata.create_default(shape=(4,))
    with pytest.raises(MetadataValidationError) as raised:
        base.update(shape=UNSET)  # pyright: ignore[reportArgumentType]
    assert [(p.loc, p.kind) for p in raised.value.problems] == [(("shape",), "missing_key")]


@pytest.mark.parametrize(
    ("shape", "kind"),
    [
        (5, "invalid_type"),
        (None, "invalid_type"),
        (("a",), "invalid_type"),
        ((-1,), "invalid_value"),
    ],
    ids=["a-number", "null", "not-integers", "negative"],
)
def test_error_create_default_reports_a_shape_it_cannot_read(shape: object, kind: str) -> None:
    """As the read reports it, rather than failing to derive a grid from it."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ArrayMetadata.create_default(shape=shape)  # pyright: ignore[reportArgumentType]
    assert [(p.loc, p.kind) for p in raised.value.problems] == [(("shape",), kind)]


def test_a_codec_that_is_not_json_is_placed_by_the_definition_that_claims_its_name() -> None:
    """Its kind is that definition's, as a codec refused for a configuration that is JSON takes it: `gzip` is a bytes -> bytes codec, so the pipeline still lacks an array -> bytes one."""
    document = {
        **ZarrV3ArrayMetadata.create_default().to_json(),
        "codecs": [{"name": "gzip", "configuration": {"level": math.nan}}],
    }
    assert [(p.loc, p.kind) for p in validate_array_metadata_v3(document)] == [
        (("codecs", 0, "configuration", "level"), "invalid_value"),
        (("codecs",), "invalid_value"),
    ]


def test_a_shard_nested_as_deep_as_a_reader_walks_is_read_and_written() -> None:
    """Every walker of fields takes more than one frame per shard, so the deepest nesting the cap admits is where the interpreter's limit would show; one shard deeper is the depth problem."""
    little = {"name": "bytes", "configuration": {"endian": "little"}}

    def nested(shards: int) -> list[object]:
        codecs: list[object] = [little]
        for _ in range(shards):
            codecs = [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": [1],
                        "codecs": codecs,
                        "index_codecs": [little],
                    },
                }
            ]
        return codecs

    # Each shard is three levels -- its object, its configuration and the
    # `codecs` in it -- and the innermost codec's configuration is the
    # last container a reader walks.
    deepest = (JSON_DEPTH - 4) // 3
    codecs = nested(deepest)
    document = {**ZarrV3ArrayMetadata.create_default(shape=(2,)).to_json(), "codecs": codecs}
    assert validate_array_metadata_v3(document) == ()
    model = ZarrV3ArrayMetadata.from_json(document)
    assert json_text(model.to_json()) == json_text(cast("JSONValue", document))
    assert ZarrV3ArrayMetadata.from_key_value(model.to_key_value()) == model
    assert hash(model) == hash(ZarrV3ArrayMetadata.from_json(document))
    assert pickle.loads(pickle.dumps(model)) == model
    assert copy.deepcopy(model) == model
    (shard,) = model.codecs
    assert json_text(canonical_of(shard, ())) == json_text(cast("JSONValue", codecs[0]))
    # A shard and its index codec at each level, and the innermost codec.
    assert len(list(with_problems(fields_of(shard), ()))) == 2 * deepest + 1
    problems = validate_array_metadata_v3({**document, "codecs": nested(deepest + 1)})
    assert {(problem.kind, len(problem.loc), problem.message) for problem in problems} == {
        ("invalid_value", JSON_DEPTH, f"nested deeper than the {JSON_DEPTH} levels a reader walks")
    }


def test_a_model_is_written_as_deep_as_it_is_read() -> None:
    """A fill value as deep as a reader walks, of a data type nothing in scope claims, which takes any JSON; pickled and deep-copied too, which take two frames a level."""
    fill_value: dict[str, object] = {}
    for _ in range(JSON_DEPTH - 2):
        fill_value = {"x": fill_value}
    document = {
        **ZarrV3ArrayMetadata.create_default().to_json(),
        "data_type": "acme.deep",
        "fill_value": fill_value,
    }
    model = ZarrV3ArrayMetadata.from_json(document)
    assert model.to_json()["fill_value"] == fill_value
    assert ZarrV3ArrayMetadata.from_key_value(model.to_key_value()) == model
    assert pickle.loads(pickle.dumps(model)) == model
    assert copy.deepcopy(model) == model


def test_a_field_s_configuration_member_is_counted_from_the_field_s_root() -> None:
    # `validate_metadata_field_v3` judges a field alone, at its own root: a
    # member of its configuration sits two levels down, and the cap counts
    # from the root, not from the member.
    nested: dict[str, object] = {}
    for _ in range(JSON_DEPTH - 2):
        nested = {"x": nested}
    problems = validate_metadata_field_v3({"name": "acme.x", "configuration": {"y": nested}})
    assert [(len(p.loc), p.kind) for p in problems] == [(JSON_DEPTH, "invalid_value")]
    assert problems[0].loc[:2] == ("configuration", "y")
    shallower = {"name": "acme.x", "configuration": {"y": nested["x"]}}
    assert validate_metadata_field_v3(shallower) == ()


def test_a_v2_dtype_of_nested_records_is_read_to_the_levels_a_reader_walks() -> None:
    """Field records nest a dtype two levels a record: the shape check recursed a record at a time with no cap, and a thousand records overflowed."""

    def records(levels: int) -> object:
        dtype: object = "<i4"
        for _ in range(levels):
            dtype = [["f", dtype]]
        return dtype

    # `dtype` sits one level down, each record's list one below the last
    # record, and the record itself one below that.
    deepest = (JSON_DEPTH - 2) // 2
    document = {
        **ZarrV2ArrayMetadata.create_default(shape=(2,)).to_json(),
        "dtype": records(deepest),
        "fill_value": None,
    }
    assert validate_array_metadata_v2(document) == ()
    model = ZarrV2ArrayMetadata.from_json(document)
    assert json_text(model.to_json()) == json_text(cast("JSONValue", document))
    for levels in (deepest + 1, 1000):
        problems = validate_array_metadata_v2({**document, "dtype": records(levels)})
        assert [(problem.kind, len(problem.loc)) for problem in problems] == [
            ("invalid_value", JSON_DEPTH)
        ]


def test_a_v2_document_nested_as_deep_as_a_reader_walks_is_read_and_written() -> None:
    """The v2 models copied with `copy.deepcopy`, two frames a level, and overflowed on documents their validators accept."""
    fill_value: dict[str, object] = {}
    for _ in range(JSON_DEPTH - 2):
        fill_value = {"x": fill_value}
    # An object type, whose fill value is any JSON.
    document = {
        **ZarrV2ArrayMetadata.create_default(shape=(2,), dtype="|O").to_json(),
        "fill_value": fill_value,
    }
    assert validate_array_metadata_v2(document) == ()
    model = ZarrV2ArrayMetadata.from_json(document)
    assert json_text(model.to_json()) == json_text(cast("JSONValue", document))
    assert ZarrV2ArrayMetadata.from_key_value(model.to_key_value()) == model
    assert hash(model) == hash(ZarrV2ArrayMetadata.from_json(document))
    assert pickle.loads(pickle.dumps(model)) == model
    assert copy.deepcopy(model) == model
    problems = validate_array_metadata_v2({**document, "fill_value": {"x": fill_value}})
    assert [(problem.kind, len(problem.loc)) for problem in problems] == [
        ("invalid_value", JSON_DEPTH)
    ]


def test_v3_node_type_literal_enforced() -> None:
    """A v3 array document claiming node_type 'group' is rejected."""
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {"node_type": "group"}
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("node_type",), "invalid_value")]


def test_missing_key_kind_is_machine_readable() -> None:
    """A missing required key is distinguishable by kind, without message matching."""
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json())
    del doc["chunk_key_encoding"]
    problems = validate_array_metadata_v3(doc)
    assert problems == (
        ValidationProblem(("chunk_key_encoding",), "missing required key", "missing_key"),
    )


# --- Unified error channels ---------------------------------------------------


def test_from_key_value_invalid_json_raises_metadata_error() -> None:
    """Undecodable store bytes raise MetadataValidationError (kind invalid_json), not JSONDecodeError."""
    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV3ArrayMetadata.from_key_value({"zarr.json": b"{not json"})
    assert [p.kind for p in exc_info.value.problems] == ["invalid_json"]


def test_from_key_value_invalid_utf8_raises_metadata_error() -> None:
    """Invalid UTF-8 store bytes use the same invalid_json error channel."""
    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV3ArrayMetadata.from_key_value({"zarr.json": b"\x80"})
    assert [p.kind for p in exc_info.value.problems] == ["invalid_json"]


def test_v2_from_key_value_scalar_root_raises_metadata_error() -> None:
    """A scalar .zarray document fails through the unified metadata error channel."""
    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV2ArrayMetadata.from_key_value({".zarray": b"null"})
    assert [(problem.loc, problem.kind) for problem in exc_info.value.problems] == [
        ((), "invalid_type")
    ]


def test_from_key_value_missing_key_kind() -> None:
    """A missing store key surfaces as a missing_key problem at the store-key loc."""
    with pytest.raises(MetadataValidationError) as exc_info:
        ZarrV2ArrayMetadata.from_key_value({})
    assert exc_info.value.problems == (
        ValidationProblem((".zarray",), "missing store key", "missing_key"),
    )


# --- Adversarial-probe fixes: documents that used to pass validation ---------


def test_shape_rejects_json_booleans() -> None:
    """JSON booleans are not integers: shape/chunks containing true/false are
    rejected (bool is an int subclass in Python, so isinstance alone passes)."""
    v3 = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {"shape": (True, True)}
    assert [p.loc for p in validate_array_metadata_v3(v3)] == [("shape",)]
    v2 = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"chunks": (True,)}
    assert [p.loc for p in validate_array_metadata_v2(v2)] == [("chunks",)]


def test_shape_rejects_negative_dimensions() -> None:
    """Dimension lengths must be non-negative; a negative entry is invalid_value."""
    v3 = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {"shape": (-1,)}
    assert [(p.loc, p.kind) for p in validate_array_metadata_v3(v3)] == [
        (("shape",), "invalid_value")
    ]
    v2 = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"chunks": (-5,)}
    assert [(p.loc, p.kind) for p in validate_array_metadata_v2(v2)] == [
        (("chunks",), "invalid_value")
    ]


def test_dimension_names_length_must_match_shape() -> None:
    """dimension_names must have one entry per dimension of shape."""
    doc = dict(ZarrV3ArrayMetadata.create_default(shape=(10,)).to_json()) | {
        "dimension_names": ("x", "y", "z")
    }
    assert [(p.loc, p.kind) for p in validate_array_metadata_v3(doc)] == [
        (("dimension_names",), "invalid_value")
    ]


def test_attributes_values_must_be_json() -> None:
    """Attribute values are JSON-checked recursively (like fill_value), so a
    non-serializable value is a validation problem, not a later TypeError."""
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {"attributes": {"a": {1, 2}}}
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("attributes", "a"), "invalid_type")]


def test_configuration_values_must_be_json() -> None:
    """Configuration values are JSON-checked recursively, so an int-keyed dict
    cannot pass validation and be silently rewritten by json.dumps."""
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": {1: 2}}}
    }
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [
        (("chunk_grid", "configuration", "chunk_shape"), "invalid_type")
    ]


def test_v3_extension_keys_must_be_strings() -> None:
    """A non-string top-level key cannot be represented by a v3 document type."""
    doc: dict[object, object] = {**ZarrV3ArrayMetadata.create_default().to_json()}
    doc[1] = {"must_understand": False}
    assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v3(doc)] == [
        ((), "invalid_type")
    ]


def test_v3_extension_values_must_be_json() -> None:
    """Extension payloads are JSON-checked before a model is constructed."""
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json())
    doc["ext"] = {"must_understand": False, "payload": object()}
    assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v3(doc)] == [
        (("ext", "payload"), "invalid_type")
    ]


def test_v3_json_extension_without_waiver_is_preserved_as_must_understand() -> None:
    """A JSON extension without an explicit false waiver remains must-understand."""
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json())
    doc["ext"] = 1
    parsed = parse_array_metadata_v3(doc)
    assert is_array_metadata_v3(parsed)
    model = ZarrV3ArrayMetadata.from_json(doc)
    assert model.extra_fields["ext"] == 1
    assert model.must_understand_fields == {"ext": 1}


def test_v2_codec_configuration_values_must_be_json() -> None:
    """Non-JSON codec parameters are rejected for compressors and filters."""
    for field, value, expected_loc in (
        ("compressor", {"id": "x", "payload": object()}, ("compressor", "payload")),
        ("filters", ({"id": "x", "payload": object()},), ("filters", 0, "payload")),
    ):
        doc = dict(ZarrV2ArrayMetadata.create_default().to_json())
        doc[field] = value
        assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v2(doc)] == [
            (expected_loc, "invalid_type")
        ]


def test_dimension_sequences_reject_binary_values() -> None:
    """Binary buffers are not JSON arrays even though they are integer sequences."""
    v3 = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {"shape": b"\x02"}
    v2 = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"chunks": b"\x02"}
    assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v3(v3)] == [
        (("shape",), "invalid_type")
    ]
    assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v2(v2)] == [
        (("chunks",), "invalid_type")
    ]


def test_array_parsers_normalize_json_lists_before_narrowing() -> None:
    """Parsers return tuple-backed document types while guards reject raw list forms."""
    v3_raw = json.loads(json.dumps(ZarrV3ArrayMetadata.create_default(shape=(2,)).to_json()))
    v2_raw = json.loads(json.dumps(ZarrV2ArrayMetadata.create_default(shape=(2,)).to_json()))

    assert validate_array_metadata_v3(v3_raw) == ()
    assert validate_array_metadata_v2(v2_raw) == ()
    assert not is_array_metadata_v3(v3_raw)
    assert not is_array_metadata_v2(v2_raw)

    v3_parsed = parse_array_metadata_v3(v3_raw)
    v2_parsed = parse_array_metadata_v2(v2_raw)
    assert isinstance(v3_parsed["shape"], tuple)
    assert isinstance(v3_parsed["codecs"], tuple)
    assert isinstance(v2_parsed["shape"], tuple)
    assert isinstance(v2_parsed["chunks"], tuple)


def test_array_guards_reject_noncanonical_nested_json() -> None:
    """Document guards cannot narrow values that only parsers can materialize."""
    # Raw bits of 16, whose fill value is two byte values.
    v3 = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {"data_type": "r16"}
    v3["fill_value"] = range(2)
    # A complex type, whose fill value is a pair.
    v2 = dict(ZarrV2ArrayMetadata.create_default(dtype="<c8", fill_value=[0, 0]).to_json())
    v2["fill_value"] = range(2)

    assert not is_array_metadata_v3(v3)
    assert not is_array_metadata_v2(v2)
    assert parse_array_metadata_v3(v3)["fill_value"] == (0, 1)
    assert parse_array_metadata_v2(v2)["fill_value"] == (0, 1)


# --- must_understand partition (spec: MUST fail to open unrecognized fields) --


def test_must_understand_fields_partition() -> None:
    """must_understand_fields contains every extra field not explicitly waived
    with must_understand: false, including implicitly-true and non-mapping
    fields, so a reader can discharge the spec's fail-to-open duty by
    subtracting the extensions it recognizes.

    https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1575-L1578
    """
    model = ZarrV3ArrayMetadata.create_default(
        ext_a={"name": "a", "must_understand": False},
        ext_b={"name": "b"},
        ext_c={"name": "c", "must_understand": True},
        ext_d=123,
    )
    assert set(model.must_understand_fields) == {"ext_b", "ext_c", "ext_d"}
    recognized = {"ext_b"}
    assert model.must_understand_fields.keys() - recognized == {"ext_c", "ext_d"}


def test_must_understand_fields_empty_when_all_waived() -> None:
    """must_understand_fields is empty when every extra field is explicitly waived."""
    model = ZarrV3ArrayMetadata.create_default(ext_a={"name": "a", "must_understand": False})
    assert model.must_understand_fields == {}


def test_dimension_names_null_field_rejected() -> None:
    """A dimension_names field whose VALUE is null is invalid: the spec permits
    null as an element (an unnamed dimension), never as the field value — "not
    specified" is spelled by omitting the key. Consumers bridging from an
    in-memory None sentinel must drop the key, not write null.

    https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L635-L638
    """
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {"dimension_names": None}
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("dimension_names",), "invalid_type")]
    # and the model's own UNSET spelling correctly maps to key absence
    model = ZarrV3ArrayMetadata.create_default()
    assert model.dimension_names is UNSET
    assert "dimension_names" not in model.to_json()


# --- create_default derives the chunk grid from shape ------------------------


def test_v3_create_default_chunk_grid_follows_shape() -> None:
    """Overriding shape without chunk_grid derives a consistent default grid:
    one chunk covering the array (chunk_shape == shape), instead of silently
    keeping the scalar default's 0-d grid."""
    model = ZarrV3ArrayMetadata.create_default(shape=(100, 100))
    assert model.chunk_grid.to_json() == {
        "name": "regular",
        "configuration": {"chunk_shape": (100, 100)},
    }


def test_v3_create_default_explicit_chunk_grid_respected() -> None:
    """An explicit chunk_grid override wins over the shape-derived default."""
    grid: ZarrV3NamedConfigJSON = {"name": "regular", "configuration": {"chunk_shape": (10, 10)}}
    model = ZarrV3ArrayMetadata.create_default(shape=(100, 100), chunk_grid=grid)
    assert model.chunk_grid.to_json() == grid


def test_v2_create_default_chunks_follow_shape() -> None:
    """Overriding shape without chunks derives chunks == shape."""
    model = ZarrV2ArrayMetadata.create_default(shape=(100, 100))
    assert model.chunks == (100, 100)


def test_v2_create_default_explicit_chunks_respected() -> None:
    """An explicit chunks override wins over the shape-derived default."""
    model = ZarrV2ArrayMetadata.create_default(shape=(100, 100), chunks=(10, 10))
    assert model.chunks == (10, 10)


def test_v3_create_default_zero_length_dimensions() -> None:
    """The derived grid gives a dimension of length 0 a chunk length of 1:
    the regular grid asks for chunk sizes greater than zero, so the written
    grid is one every reader takes, and it fits the shape it chunks."""
    model = ZarrV3ArrayMetadata.create_default(shape=(0, 3))
    assert model.chunk_grid.to_json() == {
        "name": "regular",
        "configuration": {"chunk_shape": (1, 3)},
    }
    assert validate_array_metadata_v3(model.to_json()) == ()


# --- v2 dimension_separator default (roborev job 426) -------------------------


def test_v2_absent_dimension_separator_means_dot() -> None:
    """A .zarray that omits dimension_separator uses the v2 convention default
    '.', not '/': chunk keys of real-world default-separator v2 arrays look
    like '0.0'. The model normalizes the absent key to an explicit '.' — a
    semantics-preserving spelling normalization."""
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json())
    del doc["dimension_separator"]
    model = ZarrV2ArrayMetadata.from_json(doc)
    assert model.dimension_separator == "."
    assert model.to_json().get("dimension_separator") == "."


def test_v2_from_key_value_without_separator_means_dot() -> None:
    """The .zarray store-file path applies the same '.' default for an absent
    dimension_separator key."""
    doc = {
        k: v
        for k, v in ZarrV2ArrayMetadata.create_default().to_json().items()
        if k not in ("dimension_separator", "attributes")
    }
    import json as _json

    model = ZarrV2ArrayMetadata.from_key_value({".zarray": _json.dumps(doc).encode()})
    assert model.dimension_separator == "."


def test_v2_null_dimension_separator_rejected() -> None:
    """dimension_separator may be absent, '.', or '/' — never null: the
    document grammar has no null spelling for this field."""
    doc = dict(ZarrV2ArrayMetadata.create_default().to_json()) | {"dimension_separator": None}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("dimension_separator",), "invalid_type")]


def test_dimension_names_absent_and_all_null_are_distinct() -> None:
    """An absent dimension_names field and an explicit all-null one are
    semantically different documents: the explicit form says every dimension
    has a name, which is null; absence says there are no dimension names.
    The model preserves the distinction (UNSET vs a tuple of Nones), and both
    spellings round-trip faithfully."""
    absent_doc = dict(ZarrV3ArrayMetadata.create_default(shape=(2, 3)).to_json())
    explicit_doc = absent_doc | {"dimension_names": (None, None)}

    absent = ZarrV3ArrayMetadata.from_json(absent_doc)
    explicit = ZarrV3ArrayMetadata.from_json(explicit_doc)

    assert absent.dimension_names is UNSET
    assert explicit.dimension_names == (None, None)
    assert absent != explicit
    assert "dimension_names" not in absent.to_json()
    assert absent.to_json() == absent_doc
    assert explicit.to_json() == explicit_doc


@pytest.mark.parametrize(
    ("validate", "document", "member", "value", "kind"),
    [
        (validate_array_metadata_v3, ZarrV3ArrayMetadata, "zarr_format", "3", "invalid_type"),
        (validate_array_metadata_v3, ZarrV3ArrayMetadata, "zarr_format", 2, "invalid_value"),
        (validate_array_metadata_v3, ZarrV3ArrayMetadata, "node_type", 5, "invalid_type"),
        (validate_array_metadata_v3, ZarrV3ArrayMetadata, "node_type", "group", "invalid_value"),
        (validate_array_metadata_v2, ZarrV2ArrayMetadata, "order", 1, "invalid_type"),
        (validate_array_metadata_v2, ZarrV2ArrayMetadata, "order", "Q", "invalid_value"),
        (
            validate_array_metadata_v2,
            ZarrV2ArrayMetadata,
            "dimension_separator",
            ":",
            "invalid_value",
        ),
    ],
)
def test_error_a_member_outside_the_values_it_takes(
    validate: Callable[[object], tuple[ValidationProblem, ...]],
    document: type[ZarrV3ArrayMetadata | ZarrV2ArrayMetadata],
    member: str,
    value: object,
    kind: str,
) -> None:
    # Of the wrong type when none of the values it takes is of its JSON
    # type -- a string where a number belongs -- else of the wrong value.
    written = {**document.create_default().to_json(), member: value}
    assert [(p.loc, p.kind) for p in validate(written)] == [((member,), kind)]


def test_error_a_document_nested_deeper_than_a_reader_walks_is_a_problem() -> None:
    # Not a `RecursionError`: a 10 KB document a stranger wrote, nested past
    # the interpreter's limit, is refused at the level past the last a
    # reader walks, by every validator, guard and reader.
    deep: dict[str, object] = {}
    for _ in range(2_000):
        deep = {"a": deep}
    document = {**ZarrV3ArrayMetadata.create_default().to_json(), "attributes": deep}
    problems = validate_array_metadata_v3(document)
    assert [(len(p.loc), p.kind) for p in problems] == [(JSON_DEPTH, "invalid_value")]
    assert problems[0].loc[:2] == ("attributes", "a")
    assert not is_json(document)
    assert not is_array_metadata_v3(document)
    assert not is_group_metadata_v3(document)
    assert not is_array_metadata_v2(document)
    assert not is_group_metadata_v2(document)
    with pytest.raises(MetadataValidationError):
        ZarrV3ArrayMetadata.from_json(document)


def test_error_a_field_object_in_a_document_is_not_json() -> None:
    # A `Read` built by hand, with a configuration its definition refuses,
    # smuggled into a document: refused as what it is, so nothing built by
    # hand passes as read. A model holds its own fields as read.
    smuggled = Read(
        json="gzip", name="gzip", definition=GZIP_CODEC, configuration={"level": 99, "window": 1}
    )
    document = {
        **ZarrV3ArrayMetadata.create_default(shape=(2,)).to_json(),
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, smuggled],
    }
    assert [(p.loc, p.kind) for p in validate_array_metadata_v3(document)] == [
        (("codecs", 1), "invalid_type")
    ]
    assert not is_array_metadata_v3(document)
    with pytest.raises(MetadataValidationError):
        ZarrV3ArrayMetadata.from_json(document)
    # Nor does `update` take one among the members it is given: a model's
    # own fields are no exception, since `update` reads JSON.
    model = ZarrV3ArrayMetadata.create_default(shape=(2,))
    with pytest.raises(MetadataValidationError) as raised:
        model.update(codecs=cast("Any", (model.codecs[0].to_json(), smuggled)))
    assert [(p.loc, p.kind) for p in raised.value.problems] == [(("codecs", 1), "invalid_type")]
    with pytest.raises(MetadataValidationError):
        model.update(codecs=(cast("Any", model.codecs[0]),))


def test_a_problem_shows_an_integer_too_long_to_write_by_its_size(
    interpreter_writes_4300_digits: None,
) -> None:
    # `int` refuses to write more than 4,300 digits; a message says the
    # size instead of raising.
    document = {**ZarrV3ArrayMetadata.create_default().to_json(), "zarr_format": 10**5000}
    (problem,) = validate_array_metadata_v3(document)
    assert problem.message == f"expected 3, got an integer of {(10**5000).bit_length()} bits"
    (problem,) = validate_metadata_field_v3({"name": "gzip", 10**5000: 1})
    assert problem.message == (
        f"non-string metadata field key an integer of {(10**5000).bit_length()} bits"
    )
    # Held by a container, JSON or not, or by a key, it is what the
    # interpreter will not write.
    for held in ([10**5000], [10**5000, object()]):
        document = {**ZarrV3ArrayMetadata.create_default().to_json(), "zarr_format": held}
        (problem,) = validate_array_metadata_v3(document)
        assert (
            problem.message == "expected 3, got a value of type list the interpreter will not write"
        )
    (problem,) = validate_metadata_field_v3({"name": "gzip", (10**5000,): 1})
    assert problem.message == (
        "non-string metadata field key a value of type tuple the interpreter will not write"
    )
    # So is what `refine_json` reports as no JSON at all: a set holding one.
    document = {**ZarrV3ArrayMetadata.create_default().to_json(), "attributes": {"a": {10**5000}}}
    (problem,) = validate_array_metadata_v3(document)
    assert (problem.loc, problem.message) == (
        ("attributes", "a"),
        "not a JSON-serializable value: a value of type set the interpreter will not write",
    )


def test_a_problem_shows_a_value_nested_too_deep_to_write_by_saying_so() -> None:
    # `_refine` stops at `JSON_DEPTH`, and a value nested past it is said to
    # be too deep, not shown by a repr, whose own limit the interpreter and
    # platform set: one level past, which any repr would write, is enough.
    deep: list[object] = []
    innermost = deep
    for _ in range(JSON_DEPTH + 1):
        nested: list[object] = []
        innermost.append(nested)
        innermost = nested
    document = {**ZarrV3ArrayMetadata.create_default().to_json(), "zarr_format": deep}
    # One problem: a value the literal check refuses is not walked, so the
    # depth rule does not judge it too.
    assert [
        (problem.loc, problem.kind, problem.message)
        for problem in validate_array_metadata_v3(document)
    ] == [(("zarr_format",), "invalid_type", "expected 3, got a value nested too deep to show")]
