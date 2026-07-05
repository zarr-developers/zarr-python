"""Tests for the metadata models in ``zarr_metadata.model``."""

import dataclasses
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from typing_extensions import Unpack

from tests.model._cases import Expect, ExpectFail
from zarr_metadata.model import (
    ARRAY_METADATA_OPTIONAL_KEYS_V3,
    ARRAY_METADATA_REQUIRED_KEYS_V3,
    ARRAY_METADATA_STANDARD_KEYS_V3,
    UNSET,
    ArrayMetadataModelV2,
    ArrayMetadataModelV2Partial,
    ArrayMetadataModelV3,
    ArrayMetadataModelV3Partial,
    MetadataFieldModelV3,
    MetadataValidationError,
    NamedConfigModelV3,
    ValidationProblem,
    is_array_metadata_v2,
    is_array_metadata_v3,
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
from zarr_metadata.model._validation import _prefix, arrays_to_tuples

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue
    from zarr_metadata.v2 import CodecMetadataV2

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
    doc: dict[str, object] = dict(ArrayMetadataModelV3.create_default().to_json())
    del doc["shape"]
    doc["data_type"] = 5
    with pytest.raises(MetadataValidationError) as exc_info:
        ArrayMetadataModelV3.from_json(doc)
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
    # Non-finite floats are represented as the spec strings ("NaN", "Infinity",
    # "-Infinity") by the caller — the metadata layer does not interpret dtypes.
    # The string form round-trips cleanly under default dataclass equality,
    # unlike a raw float('nan') (which is an invalid fill_value the caller must
    # not pass).
    """A string 'NaN' fill_value round-trips cleanly (non-finite floats are the caller's responsibility)."""
    m = ArrayMetadataModelV3.create_default(fill_value="NaN")
    assert ArrayMetadataModelV3.from_json(m.to_json()) == m
    assert ArrayMetadataModelV3.from_json(m.to_json()).fill_value == "NaN"


# --- NamedConfigModelV3.to_json ------------------------------------------------

ZARR_TO_JSON_CASES = [
    Expect(
        NamedConfigModelV3(name="regular", configuration={"chunk_shape": [1]}),
        {"name": "regular", "configuration": {"chunk_shape": [1]}},
        id="with-configuration",
    ),
    Expect(
        NamedConfigModelV3(name="bytes", configuration={}),
        {"name": "bytes", "configuration": {}},
        id="without-configuration",
    ),
]


@pytest.mark.parametrize("case", ZARR_TO_JSON_CASES, ids=lambda c: c.id)
def test_zarr_metadata_v3_to_json(case: Expect[NamedConfigModelV3, dict[str, object]]) -> None:
    """NamedConfigModelV3.to_json emits the canonical object form."""
    assert case.input.to_json() == case.output


# --- NamedConfigModelV3.from_json -----------------------------------------------

ZARR_FROM_JSON_CASES = [
    Expect("bytes", NamedConfigModelV3(name="bytes", configuration={}), id="bare-string"),
    Expect(
        {"name": "regular", "configuration": {"chunk_shape": [1]}},
        NamedConfigModelV3(name="regular", configuration={"chunk_shape": (1,)}),
        id="object-with-config",
    ),
    Expect(
        {"name": "bytes"},
        NamedConfigModelV3(name="bytes", configuration={}),
        id="object-without-config",
    ),
]


@pytest.mark.parametrize("case", ZARR_FROM_JSON_CASES, ids=lambda c: c.id)
def test_zarr_metadata_v3_from_json(case: Expect[object, NamedConfigModelV3]) -> None:
    """NamedConfigModelV3.from_json parses both the bare-string and object forms."""
    assert NamedConfigModelV3.from_json(case.input) == case.output


# --- V3 baseline -----------------------------------------------------------


def test_v3_to_json_emits_canonical_document() -> None:
    """V3 to_json emits exactly the expected document (which covers every
    spec-required key by construction)."""
    out = ArrayMetadataModelV3.create_default(
        shape=(10,), data_type=NamedConfigModelV3(name="int32", configuration={})
    ).to_json()
    assert out == {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (10,),
        "fill_value": 0,
        "data_type": {"name": "int32", "configuration": {}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (10,)}},
        "codecs": ({"name": "bytes", "configuration": {}},),
        "chunk_key_encoding": {"name": "default", "configuration": {}},
    }


def test_v3_dimension_names_included_when_present() -> None:
    """V3 to_json includes dimension_names when they are set."""
    out: dict[str, object] = dict(
        ArrayMetadataModelV3.create_default(dimension_names=("x",)).to_json()
    )
    assert out["dimension_names"] == ("x",)


def test_v3_dimension_names_omitted_when_none() -> None:
    """V3 to_json omits dimension_names when they are UNSET."""
    out = ArrayMetadataModelV3.create_default(dimension_names=UNSET).to_json()
    assert "dimension_names" not in out


# --- BUG 1: attributes gated on dimension_names ----------------------------


def test_v3_attributes_included_when_dimension_names_is_none() -> None:
    """Attributes must be emitted regardless of dimension_names.

    Regression: attributes were gated on ``dimension_names is not None``,
    so non-empty attributes were silently dropped when there were no
    dimension names.
    """
    out: dict[str, object] = dict(
        ArrayMetadataModelV3.create_default(
            dimension_names=UNSET, attributes={"foo": "bar"}
        ).to_json()
    )
    assert out["attributes"] == {"foo": "bar"}


# --- BUG 2: single storage transformer dropped -----------------------------


def test_v3_single_storage_transformer_included() -> None:
    """A single storage transformer must be emitted.

    Regression: the guard used ``> 1`` instead of ``> 0``, dropping a
    lone storage transformer.
    """
    st = NamedConfigModelV3(name="some_transformer", configuration={})
    out: dict[str, object] = dict(
        ArrayMetadataModelV3.create_default(storage_transformers=(st,)).to_json()
    )
    assert out["storage_transformers"] == ({"name": "some_transformer", "configuration": {}},)


def test_v3_no_storage_transformers_omitted() -> None:
    """V3 to_json omits storage_transformers when empty."""
    out = ArrayMetadataModelV3.create_default(storage_transformers=()).to_json()
    assert "storage_transformers" not in out


# --- V3 extra fields -------------------------------------------------------


def test_v3_extra_fields_merged() -> None:
    """V3 to_json merges extra_fields into the top-level document."""
    out = ArrayMetadataModelV3.create_default(
        extra_fields={"my_ext": {"must_understand": False}}
    ).to_json()
    assert out["my_ext"] == {"must_understand": False}


def test_v3_extra_fields_overlapping_standard_field_rejected() -> None:
    """Constructing a V3 model with an extra field that collides with a standard key is rejected."""
    with pytest.raises(ValueError):
        ArrayMetadataModelV3.create_default(extra_fields={"shape": {"must_understand": False}})


# --- V3 key/value ----------------------------------------------------------


def test_v3_to_key_value_is_valid_json_under_zarr_json() -> None:
    """V3 to_key_value produces valid JSON bytes under the zarr.json key."""
    kv = ArrayMetadataModelV3.create_default(attributes={"a": 1}).to_key_value()
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
    m = ArrayMetadataModelV3.create_default()
    assert m.shape == ()
    assert m.data_type == NamedConfigModelV3(name="uint8", configuration={})
    assert m.fill_value == 0
    assert m.attributes == {}
    assert m.extra_fields == {}
    # the default document is structurally valid and round-trips
    assert validate_array_metadata_v3(m.to_json()) == []
    assert ArrayMetadataModelV3.from_json(m.to_json()) == m


def test_v3_create_default_applies_overrides() -> None:
    """V3 create_default applies keyword overrides over the defaults."""
    m = ArrayMetadataModelV3.create_default(shape=(4, 4), attributes={"a": 1})
    assert m.shape == (4, 4)
    assert m.attributes == {"a": 1}
    # un-overridden fields keep their defaults
    assert m.data_type == NamedConfigModelV3(name="uint8", configuration={})


def test_v2_create_default_is_valid_empty_array() -> None:
    """V2 create_default builds a structurally valid empty array that round-trips."""
    m = ArrayMetadataModelV2.create_default()
    assert m.shape == ()
    assert m.chunks == ()
    assert m.fill_value == 0
    assert m.compressor is None
    assert m.filters is None
    assert m.attributes is UNSET
    assert validate_array_metadata_v2(m.to_json()) == []
    assert ArrayMetadataModelV2.from_json(m.to_json()) == m


def test_v2_create_default_applies_overrides() -> None:
    """V2 create_default applies keyword overrides over the defaults."""
    m = ArrayMetadataModelV2.create_default(shape=(8,), attributes={"k": "v"})
    assert m.shape == (8,)
    assert m.attributes == {"k": "v"}
    assert m.dtype == "|u1"  # default dtype unchanged


# --- V3 update -------------------------------------------------------------

# Cluster 3: update same-shape pairs across versions — parametrized

UPDATE_NEW_INSTANCE_PARAMS = [
    pytest.param(ArrayMetadataModelV3, id="v3"),
    pytest.param(ArrayMetadataModelV2, id="v2"),
]


@pytest.mark.parametrize("model_cls", UPDATE_NEW_INSTANCE_PARAMS)
def test_update_returns_new_instance(
    model_cls: type[ArrayMetadataModelV3 | ArrayMetadataModelV2],
) -> None:
    """update returns a new instance with the field replaced, leaving the original unchanged."""
    base = model_cls.create_default(shape=(10,))
    updated = base.update(shape=(20,))
    assert updated.shape == (20,)
    assert base.shape == (10,)  # original unchanged
    assert isinstance(updated, model_cls)


UPDATE_NO_ARGS_PARAMS = [
    pytest.param(ArrayMetadataModelV3, id="v3"),
    pytest.param(ArrayMetadataModelV2, id="v2"),
]


@pytest.mark.parametrize("model_cls", UPDATE_NO_ARGS_PARAMS)
def test_update_no_args_returns_equal_model(
    model_cls: type[ArrayMetadataModelV3 | ArrayMetadataModelV2],
) -> None:
    """update with no arguments returns a model equal to the original."""
    base = model_cls.create_default()
    assert base.update() == base


# V3-only update tests — kept direct (extra_fields is v3-specific)


def test_update_can_replace_extra_fields() -> None:
    """update can replace the extra_fields mapping."""
    base = ArrayMetadataModelV3.create_default(extra_fields={})
    updated = base.update(extra_fields={"my_ext": {"must_understand": False}})
    assert updated.extra_fields == {"my_ext": {"must_understand": False}}


def test_update_replaces_extra_fields_rather_than_merging() -> None:
    """update replaces extra_fields wholesale rather than merging."""
    base = ArrayMetadataModelV3.create_default(extra_fields={"a": {"must_understand": False}})
    updated = base.update(extra_fields={"b": {"must_understand": True}})
    assert updated.extra_fields == {"b": {"must_understand": True}}


def test_partial_keys_match_settable_model_fields() -> None:
    """The partial TypedDict must list exactly the constructor-settable fields.

    Guards against drift: adding/removing a settable field on the model
    without updating ``ArrayMetadataModelV3Partial`` fails here.
    """
    settable = {f.name for f in dataclasses.fields(ArrayMetadataModelV3) if f.init}
    assert set(ArrayMetadataModelV3Partial.__annotations__) == settable


# --- V2 model --------------------------------------------------------------


def test_v2_partial_keys_match_settable_model_fields() -> None:
    """The v2 partial TypedDict must list exactly the settable fields."""
    settable = {f.name for f in dataclasses.fields(ArrayMetadataModelV2) if f.init}
    assert set(ArrayMetadataModelV2Partial.__annotations__) == settable


def test_v2_to_key_value_splits_zarray_and_zattrs() -> None:
    """V2 to_key_value splits the document into .zarray and .zattrs."""
    kv = ArrayMetadataModelV2.create_default(attributes={"a": 1}).to_key_value()
    assert set(kv) == {".zarray", ".zattrs"}
    zarray = json.loads(kv[".zarray"].decode("utf-8"))
    zattrs = json.loads(kv[".zattrs"].decode("utf-8"))
    assert zarray["zarr_format"] == 2
    assert zattrs == {"a": 1}


def test_v2_zarray_excludes_attributes() -> None:
    """The on-disk ``.zarray`` document must not contain user attributes.

    In v2, attributes live only in the sibling ``.zattrs`` file. The bundled
    ``ArrayMetadataV2`` / ``to_json()`` carry attributes for convenience, but
    ``to_key_value()`` must split them out.
    """
    kv = ArrayMetadataModelV2.create_default(attributes={"a": 1}).to_key_value()
    zarray = json.loads(kv[".zarray"].decode("utf-8"))
    assert "attributes" not in zarray


def test_v2_to_json_still_includes_attributes() -> None:
    """``to_json()`` is the bundled in-memory form and keeps attributes."""
    out: dict[str, object] = dict(
        ArrayMetadataModelV2.create_default(attributes={"a": 1}).to_json()
    )
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


# --- ArrayMetadataModelV3.from_json ----------------------------------------


def test_v3_from_json_reconstructs_required_fields() -> None:
    """V3 from_json reconstructs the required fields from a document."""
    doc = ArrayMetadataModelV3.create_default(
        shape=(7,),
        attributes={"a": 1},
        data_type=NamedConfigModelV3(name="int32", configuration={}),
    ).to_json()
    model = ArrayMetadataModelV3.from_json(doc)
    assert model.shape == (7,)
    assert model.data_type == NamedConfigModelV3(name="int32", configuration={})
    assert model.attributes == {"a": 1}


def test_v3_from_json_defaults_for_omitted_optionals() -> None:
    """V3 from_json supplies defaults for omitted optional fields."""
    doc = ArrayMetadataModelV3.create_default(
        attributes={}, storage_transformers=(), dimension_names=UNSET
    ).to_json()
    # to_json omits these entirely; from_json must restore defaults
    model = ArrayMetadataModelV3.from_json(doc)
    assert model.attributes == {}
    assert model.storage_transformers == ()
    assert model.dimension_names is UNSET


def test_v3_from_json_routes_unknown_keys_to_extra_fields() -> None:
    """V3 from_json routes unknown top-level keys into extra_fields."""
    doc = ArrayMetadataModelV3.create_default(
        extra_fields={"my_ext": {"must_understand": False}}
    ).to_json()
    model = ArrayMetadataModelV3.from_json(doc)
    assert model.extra_fields == {"my_ext": {"must_understand": False}}


def test_v3_from_json_standard_keys_not_in_extra_fields() -> None:
    """V3 from_json keeps standard keys out of extra_fields."""
    doc = ArrayMetadataModelV3.create_default(
        shape=(10,), attributes={"a": 1}, dimension_names=("x",)
    ).to_json()
    model = ArrayMetadataModelV3.from_json(doc)
    assert model.extra_fields == {}


def test_v3_from_json_nested_arrays_in_attributes_become_tuples() -> None:
    """V3 from_json converts nested arrays in attributes into tuples."""
    doc = ArrayMetadataModelV3.create_default(attributes={"scale": [[1, 2], [3, 4]]}).to_json()
    model = ArrayMetadataModelV3.from_json(doc)
    assert model.attributes == {"scale": ((1, 2), (3, 4))}


# --- ArrayMetadataModelV3.from_key_value ----------------------------------


def test_v3_from_key_value_parses_zarr_json() -> None:
    """V3 from_key_value parses the zarr.json entry into a model."""
    kv = ArrayMetadataModelV3.create_default(shape=(3,)).to_key_value()
    model = ArrayMetadataModelV3.from_key_value(kv)
    assert model.shape == (3,)


# --- Cluster 2: from_key_value missing-key raises (parametrized) -----------

FROM_KEY_VALUE_MISSING_PARAMS = [
    pytest.param(
        ArrayMetadataModelV3,
        ExpectFail({}, MetadataValidationError, id="v3-missing-zarr-json", msg="missing store key"),
        id="v3-missing-zarr-json",
    ),
    pytest.param(
        ArrayMetadataModelV2,
        ExpectFail({}, MetadataValidationError, id="v2-missing-zarray", msg="missing store key"),
        id="v2-missing-zarray",
    ),
]


@pytest.mark.parametrize(("model_cls", "case"), FROM_KEY_VALUE_MISSING_PARAMS)
def test_from_key_value_missing_key_raises(
    model_cls: type[ArrayMetadataModelV3 | ArrayMetadataModelV2],
    case: ExpectFail[dict[str, bytes]],
) -> None:
    """from_key_value raises MetadataValidationError when the required store key is absent."""
    with case.raises():
        model_cls.from_key_value(case.input)


# --- Cluster 1: round-trips (model → json → model, parametrized) -----------

ROUNDTRIP_MODEL_JSON_PARAMS = [
    pytest.param(
        ArrayMetadataModelV3,
        ArrayMetadataModelV3.create_default(
            shape=(10,),
            attributes={"a": 1},
            dimension_names=("x",),
            storage_transformers=(NamedConfigModelV3(name="t", configuration={}),),
            extra_fields={"ext": {"must_understand": False}},
        ),
        id="v3-full",
    ),
    pytest.param(
        ArrayMetadataModelV3,
        ArrayMetadataModelV3.create_default(
            attributes={},
            dimension_names=UNSET,
            storage_transformers=(),
            extra_fields={},
        ),
        id="v3-empty-optionals",
    ),
    pytest.param(
        ArrayMetadataModelV2,
        ArrayMetadataModelV2.create_default(attributes={"a": 1}, filters=None, compressor=None),
        id="v2-basic",
    ),
]


@pytest.mark.parametrize(("model_cls", "model"), ROUNDTRIP_MODEL_JSON_PARAMS)
def test_roundtrip_model_json_model(
    model_cls: type[ArrayMetadataModelV3 | ArrayMetadataModelV2],
    model: ArrayMetadataModelV3 | ArrayMetadataModelV2,
) -> None:
    """A model round-trips through to_json/from_json back to an equal model."""
    assert model_cls.from_json(model.to_json()) == model


# --- Round-trips (model → key_value → model, parametrized) -----------------

ROUNDTRIP_KEY_VALUE_PARAMS = [
    pytest.param(
        ArrayMetadataModelV3,
        ArrayMetadataModelV3.create_default(attributes={"a": 1}),
        id="v3",
    ),
    pytest.param(
        ArrayMetadataModelV2,
        ArrayMetadataModelV2.create_default(attributes={"a": 1}),
        id="v2",
    ),
]


@pytest.mark.parametrize(("model_cls", "model"), ROUNDTRIP_KEY_VALUE_PARAMS)
def test_roundtrip_via_key_value(
    model_cls: type[ArrayMetadataModelV3 | ArrayMetadataModelV2],
    model: ArrayMetadataModelV3 | ArrayMetadataModelV2,
) -> None:
    """A model round-trips through to_key_value/from_key_value back to an equal model."""
    assert model_cls.from_key_value(model.to_key_value()) == model


# --- Round-trips (json → model → json, direction distinct — kept direct) ---


def test_v3_roundtrip_json_model_json() -> None:
    """A v3 document round-trips through from_json/to_json back to an equal document."""
    doc = ArrayMetadataModelV3.create_default(
        shape=(10,), attributes={"a": 1}, dimension_names=("x",)
    ).to_json()
    assert ArrayMetadataModelV3.from_json(doc).to_json() == doc


def test_v2_roundtrip_json_model_json() -> None:
    """A v2 document round-trips through from_json/to_json back to an equal document."""
    doc = ArrayMetadataModelV2.create_default(attributes={"a": 1}).to_json()
    assert ArrayMetadataModelV2.from_json(doc).to_json() == doc


def test_v3_parser_accepts_bare_string_data_type() -> None:
    """V3 from_json accepts a bare-string data_type and re-serializes it canonically."""
    doc = ArrayMetadataModelV3.create_default().to_json()
    doc["data_type"] = "int32"  # bare-string form, not canonical object form
    model = ArrayMetadataModelV3.from_json(doc)
    # parses correctly, re-serializes to canonical object form
    assert model.data_type == NamedConfigModelV3(name="int32", configuration={})
    assert model.to_json()["data_type"] == {"name": "int32", "configuration": {}}


def test_v2_roundtrip_with_compressor_and_filters() -> None:
    # Non-None compressor/filters must round-trip; extra assertion on .compressor.
    """A v2 model with non-None compressor and filters round-trips."""
    compressor: CodecMetadataV2 = {"id": "blosc", "clevel": 5}
    filters: tuple[CodecMetadataV2, ...] = ({"id": "delta"},)
    m = ArrayMetadataModelV2.create_default(compressor=compressor, filters=filters)
    restored = ArrayMetadataModelV2.from_json(m.to_json())
    assert restored == m
    assert restored.compressor == {"id": "blosc", "clevel": 5}


# --- ArrayMetadataModelV2.from_json ----------------------------------------


def test_v2_from_json_reconstructs_fields() -> None:
    """V2 from_json reconstructs the fields from a document."""
    doc = ArrayMetadataModelV2.create_default(
        shape=(4,), attributes={"a": 1}, dtype="<i4"
    ).to_json()
    model = ArrayMetadataModelV2.from_json(doc)
    assert model.shape == (4,)
    assert model.dtype == "<i4"
    assert model.attributes == {"a": 1}


def test_v2_from_json_attributes_absent_is_unset() -> None:
    """V2 from_json reads an absent attributes key as UNSET, distinct from an
    explicit empty mapping."""
    absent = ArrayMetadataModelV2.from_json(ArrayMetadataModelV2.create_default().to_json())
    explicit = ArrayMetadataModelV2.from_json(
        ArrayMetadataModelV2.create_default(attributes={}).to_json()
    )
    assert absent.attributes is UNSET
    assert explicit.attributes == {}
    assert absent != explicit


# --- ArrayMetadataModelV2.from_key_value --------------------------------


def test_v2_from_key_value_remerges_zattrs() -> None:
    """V2 from_key_value re-merges .zattrs back into attributes."""
    kv = ArrayMetadataModelV2.create_default(attributes={"a": 1}, shape=(10,)).to_key_value()
    model = ArrayMetadataModelV2.from_key_value(kv)
    assert model.attributes == {"a": 1}
    assert model.shape == (10,)


def test_v2_zattrs_presence_round_trips() -> None:
    """The .zattrs file's presence is part of the store: an absent file reads
    as UNSET and emits no .zattrs; an explicit empty file reads as {} and
    emits .zattrs — the two stores stay distinct through a round-trip."""
    explicit_kv = dict(ArrayMetadataModelV2.create_default(attributes={}).to_key_value())
    assert ".zattrs" in explicit_kv
    absent_kv = dict(explicit_kv)
    del absent_kv[".zattrs"]

    absent = ArrayMetadataModelV2.from_key_value(absent_kv)
    explicit = ArrayMetadataModelV2.from_key_value(explicit_kv)
    assert absent.attributes is UNSET
    assert explicit.attributes == {}
    assert ".zattrs" not in absent.to_key_value()
    assert ".zattrs" in explicit.to_key_value()


def test_v2_from_json_nested_arrays_in_attributes_become_tuples() -> None:
    """V2 from_json converts nested arrays in attributes into tuples."""
    doc = ArrayMetadataModelV2.create_default(attributes={"axes": [[0, 1], [2, 3]]}).to_json()
    model = ArrayMetadataModelV2.from_json(doc)
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
    Expect(float("nan"), frozenset(), id="nan"),
    Expect(float("inf"), frozenset(), id="inf"),
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
    assert (problems == []) is (case.output == frozenset())
    assert {p.loc for p in problems} >= case.output


@pytest.mark.parametrize("case", JSON_VALIDATE_CASES, ids=lambda c: c.id)
def test_parse_json(case: Expect[object, frozenset[tuple[str | int, ...]]]) -> None:
    """parse_json returns valid JSON values and raises on invalid ones."""
    if case.output == frozenset():
        assert parse_json(case.input) is case.input
    else:
        with pytest.raises(MetadataValidationError):
            parse_json(case.input)


def test_validate_json_reports_json_in_message() -> None:
    """validate_json's message for a non-JSON value mentions JSON."""
    problems = validate_json(object())
    assert problems[0].loc == ()
    assert "JSON" in problems[0].message


METADATA_FIELD_VALIDATE_CASES: list[Expect[object, frozenset[tuple[str | int, ...]]]] = [
    Expect("bytes", frozenset(), id="bare-string"),
    Expect({"name": "x", "configuration": {"a": 1}}, frozenset(), id="named-config"),
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
    assert (problems == []) is (case.output == frozenset())
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


def _build_v3(**overrides: Unpack[ArrayMetadataModelV3Partial]) -> dict[str, object]:
    return dict(ArrayMetadataModelV3.create_default(**overrides).to_json())


def _build_v2(**overrides: Unpack[ArrayMetadataModelV2Partial]) -> dict[str, object]:
    return dict(ArrayMetadataModelV2.create_default(**overrides).to_json())


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
        lambda: _build_v3(extra_fields={"my_ext": {"must_understand": False}}),
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
    assert (problems == []) is valid
    assert {p.loc for p in problems} >= case.output
    if valid:
        assert parse_fn(doc) is doc
    else:
        with pytest.raises(MetadataValidationError):
            parse_fn(doc)


# --- strict from_json validation -------------------------------------------


FROM_JSON_REJECT_PARAMS = [
    pytest.param(
        ArrayMetadataModelV3,
        ExpectFail(lambda: {"zarr_format": 3}, MetadataValidationError, id="x"),
        id="v3-missing-required",
    ),
    pytest.param(
        ArrayMetadataModelV3,
        ExpectFail(_mutate(_build_v3, _set("data_type", 5)), MetadataValidationError, id="x"),
        id="v3-bad-field-type",
    ),
    pytest.param(
        ArrayMetadataModelV2,
        ExpectFail(lambda: {"zarr_format": 2}, MetadataValidationError, id="x"),
        id="v2-missing-required",
    ),
    pytest.param(
        NamedConfigModelV3,
        ExpectFail(lambda: 5, MetadataValidationError, id="x"),
        id="zarr-metadata-bad-input",
    ),
]


@pytest.mark.parametrize(("model", "case"), FROM_JSON_REJECT_PARAMS)
def test_from_json_rejects_malformed(
    model: type[ArrayMetadataModelV3 | ArrayMetadataModelV2 | NamedConfigModelV3],
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
    assert err.problems == problems
    assert "shape: missing required key" in str(err)
    assert "data_type: expected a metadata field" in str(err)


def test_prefix_prepends_loc_head() -> None:
    """_prefix prepends a loc head to each problem's loc."""
    problems = [ValidationProblem(loc=("name",), message="expected str", kind="invalid_type")]
    prefixed = _prefix(0, problems)
    assert prefixed == [
        ValidationProblem(loc=(0, "name"), message="expected str", kind="invalid_type")
    ]


# --- Stricter v2/v3 field validation and error kinds -------------------------


def test_v2_dtype_must_be_string_or_records() -> None:
    """A non-string, non-records v2 dtype is rejected with an invalid_type problem."""
    doc = dict(ArrayMetadataModelV2.create_default().to_json()) | {"dtype": 42}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("dtype",), "invalid_type")]


def test_v2_structured_dtype_records_accepted() -> None:
    """A structured v2 dtype (field records, optionally nested/shaped) validates."""
    dtype = (("a", "<i4"), ("b", (("c", "|u1"),)), ("d", "<f8", (2, 2)))
    doc = dict(ArrayMetadataModelV2.create_default().to_json()) | {"dtype": dtype}
    assert validate_array_metadata_v2(doc) == []


def test_v2_structured_dtype_malformed_record_rejected() -> None:
    """A field record with the wrong arity is rejected."""
    doc = dict(ArrayMetadataModelV2.create_default().to_json()) | {"dtype": (("a",),)}
    problems = validate_array_metadata_v2(doc)
    assert [p.loc for p in problems] == [("dtype",)]


def test_v2_order_literal_enforced() -> None:
    """An order other than 'C' or 'F' is rejected with an invalid_value problem."""
    doc = dict(ArrayMetadataModelV2.create_default().to_json()) | {"order": "Q"}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("order",), "invalid_value")]


def test_v2_compressor_must_be_codec_or_none() -> None:
    """A compressor that is not null or a codec config mapping is rejected."""
    doc = dict(ArrayMetadataModelV2.create_default().to_json()) | {"compressor": "zlib"}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("compressor",), "invalid_type")]


def test_v2_compressor_requires_string_id() -> None:
    """A compressor mapping without a string id is rejected."""
    doc = dict(ArrayMetadataModelV2.create_default().to_json()) | {"compressor": {"level": 3}}
    problems = validate_array_metadata_v2(doc)
    assert [p.loc for p in problems] == [("compressor",)]


def test_v2_filters_must_be_codec_sequence_or_none() -> None:
    """Filters that are not null or a sequence of codec configs are rejected."""
    for bad in (7, (5,), "gzip"):
        doc = dict(ArrayMetadataModelV2.create_default().to_json()) | {"filters": bad}
        problems = validate_array_metadata_v2(doc)
        assert [(p.loc, p.kind) for p in problems] == [(("filters",), "invalid_type")], bad


def test_v2_dimension_separator_literal_enforced() -> None:
    """A dimension_separator other than '.' or '/' is rejected."""
    doc = dict(ArrayMetadataModelV2.create_default().to_json()) | {"dimension_separator": "-"}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("dimension_separator",), "invalid_value")]


def test_v2_zarr_format_literal_enforced() -> None:
    """A v2 document claiming zarr_format 3 is rejected with an invalid_value problem."""
    doc = dict(ArrayMetadataModelV2.create_default().to_json()) | {"zarr_format": 3}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("zarr_format",), "invalid_value")]


def test_v3_zarr_format_literal_enforced() -> None:
    """A v3 document claiming zarr_format 2 is rejected with an invalid_value problem."""
    doc = dict(ArrayMetadataModelV3.create_default().to_json()) | {"zarr_format": 2}
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("zarr_format",), "invalid_value")]


def test_v3_node_type_literal_enforced() -> None:
    """A v3 array document claiming node_type 'group' is rejected."""
    doc = dict(ArrayMetadataModelV3.create_default().to_json()) | {"node_type": "group"}
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("node_type",), "invalid_value")]


def test_missing_key_kind_is_machine_readable() -> None:
    """A missing required key is distinguishable by kind, without message matching."""
    doc = dict(ArrayMetadataModelV3.create_default().to_json())
    del doc["chunk_key_encoding"]
    problems = validate_array_metadata_v3(doc)
    assert problems == [
        ValidationProblem(("chunk_key_encoding",), "missing required key", "missing_key")
    ]


# --- Unified error channels ---------------------------------------------------


def test_from_key_value_invalid_json_raises_metadata_error() -> None:
    """Undecodable store bytes raise MetadataValidationError (kind invalid_json), not JSONDecodeError."""
    with pytest.raises(MetadataValidationError) as exc_info:
        ArrayMetadataModelV3.from_key_value({"zarr.json": b"{not json"})
    assert [p.kind for p in exc_info.value.problems] == ["invalid_json"]


def test_from_key_value_missing_key_kind() -> None:
    """A missing store key surfaces as a missing_key problem at the store-key loc."""
    with pytest.raises(MetadataValidationError) as exc_info:
        ArrayMetadataModelV2.from_key_value({})
    assert exc_info.value.problems == [
        ValidationProblem((".zarray",), "missing store key", "missing_key")
    ]


def test_extra_fields_overlap_raises_metadata_error() -> None:
    """The extra-fields overlap invariant raises MetadataValidationError (a ValueError)."""
    with pytest.raises(MetadataValidationError, match="Extra fields") as exc_info:
        ArrayMetadataModelV3.create_default(extra_fields={"shape": {"must_understand": False}})
    assert [p.kind for p in exc_info.value.problems] == ["invalid_value"]


def test_extension_point_fields_annotated_with_role_alias() -> None:
    """Extension-point fields are annotated with MetadataFieldModelV3 (the
    logical role), not NamedConfigModelV3 (the current serialized form), so a
    future widening of the field union does not move annotation sites."""
    assert MetadataFieldModelV3 is NamedConfigModelV3
    annotations = ArrayMetadataModelV3.__annotations__
    for field_name in ("data_type", "chunk_grid", "chunk_key_encoding"):
        assert annotations[field_name] == "MetadataFieldModelV3"
    for field_name in ("codecs", "storage_transformers"):
        assert annotations[field_name] == "tuple[MetadataFieldModelV3, ...]"


# --- Adversarial-probe fixes: documents that used to pass validation ---------


def test_shape_rejects_json_booleans() -> None:
    """JSON booleans are not integers: shape/chunks containing true/false are
    rejected (bool is an int subclass in Python, so isinstance alone passes)."""
    v3 = dict(ArrayMetadataModelV3.create_default().to_json()) | {"shape": (True, True)}
    assert [p.loc for p in validate_array_metadata_v3(v3)] == [("shape",)]
    v2 = dict(ArrayMetadataModelV2.create_default().to_json()) | {"chunks": (True,)}
    assert [p.loc for p in validate_array_metadata_v2(v2)] == [("chunks",)]


def test_shape_rejects_negative_dimensions() -> None:
    """Dimension lengths must be non-negative; a negative entry is invalid_value."""
    v3 = dict(ArrayMetadataModelV3.create_default().to_json()) | {"shape": (-1,)}
    assert [(p.loc, p.kind) for p in validate_array_metadata_v3(v3)] == [
        (("shape",), "invalid_value")
    ]
    v2 = dict(ArrayMetadataModelV2.create_default().to_json()) | {"chunks": (-5,)}
    assert [(p.loc, p.kind) for p in validate_array_metadata_v2(v2)] == [
        (("chunks",), "invalid_value")
    ]


def test_dimension_names_length_must_match_shape() -> None:
    """dimension_names must have one entry per dimension of shape."""
    doc = dict(ArrayMetadataModelV3.create_default(shape=(10,)).to_json()) | {
        "dimension_names": ("x", "y", "z")
    }
    assert [(p.loc, p.kind) for p in validate_array_metadata_v3(doc)] == [
        (("dimension_names",), "invalid_value")
    ]


def test_attributes_values_must_be_json() -> None:
    """Attribute values are JSON-checked recursively (like fill_value), so a
    non-serializable value is a validation problem, not a later TypeError."""
    doc = dict(ArrayMetadataModelV3.create_default().to_json()) | {"attributes": {"a": {1, 2}}}
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("attributes", "a"), "invalid_type")]


def test_configuration_values_must_be_json() -> None:
    """Configuration values are JSON-checked recursively, so an int-keyed dict
    cannot pass validation and be silently rewritten by json.dumps."""
    doc = dict(ArrayMetadataModelV3.create_default().to_json()) | {
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": {1: 2}}}
    }
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [
        (("chunk_grid", "configuration", "chunk_shape"), "invalid_type")
    ]


# --- must_understand partition (spec: MUST fail to open unrecognized fields) --


def test_must_understand_fields_partition() -> None:
    """must_understand_fields contains every extra field not explicitly waived
    with must_understand: false, including implicitly-true and non-mapping
    fields, so a reader can discharge the spec's fail-to-open duty by
    subtracting the extensions it recognizes."""
    model = ArrayMetadataModelV3.create_default(
        extra_fields={
            "ext_a": {"name": "a", "must_understand": False},
            "ext_b": {"name": "b"},
            "ext_c": {"name": "c", "must_understand": True},
            "ext_d": 123,
        }
    )
    assert set(model.must_understand_fields) == {"ext_b", "ext_c", "ext_d"}
    recognized = {"ext_b"}
    assert model.must_understand_fields.keys() - recognized == {"ext_c", "ext_d"}


def test_must_understand_fields_empty_when_all_waived() -> None:
    """must_understand_fields is empty when every extra field is explicitly waived."""
    model = ArrayMetadataModelV3.create_default(
        extra_fields={"ext_a": {"name": "a", "must_understand": False}}
    )
    assert model.must_understand_fields == {}


def test_dimension_names_null_field_rejected() -> None:
    """A dimension_names field whose VALUE is null is invalid: the spec permits
    null as an element (an unnamed dimension), never as the field value — "not
    specified" is spelled by omitting the key. Consumers bridging from an
    in-memory None sentinel must drop the key, not write null."""
    doc = dict(ArrayMetadataModelV3.create_default().to_json()) | {"dimension_names": None}
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("dimension_names",), "invalid_type")]
    # and the model's own None spelling correctly maps to key absence
    assert (
        "dimension_names"
        not in ArrayMetadataModelV3.create_default(dimension_names=UNSET).to_json()
    )


# --- create_default derives the chunk grid from shape ------------------------


def test_v3_create_default_chunk_grid_follows_shape() -> None:
    """Overriding shape without chunk_grid derives a consistent default grid:
    one chunk covering the array (chunk_shape == shape), instead of silently
    keeping the scalar default's 0-d grid."""
    model = ArrayMetadataModelV3.create_default(shape=(100, 100))
    assert model.chunk_grid == NamedConfigModelV3(
        name="regular", configuration={"chunk_shape": (100, 100)}
    )


def test_v3_create_default_explicit_chunk_grid_respected() -> None:
    """An explicit chunk_grid override wins over the shape-derived default."""
    grid = NamedConfigModelV3(name="regular", configuration={"chunk_shape": (10, 10)})
    model = ArrayMetadataModelV3.create_default(shape=(100, 100), chunk_grid=grid)
    assert model.chunk_grid == grid


def test_v2_create_default_chunks_follow_shape() -> None:
    """Overriding shape without chunks derives chunks == shape."""
    model = ArrayMetadataModelV2.create_default(shape=(100, 100))
    assert model.chunks == (100, 100)


def test_v2_create_default_explicit_chunks_respected() -> None:
    """An explicit chunks override wins over the shape-derived default."""
    model = ArrayMetadataModelV2.create_default(shape=(100, 100), chunks=(10, 10))
    assert model.chunks == (10, 10)


def test_v3_create_default_zero_length_dimensions() -> None:
    """chunk_shape == shape is spec-sound even with zero-length dimensions:
    'The chunk shape elements are non-zero when the corresponding dimensions
    of the arrays have non-zero length' — the constraint is conditional, so a
    zero chunk length is permitted exactly where the dimension is empty."""
    model = ArrayMetadataModelV3.create_default(shape=(0, 3))
    assert model.chunk_grid.configuration["chunk_shape"] == (0, 3)


def test_create_default_derivation_is_one_way() -> None:
    """Overriding the chunk grid (v3) or chunks (v2) without shape leaves the
    scalar default shape=() untouched: a user-supplied chunk_grid is an
    extension point taken verbatim, and deriving shape from it would require
    interpreting grid configurations, which the model layer never does."""
    grid = NamedConfigModelV3(name="regular", configuration={"chunk_shape": (10, 10)})
    v3 = ArrayMetadataModelV3.create_default(chunk_grid=grid)
    assert v3.shape == ()
    assert v3.chunk_grid == grid
    v2 = ArrayMetadataModelV2.create_default(chunks=(10, 10))
    assert v2.shape == ()
    assert v2.chunks == (10, 10)


# --- v2 dimension_separator default (roborev job 426) -------------------------


def test_v2_absent_dimension_separator_means_dot() -> None:
    """A .zarray that omits dimension_separator uses the v2 convention default
    '.', not '/': chunk keys of real-world default-separator v2 arrays look
    like '0.0'. The model normalizes the absent key to an explicit '.' — a
    semantics-preserving spelling normalization."""
    doc = dict(ArrayMetadataModelV2.create_default().to_json())
    del doc["dimension_separator"]
    model = ArrayMetadataModelV2.from_json(doc)
    assert model.dimension_separator == "."
    assert model.to_json()["dimension_separator"] == "."


def test_v2_from_key_value_without_separator_means_dot() -> None:
    """The .zarray store-file path applies the same '.' default for an absent
    dimension_separator key."""
    doc = {
        k: v
        for k, v in ArrayMetadataModelV2.create_default().to_json().items()
        if k not in ("dimension_separator", "attributes")
    }
    import json as _json

    model = ArrayMetadataModelV2.from_key_value({".zarray": _json.dumps(doc).encode()})
    assert model.dimension_separator == "."


def test_v2_null_dimension_separator_rejected() -> None:
    """dimension_separator may be absent, '.', or '/' — never null: the
    document grammar has no null spelling for this field."""
    doc = dict(ArrayMetadataModelV2.create_default().to_json()) | {"dimension_separator": None}
    problems = validate_array_metadata_v2(doc)
    assert [(p.loc, p.kind) for p in problems] == [(("dimension_separator",), "invalid_value")]


def test_dimension_names_absent_and_all_null_are_distinct() -> None:
    """An absent dimension_names field and an explicit all-null one are
    semantically different documents: the explicit form says every dimension
    has a name, which is null; absence says there are no dimension names.
    The model preserves the distinction (UNSET vs a tuple of Nones), and both
    spellings round-trip faithfully."""
    absent_doc = dict(ArrayMetadataModelV3.create_default(shape=(2, 3)).to_json())
    explicit_doc = absent_doc | {"dimension_names": (None, None)}

    absent = ArrayMetadataModelV3.from_json(absent_doc)
    explicit = ArrayMetadataModelV3.from_json(explicit_doc)

    assert absent.dimension_names is UNSET
    assert explicit.dimension_names == (None, None)
    assert absent != explicit
    assert "dimension_names" not in absent.to_json()
    assert absent.to_json() == absent_doc
    assert explicit.to_json() == explicit_doc
